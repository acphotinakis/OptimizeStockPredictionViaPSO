"""Expanding-window walk-forward validation for the LSTM/IPSO pipeline.

For each fold:
    1. Slice the raw train and val arrays (expanding train, fixed-size val).
    2. Fit fresh feature and target ``FrozenMinMaxScaler`` instances on the
       train slice, then transform both slices.
    3. Build LSTM windows on scaled data using the canonical 1-bar-ahead
       alignment from ``src/data/windowing.py``.
    4. Optionally run IPSO on a chronological 90/10 split of the fold's train
       sequences; otherwise read baseline hyperparameters from
       ``config.lstm_baseline``.
    5. Train a fresh LSTM with the chosen hyperparameters via ``LSTMTrainer``.
       Predict on the val sequences.
    6. Inverse-transform predictions and ground-truth and compute statistical
       metrics on the original return scale.
    7. Aggregate per-fold metrics across folds.

The module exports:
    - ``ExpandingWindowWalkForward``: validator class with
      ``.validate(X_raw, y_raw, timestamps=None)``.
    - ``validate_walk_forward_compliance``: post-hoc sanity checks on the
      returned ``fold_boundaries``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.windowing import build_lstm_windows
from src.evaluation.metrics import compute_and_log_all_statistical_metrics
from src.features.scaler import FrozenMinMaxScaler
from src.models import LSTMModel, LSTMTrainer
from src.optimizer import IPSO, SpecCompliantFitness

logger = logging.getLogger(__name__)


_BASELINE_DEFAULT: Dict[str, Any] = {
    "lstm_units_1": 128,
    "lstm_units_2": 64,
    "dropout_rate": 0.2,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "epochs": 100,
}


_METRIC_KEYS: Tuple[str, ...] = (
    "rmse",
    "mae",
    "mape",
    "r2",
    "directional_accuracy",
)


class ExpandingWindowWalkForward:
    """Expanding-window walk-forward validator with optional per-fold IPSO.

    Args:
        initial_train_pct: Initial train window as a fraction of total data.
        fold_step_pct: Increment to the train window per fold.
        val_pct: Validation window size per fold.
        lookback: LSTM lookback window (default 20).
        config: Project ``Config`` object. Required when ``run_pso`` is True
            (the ``pso`` block is read for IPSO settings) and used as the
            source of baseline hyperparameters when ``run_pso`` is False.
        run_pso: When True, run IPSO per fold to choose hyperparameters.
        max_folds: Cap on the number of folds; ``0`` disables the cap.
    """

    def __init__(
        self,
        initial_train_pct: float = 0.60,
        fold_step_pct: float = 0.05,
        val_pct: float = 0.05,
        lookback: int = 20,
        config: Any = None,
        run_pso: bool = False,
        max_folds: int = 0,
    ) -> None:
        if not 0.0 < initial_train_pct < 1.0:
            raise ValueError("initial_train_pct must be in (0, 1)")
        if not 0.0 < fold_step_pct <= 1.0:
            raise ValueError("fold_step_pct must be in (0, 1]")
        if not 0.0 < val_pct < 1.0:
            raise ValueError("val_pct must be in (0, 1)")
        if lookback < 1:
            raise ValueError("lookback must be >= 1")

        self.initial_train_pct = initial_train_pct
        self.fold_step_pct = fold_step_pct
        self.val_pct = val_pct
        self.lookback = lookback
        self.config = config
        self.run_pso = run_pso
        self.max_folds = max_folds

        if run_pso:
            logger.warning(
                "Per-fold IPSO is enabled; expect significant runtime cost "
                "(n_particles x n_iterations LSTM trainings per fold)."
            )

    def validate(
        self,
        X_raw: np.ndarray,
        y_raw: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None,
    ) -> Dict[str, Any]:
        """Run the expanding-window walk-forward loop.

        Args:
            X_raw: Unscaled feature matrix, shape ``(N, F)``.
            y_raw: Unscaled target vector, shape ``(N,)``.
            timestamps: Optional matching ``DatetimeIndex`` for diagnostic
                output; passed through to per-fold prediction payloads.

        Returns:
            Dict with keys ``aggregated_metrics``, ``fold_results``,
            ``config``, and ``fold_boundaries``.
        """
        if X_raw.ndim != 2:
            raise ValueError(f"X_raw must be 2D (N, F), got shape {X_raw.shape}")
        if y_raw.ndim != 1:
            raise ValueError(f"y_raw must be 1D (N,), got shape {y_raw.shape}")
        if len(X_raw) != len(y_raw):
            raise ValueError(
                f"length mismatch: X_raw={len(X_raw)} y_raw={len(y_raw)}"
            )

        n = len(X_raw)
        boundaries = self._compute_fold_boundaries(n)

        fold_results: List[Dict[str, Any]] = []
        for fold_idx, (ts, te, vs, ve) in enumerate(boundaries):
            logger.info(
                "Fold %d: train [%d, %d) val [%d, %d)",
                fold_idx,
                ts,
                te,
                vs,
                ve,
            )
            fold_results.append(
                self._process_fold(
                    fold_idx,
                    ts,
                    te,
                    vs,
                    ve,
                    X_raw,
                    y_raw,
                    timestamps,
                )
            )

        aggregated = self._aggregate(fold_results)

        return {
            "aggregated_metrics": aggregated,
            "fold_results": fold_results,
            "config": self._config_snapshot(),
            "fold_boundaries": boundaries,
        }

    def _compute_fold_boundaries(
        self, n: int
    ) -> List[Tuple[int, int, int, int]]:
        initial_size = int(n * self.initial_train_pct)
        step_size = max(1, int(n * self.fold_step_pct))
        val_size = max(1, int(n * self.val_pct))

        if initial_size <= self.lookback:
            raise ValueError(
                f"initial train window ({initial_size}) is not larger than "
                f"lookback ({self.lookback})"
            )
        if initial_size + val_size > n:
            raise ValueError(
                f"initial train + val exceeds dataset; n={n} "
                f"initial={initial_size} val={val_size}"
            )

        boundaries: List[Tuple[int, int, int, int]] = []
        train_end = initial_size
        while train_end + val_size <= n:
            boundaries.append((0, train_end, train_end, train_end + val_size))
            train_end += step_size
            if self.max_folds and len(boundaries) >= self.max_folds:
                break

        if not boundaries:
            raise ValueError(
                "no valid folds; check window sizes vs. data length"
            )

        return boundaries

    def _process_fold(
        self,
        fold_idx: int,
        ts: int,
        te: int,
        vs: int,
        ve: int,
        X_raw: np.ndarray,
        y_raw: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex],
    ) -> Dict[str, Any]:
        X_train = X_raw[ts:te]
        y_train = y_raw[ts:te]
        X_val = X_raw[vs:ve]
        y_val = y_raw[vs:ve]

        feature_scaler = FrozenMinMaxScaler(feature_range=(-1.0, 1.0))
        feature_scaler.fit(X_train)
        X_train_s = feature_scaler.transform(X_train)
        X_val_s = feature_scaler.transform(X_val)

        target_scaler = FrozenMinMaxScaler(feature_range=(-1.0, 1.0))
        target_scaler.fit(y_train.reshape(-1, 1))
        y_train_s = target_scaler.transform(y_train.reshape(-1, 1)).ravel()
        y_val_s = target_scaler.transform(y_val.reshape(-1, 1)).ravel()

        X_train_seq, y_train_seq = build_lstm_windows(
            X_train_s, y_train_s, self.lookback
        )
        X_val_seq, y_val_seq = build_lstm_windows(
            X_val_s, y_val_s, self.lookback
        )

        if len(X_val_seq) == 0:
            raise ValueError(
                f"Fold {fold_idx}: val window is too small for the lookback"
            )

        seed = self._fold_seed(fold_idx)
        if self.run_pso:
            best_params, pso_fitness = self._run_pso(
                X_train_seq, y_train_seq, fold_idx, seed
            )
        else:
            best_params = self._baseline_params()
            pso_fitness = float("nan")

        trained_wrapper, _ = self._build_and_train(
            best_params, X_train_seq, y_train_seq, seed=seed
        )

        y_pred_s = trained_wrapper.predict(X_val_seq)
        y_pred = target_scaler.inverse_transform(
            np.asarray(y_pred_s, dtype=np.float32).reshape(-1, 1)
        ).ravel()
        y_true = target_scaler.inverse_transform(
            y_val_seq.reshape(-1, 1)
        ).ravel()

        metrics = compute_and_log_all_statistical_metrics(
            y_true=y_true,
            y_pred=y_pred,
            label=f"Fold {fold_idx}",
        )
        metrics["n_samples"] = int(len(y_true))

        prediction_payload: Dict[str, Any] = {
            "y_pred": y_pred.tolist(),
            "y_true": y_true.tolist(),
        }
        if timestamps is not None:
            val_timestamps = timestamps[vs:ve]
            aligned = val_timestamps[self.lookback - 1 : -1]
            prediction_payload["timestamps"] = [str(t) for t in aligned]

        return {
            "fold_idx": fold_idx,
            "train_size": int(te - ts),
            "val_size": int(ve - vs),
            "train_indices": [int(ts), int(te)],
            "val_indices": [int(vs), int(ve)],
            "pso_params": best_params,
            "pso_fitness": float(pso_fitness),
            "metrics": metrics,
            "predictions": prediction_payload,
            "scalers": {
                "feature_scaler_params": _scaler_to_jsonable(
                    feature_scaler.get_params()
                ),
                "target_scaler_params": _scaler_to_jsonable(
                    target_scaler.get_params()
                ),
            },
        }

    def _baseline_params(self) -> Dict[str, Any]:
        if self.config is None:
            return dict(_BASELINE_DEFAULT)
        baseline = self.config.lstm_baseline
        return {
            "lstm_units_1": int(baseline.lstm_units_1),
            "lstm_units_2": int(baseline.lstm_units_2),
            "dropout_rate": float(baseline.dropout_rate),
            "learning_rate": float(baseline.learning_rate),
            "batch_size": int(baseline.batch_size),
            "epochs": int(baseline.epochs),
        }

    def _run_pso(
        self,
        X_seq: np.ndarray,
        y_seq: np.ndarray,
        fold_idx: int,
        seed: int,
    ) -> Tuple[Dict[str, Any], float]:
        if self.config is None:
            raise RuntimeError(
                "run_pso=True requires a config with a `pso` block"
            )

        pso_config = self.config.pso

        split = max(1, int(len(X_seq) * 0.9))
        if split >= len(X_seq):
            raise ValueError(
                f"Fold {fold_idx}: not enough train sequences ({len(X_seq)}) "
                "for the 90/10 PSO split"
            )

        X_pso_tr = X_seq[:split]
        y_pso_tr = y_seq[:split]
        X_pso_vl = X_seq[split:]
        y_pso_vl = y_seq[split:]

        captured_seed = seed

        def model_builder(
            params: Dict[str, Any],
            X_tr: np.ndarray,
            y_tr: np.ndarray,
            X_vl: np.ndarray,
            y_vl: np.ndarray,
        ) -> Tuple[np.ndarray, Any]:
            architecture = {
                "input_size": X_tr.shape[2],
                "lstm_units_1": int(params["units_1"]),
                "lstm_units_2": int(params["units_2"]),
                "dropout_rate": float(params["dropout"]),
                "output_units": 1,
                "activation": "relu",
                "output_activation": "linear",
            }
            trainer_cfg = {
                "learning_rate": float(params["learning_rate"]),
                "epochs": int(params["epochs"]),
                "batch_size": int(params["batch_size"]),
                "optimizer": "adam",
                "loss": "mse",
                "shuffle": False,
                "grad_clip": 1.0,
                "use_amp": False,
                "accumulation_steps": 1,
                "early_stopping": {
                    "enabled": True,
                    "monitor": "val_loss",
                    "patience": 10,
                    "restore_best_weights": True,
                },
            }

            wrapper = LSTMModel(seed=captured_seed)
            wrapper.build_model(architecture)
            trainer = LSTMTrainer(
                lstm_model=wrapper, config=trainer_cfg, seed=captured_seed
            )
            trained, _ = trainer.train(X_tr, y_tr, X_vl, y_vl)
            return trained.predict(X_vl), trained.model

        optimizer = IPSO(
            n_particles=pso_config.n_particles,
            n_iterations=pso_config.n_iterations,
            fitness_fn=SpecCompliantFitness(gamma=pso_config.fitness.gamma),
            model_builder=model_builder,
            w_max=pso_config.inertia_max,
            w_min=pso_config.inertia_min,
            c1=pso_config.c1,
            c2=pso_config.c2,
            v_clamp_fraction=pso_config.v_clamp_fraction,
            seed=seed,
        )

        best_params_raw, best_fitness = optimizer.run(
            X_pso_tr, y_pso_tr, X_pso_vl, y_pso_vl
        )

        best_params = {
            "lstm_units_1": int(best_params_raw["units_1"]),
            "lstm_units_2": int(best_params_raw["units_2"]),
            "dropout_rate": float(best_params_raw["dropout"]),
            "learning_rate": float(best_params_raw["learning_rate"]),
            "batch_size": int(best_params_raw["batch_size"]),
            "epochs": int(best_params_raw["epochs"]),
        }

        return best_params, float(best_fitness)

    def _build_and_train(
        self,
        params: Dict[str, Any],
        X_seq: np.ndarray,
        y_seq: np.ndarray,
        seed: int,
    ) -> Tuple[LSTMModel, Dict[str, Any]]:
        # LSTMTrainer requires non-empty val arrays; carve a chronological
        # tail slice purely to satisfy its input validation. Early stopping
        # remains enabled so the fold's reported model uses the best
        # checkpoint found on that internal val slice.
        split = max(1, int(len(X_seq) * 0.9))
        X_tr, X_vl = X_seq[:split], X_seq[split:]
        y_tr, y_vl = y_seq[:split], y_seq[split:]
        if len(X_vl) == 0:
            raise ValueError(
                f"Fold has too few train sequences ({len(X_seq)}) for an "
                "internal 90/10 split"
            )

        architecture = {
            "input_size": X_seq.shape[2],
            "lstm_units_1": int(params["lstm_units_1"]),
            "lstm_units_2": int(params["lstm_units_2"]),
            "dropout_rate": float(params["dropout_rate"]),
            "output_units": 1,
            "activation": "relu",
            "output_activation": "linear",
        }
        trainer_cfg = {
            "learning_rate": float(params["learning_rate"]),
            "epochs": int(params["epochs"]),
            "batch_size": int(params["batch_size"]),
            "optimizer": "adam",
            "loss": "mse",
            "shuffle": False,
            "grad_clip": 1.0,
            "use_amp": False,
            "accumulation_steps": 1,
            "early_stopping": {
                "enabled": True,
                "monitor": "val_loss",
                "patience": 10,
                "restore_best_weights": True,
            },
        }

        wrapper = LSTMModel(seed=seed)
        wrapper.build_model(architecture)
        trainer = LSTMTrainer(
            lstm_model=wrapper, config=trainer_cfg, seed=seed
        )
        trained, artifacts = trainer.train(X_tr, y_tr, X_vl, y_vl)
        return trained, artifacts

    def _fold_seed(self, fold_idx: int) -> int:
        base = (
            self.config.lstm_baseline.random_seed
            if self.config is not None
            else 42
        )
        return int(base) + int(fold_idx)

    def _aggregate(self, folds: List[Dict[str, Any]]) -> Dict[str, Any]:
        n_folds = len(folds)
        if n_folds == 0:
            return {"n_folds": 0, "total_predictions": 0}

        total_predictions = sum(f["metrics"]["n_samples"] for f in folds)

        agg: Dict[str, Any] = {
            "n_folds": int(n_folds),
            "total_predictions": int(total_predictions),
        }
        for key in _METRIC_KEYS:
            raw = [f["metrics"].get(key, float("nan")) for f in folds]
            values = np.array(
                [v for v in raw if v is not None and not np.isnan(v)],
                dtype=float,
            )
            if values.size == 0:
                stats = {
                    f"{key}_mean": float("nan"),
                    f"{key}_std": float("nan"),
                    f"{key}_min": float("nan"),
                    f"{key}_max": float("nan"),
                    f"{key}_median": float("nan"),
                }
            else:
                stats = {
                    f"{key}_mean": float(values.mean()),
                    f"{key}_std": float(values.std(ddof=0)),
                    f"{key}_min": float(values.min()),
                    f"{key}_max": float(values.max()),
                    f"{key}_median": float(np.median(values)),
                }
            agg.update(stats)
        return agg

    def _config_snapshot(self) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {
            "initial_train_pct": self.initial_train_pct,
            "fold_step_pct": self.fold_step_pct,
            "val_pct": self.val_pct,
            "lookback": self.lookback,
            "run_pso": self.run_pso,
            "max_folds": self.max_folds,
        }
        if self.config is not None:
            snapshot["lstm_baseline_seed"] = int(
                self.config.lstm_baseline.random_seed
            )
        return snapshot


def validate_walk_forward_compliance(
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    fold_boundaries: List[Tuple[int, int, int, int]],
) -> None:
    """Sanity-check fold boundaries against the canonical walk-forward rules.

    Args:
        X_raw: Raw feature matrix.
        y_raw: Raw target vector.
        fold_boundaries: List of ``(train_start, train_end, val_start, val_end)``.

    Raises:
        AssertionError: If any fold violates the rules.
    """
    n = len(X_raw)
    assert n == len(y_raw), "X_raw and y_raw length mismatch"
    assert fold_boundaries, "no folds to validate"

    prev_train_end = -1
    for i, (ts, te, vs, ve) in enumerate(fold_boundaries):
        assert 0 <= ts < te <= n, f"fold {i}: bad train range [{ts}, {te})"
        assert te <= vs < ve <= n, f"fold {i}: bad val range [{vs}, {ve})"
        assert (te - ts) >= 20, (
            f"fold {i}: insufficient train data ({te - ts} < 20)"
        )
        assert (ve - vs) >= 1, (
            f"fold {i}: insufficient val data ({ve - vs} < 1)"
        )
        assert te > prev_train_end, (
            f"fold {i}: train_end {te} not strictly increasing "
            f"(previous {prev_train_end})"
        )
        prev_train_end = te

    logger.info(
        "Walk-forward compliance OK: %d folds, n=%d",
        len(fold_boundaries),
        n,
    )


def _scaler_to_jsonable(params: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(params)
    if "feature_range" in out and not isinstance(out["feature_range"], list):
        out["feature_range"] = list(out["feature_range"])
    return out
