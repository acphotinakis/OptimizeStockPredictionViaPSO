from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import math
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn

from ..utils.ou_runtime import resolve_device, set_global_seed
from ..evaluation.ou_metrics import evaluate_forecast, rmse
from ..data.yfinance_data import train_test_split_series, fit_scaler_on_train, transform_series, make_supervised

@dataclass
class LSTMHyperparams:
    epochs: int
    node1: int
    node2: int
    learning_rate: float
    batch_size: int = 64

    def clipped(self) -> "LSTMHyperparams":
        return LSTMHyperparams(
            epochs=int(np.clip(round(self.epochs), 1, 200)),
            node1=int(np.clip(round(self.node1), 1, 200)),
            node2=int(np.clip(round(self.node2), 1, 200)),
            learning_rate=float(np.clip(self.learning_rate, 0.001, 0.01)),
            batch_size=self.batch_size,
        )

    def as_vector(self) -> np.ndarray:
        return np.array([self.epochs, self.node1, self.node2, self.learning_rate], dtype=np.float64)

    @staticmethod
    def from_vector(v: Sequence[float], batch_size: int = 64) -> "LSTMHyperparams":
        return LSTMHyperparams(int(v[0]), int(v[1]), int(v[2]), float(v[3]), batch_size=batch_size).clipped()


class TorchLSTMRegressor(nn.Module):
    """Two-layer LSTM matching the original Keras structure.

    Keras original:
        LSTM(node1, return_sequences=True)
        LSTM(node2)
        Dense(1)

    PyTorch equivalent here uses two separate nn.LSTM modules so the first and
    second recurrent layers can have different hidden sizes.
    """

    def __init__(self, input_size: int, hidden1: int, hidden2: int):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden1, hidden_size=hidden2, batch_first=True)
        self.fc = nn.Linear(hidden2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        last = out[:, -1, :]
        return self.fc(last)


class TorchRegressorFactory:
    def __init__(
        self,
        lookback: int,
        batch_size: int = 64,
        verbose: int = 0,
        device: Optional[torch.device] = None,
        amp: bool = False,
    ):
        self.lookback = lookback
        self.batch_size = batch_size
        self.verbose = verbose
        self.device = device or resolve_device("auto")
        self.amp = bool(amp and self.device.type == "cuda")

    def build(self, hp: LSTMHyperparams) -> nn.Module:
        hp = hp.clipped()
        return TorchLSTMRegressor(input_size=1, hidden1=hp.node1, hidden2=hp.node2).to(self.device)

    def fit_predict(
        self,
        hp: LSTMHyperparams,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        scaler: MinMaxScaler,
        seed: int,
        use_early_stopping: bool = False,
    ) -> Tuple[np.ndarray, nn.Module, Dict[str, List[float]]]:
        hp = hp.clipped()
        set_global_seed(seed)

        model = self.build(hp)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)

        # Keep the whole training set on the selected device.  For these daily index datasets
        # this avoids repeated CPU -> GPU transfers inside the PSO/IPSO fitness loop.
        X_tensor = torch.as_tensor(X_train, dtype=torch.float32, device=self.device)
        y_tensor = torch.as_tensor(y_train, dtype=torch.float32, device=self.device)

        history: Dict[str, List[float]] = {"loss": []}
        best_loss = float("inf")
        best_state = None
        bad_epochs = 0
        patience = 8

        for epoch in range(1, hp.epochs + 1):
            model.train()
            total_loss = 0.0
            total_count = 0
            n_samples = X_tensor.shape[0]
            for start_idx in range(0, n_samples, hp.batch_size):
                xb = X_tensor[start_idx:start_idx + hp.batch_size]
                yb = y_tensor[start_idx:start_idx + hp.batch_size]

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp):
                    pred = model(xb)
                    loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

                batch_count = xb.shape[0]
                total_loss += float(loss.detach().cpu().item()) * batch_count
                total_count += batch_count

            epoch_loss = total_loss / max(total_count, 1)
            history["loss"].append(epoch_loss)
            if self.verbose:
                print(f"epoch={epoch:03d} loss={epoch_loss:.8f}")

            if use_early_stopping:
                if epoch_loss + 1e-12 < best_loss:
                    best_loss = epoch_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        break

        if use_early_stopping and best_state is not None:
            model.load_state_dict(best_state)
            model.to(self.device)

        model.eval()
        with torch.no_grad():
            Xv = torch.as_tensor(X_val, dtype=torch.float32, device=self.device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp):
                pred_scaled = model(Xv).detach().float().cpu().numpy()
        pred = scaler.inverse_transform(pred_scaled)
        return pred.reshape(-1), model, history


@dataclass
class SearchBounds:
    low: np.ndarray
    high: np.ndarray

    @staticmethod
    def default() -> "SearchBounds":
        return SearchBounds(
            low=np.array([1, 1, 1, 0.001], dtype=np.float64),
            high=np.array([200, 200, 200, 0.01], dtype=np.float64),
        )


@dataclass
class Particle:
    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray
    best_fitness: float


@dataclass
class SearchResult:
    best_hparams: LSTMHyperparams
    best_fitness: float
    history: List[Dict[str, float]]


class PSOBase:
    def __init__(
        self,
        fitness_fn: Callable[[LSTMHyperparams, int], float],
        bounds: SearchBounds,
        swarm_size: int = 10,
        max_iter: int = 30,
        c1: float = 1.5,
        c2: float = 1.5,
        w: float = 0.6,
        seed: int = 42,
        batch_size: int = 64,
    ):
        self.fitness_fn = fitness_fn
        self.bounds = bounds
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.batch_size = batch_size

    def _clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.bounds.low, self.bounds.high)

    def _random_position(self) -> np.ndarray:
        return self.rng.uniform(self.bounds.low, self.bounds.high)

    def _random_velocity(self) -> np.ndarray:
        span = self.bounds.high - self.bounds.low
        return self.rng.uniform(-0.1 * span, 0.1 * span)

    def _hp(self, pos: np.ndarray) -> LSTMHyperparams:
        return LSTMHyperparams.from_vector(pos, batch_size=self.batch_size)

    def _init_particles(self) -> List[Particle]:
        particles: List[Particle] = []
        for idx in range(self.swarm_size):
            pos = self._random_position()
            vel = self._random_velocity()
            fit = self.fitness_fn(self._hp(pos), self.seed + idx)
            particles.append(Particle(pos.copy(), vel.copy(), pos.copy(), fit))
        return particles

    def run(self) -> SearchResult:
        particles = self._init_particles()
        gbest = min(particles, key=lambda p: p.best_fitness)
        gbest_position = gbest.best_position.copy()
        gbest_fitness = float(gbest.best_fitness)
        history: List[Dict[str, float]] = []

        for t in range(1, self.max_iter + 1):
            for i, p in enumerate(particles):
                r1 = self.rng.random(size=p.position.shape)
                r2 = self.rng.random(size=p.position.shape)
                p.velocity = (
                    self.w * p.velocity
                    + self.c1 * r1 * (p.best_position - p.position)
                    + self.c2 * r2 * (gbest_position - p.position)
                )
                p.position = self._clip(p.position + p.velocity)

                fit = self.fitness_fn(self._hp(p.position), self.seed + 1000 * t + i)
                if fit < p.best_fitness:
                    p.best_fitness = fit
                    p.best_position = p.position.copy()
                if fit < gbest_fitness:
                    gbest_fitness = fit
                    gbest_position = p.position.copy()

            history.append({"iter": t, "best_fitness": gbest_fitness})
            print(f"iter={t:03d}/{self.max_iter:03d} best_RMSE={gbest_fitness:.6f}")

        return SearchResult(self._hp(gbest_position), gbest_fitness, history)


class IPSO(PSOBase):
    def __init__(
        self,
        fitness_fn: Callable[[LSTMHyperparams, int], float],
        bounds: SearchBounds,
        swarm_size: int = 10,
        max_iter: int = 30,
        c1: float = 1.5,
        c2: float = 1.5,
        w_min: float = 0.6,
        w_max: float = 0.8,
        seed: int = 42,
        batch_size: int = 64,
    ):
        super().__init__(fitness_fn, bounds, swarm_size, max_iter, c1, c2, w_max, seed, batch_size=batch_size)
        self.w_min = w_min
        self.w_max = w_max

    def inertia_weight(self, t: int) -> float:
        return float(self.w_max - (self.w_max - self.w_min) * math.tanh((math.pi / 4.0) * (t / self.max_iter)))

    def alpha_mf(self, t: int) -> float:
        return float(0.3 * (t / self.max_iter) + 0.7)

    def mutate_particle(self, p: Particle) -> None:
        p.position = self._random_position()
        p.velocity = self._random_velocity()

    def run(self) -> SearchResult:
        particles = self._init_particles()
        gbest = min(particles, key=lambda p: p.best_fitness)
        gbest_position = gbest.best_position.copy()
        gbest_fitness = float(gbest.best_fitness)
        history: List[Dict[str, float]] = []

        for t in range(1, self.max_iter + 1):
            w_t = self.inertia_weight(t)
            a_mf = self.alpha_mf(t)

            for i, p in enumerate(particles):
                r1 = self.rng.random(size=p.position.shape)
                r2 = self.rng.random(size=p.position.shape)
                p.velocity = (
                    w_t * p.velocity
                    + self.c1 * r1 * (p.best_position - p.position)
                    + self.c2 * r2 * (gbest_position - p.position)
                )
                p.position = self._clip(p.position + p.velocity)

                alpha = float(self.rng.random())
                if alpha > a_mf:
                    self.mutate_particle(p)

                fit = self.fitness_fn(self._hp(p.position), self.seed + 1000 * t + i)
                if fit < p.best_fitness:
                    p.best_fitness = fit
                    p.best_position = p.position.copy()
                if fit < gbest_fitness:
                    gbest_fitness = fit
                    gbest_position = p.position.copy()

            history.append({"iter": t, "best_fitness": gbest_fitness, "w_t": w_t, "alpha_mf": a_mf})
            print(f"iter={t:03d}/{self.max_iter:03d} best_RMSE={gbest_fitness:.6f} w_t={w_t:.4f} alpha_mf={a_mf:.4f}")

        return SearchResult(self._hp(gbest_position), gbest_fitness, history)


@dataclass
class ForecastArtifacts:
    metrics: Dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray
    best_hparams: Optional[LSTMHyperparams]
    search_history: List[Dict[str, float]]
    training_history: Dict[str, List[float]]


def prepare_windows(series: pd.Series, lookback: int) -> Tuple[MinMaxScaler, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_series, test_series = train_test_split_series(series, train_ratio=0.7)
    scaler = fit_scaler_on_train(train_series)

    train_scaled = transform_series(scaler, train_series)
    combined_for_test = pd.concat([train_series.iloc[-lookback:], test_series])
    test_scaled_full = transform_series(scaler, combined_for_test)

    X_train, y_train = make_supervised(train_scaled, lookback)
    X_test, y_test_scaled = make_supervised(test_scaled_full, lookback)
    y_true = scaler.inverse_transform(y_test_scaled).reshape(-1)
    return scaler, X_train, y_train, X_test, y_test_scaled, y_true


def run_baseline_lstm(
    series: pd.Series,
    lookback: int,
    hp: LSTMHyperparams,
    seed: int,
    verbose: int = 0,
    device: Optional[torch.device] = None,
    amp: bool = False,
) -> ForecastArtifacts:
    scaler, X_train, y_train, X_test, _, y_true = prepare_windows(series, lookback)
    factory = TorchRegressorFactory(lookback=lookback, batch_size=hp.batch_size, verbose=verbose, device=device, amp=amp)
    pred, _, training_history = factory.fit_predict(hp, X_train, y_train, X_test, scaler=scaler, seed=seed)
    metrics = evaluate_forecast(y_true, pred)
    return ForecastArtifacts(metrics, y_true, pred, hp, [], training_history)


def make_fitness_function(
    series: pd.Series,
    lookback: int,
    batch_size: int,
    verbose: int = 0,
    device: Optional[torch.device] = None,
    amp: bool = False,
) -> Callable[[LSTMHyperparams, int], float]:
    scaler, X_train, y_train, X_test, _, y_true = prepare_windows(series, lookback)
    factory = TorchRegressorFactory(lookback=lookback, batch_size=batch_size, verbose=verbose, device=device, amp=amp)
    cache: Dict[Tuple[int, int, int, float], float] = {}

    def fitness(hp: LSTMHyperparams, seed: int) -> float:
        hp = hp.clipped()
        key = (hp.epochs, hp.node1, hp.node2, round(hp.learning_rate, 6))
        if key in cache:
            return cache[key]
        pred, _, _ = factory.fit_predict(hp, X_train, y_train, X_test, scaler=scaler, seed=seed)
        value = rmse(y_true, pred)
        cache[key] = value
        return value

    return fitness


def run_search_model(
    series: pd.Series,
    lookback: int,
    method: str,
    search_iters: int,
    swarm_size: int,
    batch_size: int,
    seed: int,
    verbose: int = 0,
    device: Optional[torch.device] = None,
    amp: bool = False,
) -> ForecastArtifacts:
    fitness_fn = make_fitness_function(series, lookback, batch_size=batch_size, verbose=verbose, device=device, amp=amp)
    bounds = SearchBounds.default()

    if method.lower() == "pso":
        searcher = PSOBase(fitness_fn, bounds, swarm_size, search_iters, c1=1.5, c2=1.5, w=0.6, seed=seed, batch_size=batch_size)
    elif method.lower() == "ipso":
        searcher = IPSO(fitness_fn, bounds, swarm_size, search_iters, c1=1.5, c2=1.5, w_min=0.6, w_max=0.8, seed=seed, batch_size=batch_size)
    else:
        raise ValueError(f"Unsupported method: {method}")

    search_result = searcher.run()
    best_hp = search_result.best_hparams.clipped()
    artifacts = run_baseline_lstm(series, lookback, best_hp, seed=seed, verbose=verbose, device=device, amp=amp)
    artifacts.best_hparams = best_hp
    artifacts.search_history = search_result.history
    return artifacts
