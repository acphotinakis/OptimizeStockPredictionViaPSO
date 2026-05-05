from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler

from ..utils.ou_runtime import resolve_device, set_global_seed
from ..evaluation.ou_metrics import rmse, evaluate_forecast as _evaluate_forecast
from ..data.alpaca_config import TARGET_COL
from ..data.alpaca_data import prepare_windows

def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return _evaluate_forecast(y_true, y_pred, include_mse=True)

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
            learning_rate=float(np.clip(self.learning_rate, 0.0001, 0.05)),
            batch_size=int(np.clip(round(self.batch_size), 1, 2048)),
        )

    @staticmethod
    def from_vector(v: Sequence[float], batch_size: int = 64) -> "LSTMHyperparams":
        return LSTMHyperparams(int(v[0]), int(v[1]), int(v[2]), float(v[3]), batch_size=batch_size).clipped()


class TorchLSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden1: int, hidden2: int):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden1, hidden_size=hidden2, batch_first=True)
        self.fc = nn.Linear(hidden2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        return self.fc(out[:, -1, :])


class TorchRegressorFactory:
    def __init__(self, n_features: int, verbose: int = 0, device: Optional[torch.device] = None, amp: bool = False):
        self.n_features = n_features
        self.verbose = verbose
        self.device = device or resolve_device("auto")
        self.amp = bool(amp and self.device.type == "cuda")

    def build(self, hp: LSTMHyperparams) -> nn.Module:
        hp = hp.clipped()
        return TorchLSTMRegressor(input_size=self.n_features, hidden1=hp.node1, hidden2=hp.node2).to(self.device)

    def fit_predict(
        self,
        hp: LSTMHyperparams,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        target_scaler: MinMaxScaler,
        seed: int,
        use_early_stopping: bool = False,
    ) -> Tuple[np.ndarray, nn.Module, Dict[str, List[float]]]:
        hp = hp.clipped()
        set_global_seed(seed)
        model = self.build(hp)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)

        X_tensor = torch.as_tensor(X_train, dtype=torch.float32, device=self.device)
        y_tensor = torch.as_tensor(y_train, dtype=torch.float32, device=self.device)

        history: Dict[str, List[float]] = {"loss": []}
        best_loss = float("inf")
        best_state = None
        bad_epochs = 0
        patience = 8

        for epoch in range(1, hp.epochs + 1):
            model.train()
            # Shuffle mini-batches to make training less order-sensitive.
            order = torch.randperm(X_tensor.shape[0], device=self.device)
            total_loss = 0.0
            total_count = 0
            for start_idx in range(0, X_tensor.shape[0], hp.batch_size):
                idx = order[start_idx:start_idx + hp.batch_size]
                xb = X_tensor[idx]
                yb = y_tensor[idx]
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
        pred = target_scaler.inverse_transform(pred_scaled).reshape(-1)
        return pred, model, history


@dataclass
class SearchBounds:
    low: np.ndarray
    high: np.ndarray

    @staticmethod
    def default() -> "SearchBounds":
        return SearchBounds(
            low=np.array([1, 1, 1, 0.0001], dtype=np.float64),
            high=np.array([200, 200, 200, 0.05], dtype=np.float64),
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
                p.velocity = self.w * p.velocity + self.c1 * r1 * (p.best_position - p.position) + self.c2 * r2 * (gbest_position - p.position)
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
    def __init__(self, fitness_fn: Callable[[LSTMHyperparams, int], float], bounds: SearchBounds, swarm_size: int = 10, max_iter: int = 30, c1: float = 1.5, c2: float = 1.5, w_min: float = 0.6, w_max: float = 0.8, seed: int = 42, batch_size: int = 64):
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
                p.velocity = w_t * p.velocity + self.c1 * r1 * (p.best_position - p.position) + self.c2 * r2 * (gbest_position - p.position)
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


def run_baseline_lstm(df: pd.DataFrame, feature_cols: Sequence[str], lookback: int, hp: LSTMHyperparams, seed: int, verbose: int = 0, device: Optional[torch.device] = None, amp: bool = False) -> ForecastArtifacts:
    windows = prepare_windows(df, feature_cols=feature_cols, lookback=lookback)
    factory = TorchRegressorFactory(n_features=windows.n_features, verbose=verbose, device=device, amp=amp)
    pred, _, training_history = factory.fit_predict(hp, windows.X_train, windows.y_train, windows.X_test, target_scaler=windows.target_scaler, seed=seed)
    metrics = evaluate_forecast(windows.y_true, pred)
    return ForecastArtifacts(metrics, windows.y_true, pred, hp, [], training_history)


def make_fitness_function(df: pd.DataFrame, feature_cols: Sequence[str], lookback: int, batch_size: int, verbose: int = 0, device: Optional[torch.device] = None, amp: bool = False) -> Callable[[LSTMHyperparams, int], float]:
    windows = prepare_windows(df, feature_cols=feature_cols, lookback=lookback)
    factory = TorchRegressorFactory(n_features=windows.n_features, verbose=verbose, device=device, amp=amp)
    cache: Dict[Tuple[int, int, int, float], float] = {}

    def fitness(hp: LSTMHyperparams, seed: int) -> float:
        hp = hp.clipped()
        key = (hp.epochs, hp.node1, hp.node2, round(hp.learning_rate, 6))
        if key in cache:
            return cache[key]
        pred, _, _ = factory.fit_predict(hp, windows.X_train, windows.y_train, windows.X_test, target_scaler=windows.target_scaler, seed=seed)
        value = rmse(windows.y_true, pred)
        cache[key] = value
        return value

    return fitness


def run_search_model(df: pd.DataFrame, feature_cols: Sequence[str], lookback: int, method: str, search_iters: int, swarm_size: int, batch_size: int, seed: int, verbose: int = 0, device: Optional[torch.device] = None, amp: bool = False) -> ForecastArtifacts:
    fitness_fn = make_fitness_function(df, feature_cols, lookback, batch_size=batch_size, verbose=verbose, device=device, amp=amp)
    bounds = SearchBounds.default()
    if method.lower() == "pso":
        searcher = PSOBase(fitness_fn, bounds, swarm_size, search_iters, c1=1.5, c2=1.5, w=0.6, seed=seed, batch_size=batch_size)
    elif method.lower() == "ipso":
        searcher = IPSO(fitness_fn, bounds, swarm_size, search_iters, c1=1.5, c2=1.5, w_min=0.6, w_max=0.8, seed=seed, batch_size=batch_size)
    else:
        raise ValueError(f"Unsupported method: {method}")
    search_result = searcher.run()
    best_hp = search_result.best_hparams.clipped()
    artifacts = run_baseline_lstm(df, feature_cols, lookback, best_hp, seed=seed, verbose=verbose, device=device, amp=amp)
    artifacts.best_hparams = best_hp
    artifacts.search_history = search_result.history
    return artifacts
