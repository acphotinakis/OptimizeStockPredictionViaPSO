import numpy as np
import time
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VarianceStageOriginal:
    def __init__(self, threshold: float):
        self.variance_threshold = threshold

    def run(
        self,
        X: np.ndarray,
        names: List[str],
        original_indices: List[int],
    ) -> Tuple[np.ndarray, List[str], List[int]]:
        variances = np.var(X, axis=0, dtype=np.float64)
        keep_mask = variances > self.variance_threshold

        dropped = [n for n, k in zip(names, keep_mask) if not k]
        if dropped:
            logger.debug("Variance drop: %s", dropped)

        return (
            X[:, keep_mask],
            [n for n, k in zip(names, keep_mask) if k],
            [i for i, k in zip(original_indices, keep_mask) if k],
        )


class VarianceStageOptimized:
    def __init__(self, threshold: float):
        self.variance_threshold = threshold

    def run(
        self,
        X: np.ndarray,
        names: List[str],
        original_indices: List[int],
    ) -> Tuple[np.ndarray, List[str], List[int]]:
        variances = np.var(X, axis=0, dtype=np.float64)
        keep_mask = variances > self.variance_threshold

        names_arr = np.asarray(names)
        indices_arr = np.asarray(original_indices)

        kept_names = names_arr[keep_mask].tolist()
        kept_indices = indices_arr[keep_mask].tolist()

        if logger.isEnabledFor(logging.DEBUG):
            dropped = names_arr[~keep_mask]
            if dropped.size:
                logger.debug("Variance drop: %s", dropped.tolist())

        return X[:, keep_mask], kept_names, kept_indices


def generate_data(n_samples: int, n_features: int):
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # Inject some near-constant features
    for i in range(0, n_features, 10):
        X[:, i] = 0.001

    names = [f"f_{i}" for i in range(n_features)]
    indices = list(range(n_features))

    return X, names, indices


def benchmark(stage, X, names, indices, n_runs=5):
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        stage.run(X, names, indices)
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times), np.std(times)


def main():
    n_samples = 100_000
    n_features = 1_000

    logger.info("Generating data: samples=%d, features=%d", n_samples, n_features)
    X, names, indices = generate_data(n_samples, n_features)

    threshold = 1e-4

    original = VarianceStageOriginal(threshold)
    optimized = VarianceStageOptimized(threshold)

    logger.info("Running benchmark...")

    orig_mean, orig_std = benchmark(original, X, names, indices)
    opt_mean, opt_std = benchmark(optimized, X, names, indices)

    logger.info("\nResults:")
    logger.info("Original  : %.6f sec ± %.6f", orig_mean, orig_std)
    logger.info("Optimized : %.6f sec ± %.6f", opt_mean, opt_std)
    logger.info(
        "Speedup   : %.2fx", orig_mean / opt_mean if opt_mean > 0 else float("inf")
    )


if __name__ == "__main__":
    main()
