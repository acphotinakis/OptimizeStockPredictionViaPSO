import numpy as np
from scipy.stats import pearsonr

# ======================================================================
# STATISTICAL METRICS (Add this before Section 9 - Training Loop)
# ======================================================================


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_pred - y_true)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-10) -> float:
    """Mean Absolute Percentage Error (%)."""
    return float(np.mean(np.abs((y_pred - y_true) / (np.abs(y_true) + eps))) * 100)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-10))


def pearson_corr(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]


def information_coefficient(y_true, y_pred):
    return pearson_corr(y_true, y_pred)


def hit_rate(y_true, y_pred, threshold=0.0):
    mask = np.abs(y_true) > threshold
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask]))


def sharpe_ratio(y_true, y_pred):
    positions = np.sign(y_pred)
    strat_returns = positions * y_true
    return np.mean(strat_returns) / (np.std(strat_returns) + 1e-12)


def directional_accuracy(y_true, y_pred, threshold=0.0, exclude_zeros=True):
    """
    Directional accuracy with correct zero handling.

    CRITICAL FIX (2026-04-28): Treats near-zero returns correctly.

    Args:
        y_true: Actual returns
        y_pred: Predicted returns
        threshold: Minimum absolute value to classify as directional
        exclude_zeros: If True, exclude near-zero actuals (RECOMMENDED)

    Returns:
        Fraction of correct direction predictions
    """
    if exclude_zeros:
        # Only evaluate where actual has clear direction
        mask = np.abs(y_true) > threshold

        if mask.sum() == 0:
            print("WARNING: No directional samples (all |y| <= threshold)")
            return float("nan")

        y_true_filt = y_true[mask]
        y_pred_filt = y_pred[mask]

        # Safe to use sign() now (no zeros)
        correct = np.sign(y_true_filt) == np.sign(y_pred_filt)
        return float(correct.mean())
    else:
        # Ternary: treats (0,0) as correct but (0,±1) as wrong
        yt = np.where(np.abs(y_true) < threshold, 0, np.sign(y_true))
        yp = np.where(np.abs(y_pred) < threshold, 0, np.sign(y_pred))
        return float((yt == yp).mean())


def f1_ternary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 1e-4,
) -> float:
    """Micro-averaged F1 for ternary classification (up / flat / down)."""
    from sklearn.metrics import f1_score

    def classify(arr: np.ndarray) -> np.ndarray:
        c = np.zeros(len(arr), dtype=int)  # flat = 0
        c[arr > threshold] = 1  # up
        c[arr < -threshold] = -1  # down
        return c

    return float(f1_score(classify(y_true), classify(y_pred), average="micro"))


def auc_ternary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 1e-4,
) -> float:
    """Micro-averaged one-vs-rest AUC for ternary classification."""
    try:
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import label_binarize

        def classify(arr):
            c = np.zeros(len(arr), dtype=int)
            c[arr > threshold] = 1
            c[arr < -threshold] = -1
            return c

        y_true_c = classify(y_true)
        classes = [-1, 0, 1]
        y_bin = label_binarize(y_true_c, classes=classes)
        # Use predicted probabilities from soft output (treated as rank scores)
        score_matrix = np.column_stack(
            [
                -y_pred,  # score for "down"
                -np.abs(y_pred),  # score for "flat"  (low magnitude)
                y_pred,  # score for "up"
            ]
        )
        return float(
            roc_auc_score(y_bin, score_matrix, average="micro", multi_class="ovr")
        )
    except Exception:
        return float("nan")
