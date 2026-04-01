import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)


def calculate_rmse(actual, predicted):
    """
    Computes Root Mean Square Error (RMSE) to measure the magnitude of prediction error.
    Used as a primary statistical benchmark in the project proposal.
    """
    return np.sqrt(mean_squared_error(actual, predicted))


def calculate_mae(actual, predicted):
    """
    Computes Mean Absolute Error (MAE) for a robust measure of average error magnitude.
    """
    return mean_absolute_error(actual, predicted)


def calculate_directional_accuracy(actual_returns, predicted_returns):
    """
    Measures the ability to predict market momentum.
    Defined as: Correct sign predictions / Total predictions.
    """
    # Compare the signs of the returns (1 for up, -1 for down, 0 for flat)
    correct_direction = np.sign(actual_returns) == np.sign(predicted_returns)
    return np.mean(correct_direction)


def calculate_sharpe_ratio(strategy_returns, risk_free_rate=0.0):
    """
    Primary measure of risk-adjusted return.
    Calculated as the ratio of excess return to the standard deviation of return.
    """
    excess_returns = strategy_returns - (
        risk_free_rate / 252 / 390
    )  # Adjust for 1-min frequency
    if len(excess_returns) < 2 or excess_returns.std() == 0:
        return 0.0

    # Annualized Sharpe Ratio for intraday data
    return np.sqrt(252 * 390) * (excess_returns.mean() / excess_returns.std())


def calculate_max_drawdown(cumulative_returns):
    """
    Measures the worst peak-to-trough decline (risk).
    Calculated as: (Peak - Trough) / Peak.
    """
    # Ensure cumulative_returns is a Series for peak tracking
    if not isinstance(cumulative_returns, pd.Series):
        cumulative_returns = pd.Series(cumulative_returns)

    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    return drawdown.min()  # Returns the most negative value (the max drop)


def evaluate_performance(df, actual_col="actual", pred_col="predicted"):
    """
    Orchestrates the calculation of all metrics defined in the research plan .

    Args:
        df (pd.DataFrame): Results dataframe containing actual and predicted values.
        actual_col (str): Column name for ground truth mid-price movement.
        pred_col (str): Column name for LSTM predictions.

    Returns:
        dict: A structured evaluation report.
    """
    logger.info("Calculating final performance metrics...")

    actual = df[actual_col]
    predicted = df[pred_col]

    # Statistical Metrics
    rmse = calculate_rmse(actual, predicted)
    mae = calculate_mae(actual, predicted)
    dir_acc = calculate_directional_accuracy(actual, predicted)

    # Trading/Financial Metrics (Assuming simple sign-based strategy)
    # Strategy return = prediction_direction * actual_price_movement
    strategy_returns = np.sign(predicted) * actual
    cumulative_returns = (1 + strategy_returns).cumprod()

    sharpe = calculate_sharpe_ratio(strategy_returns)
    mdd = calculate_max_drawdown(cumulative_returns)

    report = {
        "rmse": float(rmse),
        "mae": float(mae),
        "directional_accuracy": float(dir_acc),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(mdd),
        "total_return": float(cumulative_returns.iloc[-1] - 1),
    }

    logger.info(
        f"Evaluation Complete | Sharpe: {report['sharpe_ratio']:.2f} | RMSE: {report['rmse']:.6f}"
    )
    return report
