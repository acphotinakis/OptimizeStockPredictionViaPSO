"""
Backtest Results - Persistence and Loading for Backtesting Results

Provides standardized storage and retrieval for backtesting outputs,
including metrics, time series, and metadata.

Author: Production System
Version: 1.0
Source: BACKTEST_DESIGN.md Section 16.2
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass
class BacktestResults:
    """Container for complete backtesting results.
    
    Stores all data needed for analysis, visualization, and comparison.
    """

    # Identification
    model_type: str
    ticker: str
    timestamp: str

    # Predictions & actuals
    predictions: np.ndarray
    actual_returns: np.ndarray
    dates: Optional[pd.DatetimeIndex]

    # Trading simulation
    signals: np.ndarray
    strategy_returns: np.ndarray
    equity_curve: np.ndarray
    trade_costs: np.ndarray

    # Metrics
    statistical_metrics: Dict[str, float]
    trading_metrics: Dict[str, float]

    # Model metadata
    model_metadata: Dict

    # Configuration
    backtest_config: Dict


def save_backtest_results(
    results: BacktestResults,
    output_dir: Path,
) -> None:
    """
    Save complete backtesting results to disk.

    Creates structured output directory with:
    - backtest_results.json (metrics + summary)
    - equity_curve.csv (time series)
    - predictions.csv (predictions + actuals)
    - metadata.yaml (model + config info)

    Args:
        results: BacktestResults instance
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving backtest results to {output_dir}")

    # 1. Save metrics (JSON)
    metrics_data = {
        "model_type": results.model_type,
        "ticker": results.ticker,
        "timestamp": results.timestamp,
        "statistical_metrics": results.statistical_metrics,
        "trading_metrics": results.trading_metrics,
        "summary": {
            "total_return": float(results.equity_curve[-1] / results.equity_curve[0] - 1),
            "final_capital": float(results.equity_curve[-1]),
            "n_trades": int(np.sum(np.abs(np.diff(results.signals)) > 0)),
            "n_days": len(results.dates) if results.dates is not None else len(results.predictions),
            "total_costs": float(np.sum(results.trade_costs)),
        },
    }

    metrics_path = output_dir / "backtest_results.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    logger.info(f"  ✓ Metrics saved to {metrics_path.name}")

    # 2. Save time series (CSV)
    if results.dates is not None:
        dates_col = results.dates
    else:
        dates_col = pd.RangeIndex(len(results.predictions))

    equity_df = pd.DataFrame(
        {
            "date": dates_col,
            "prediction": results.predictions,
            "actual_return": results.actual_returns,
            "signal": results.signals,
            "strategy_return": results.strategy_returns,
            "equity": results.equity_curve,
            "trade_cost": results.trade_costs,
        }
    )

    equity_path = output_dir / "equity_curve.csv"
    equity_df.to_csv(equity_path, index=False)
    logger.info(f"  ✓ Equity curve saved to {equity_path.name}")

    # 3. Save predictions separately (CSV)
    predictions_df = pd.DataFrame(
        {
            "date": dates_col,
            "prediction": results.predictions,
            "actual_return": results.actual_returns,
        }
    )

    pred_path = output_dir / "predictions.csv"
    predictions_df.to_csv(pred_path, index=False)
    logger.info(f"  ✓ Predictions saved to {pred_path.name}")

    # 4. Save metadata (YAML)
    metadata = {
        "model_metadata": results.model_metadata,
        "backtest_config": results.backtest_config,
        "identification": {
            "model_type": results.model_type,
            "ticker": results.ticker,
            "timestamp": results.timestamp,
        },
    }

    metadata_path = output_dir / "metadata.yaml"
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)
    logger.info(f"  ✓ Metadata saved to {metadata_path.name}")

    # 5. Generate summary report (Markdown)
    report_path = output_dir / "backtest_report.md"
    generate_backtest_report(results, report_path)
    logger.info(f"  ✓ Report saved to {report_path.name}")

    logger.info("=" * 80)


def generate_backtest_report(
    results: BacktestResults,
    output_path: Path,
) -> None:
    """
    Generate human-readable backtest report in Markdown.

    Args:
        results: BacktestResults instance
        output_path: Path to save report
    """
    stat = results.statistical_metrics
    trade = results.trading_metrics

    total_return = results.equity_curve[-1] / results.equity_curve[0] - 1
    n_trades = int(np.sum(np.abs(np.diff(results.signals)) > 0))
    total_costs = np.sum(results.trade_costs)

    report = f"""# Backtest Report: {results.model_type.upper()}

**Ticker:** {results.ticker}  
**Date:** {results.timestamp}  
**Model:** {results.model_type}

---

## Performance Summary

| Metric | Value |
|--------|-------|
| **Total Return** | {total_return:.2%} |
| **Annualized Return** | {trade.get('annualized_return', trade.get('cagr', 0.0)):.2%} |
| **Sharpe Ratio** | {trade.get('sharpe', 0.0):.2f} |
| **Max Drawdown** | {trade.get('max_drawdown', 0.0):.2%} |
| **Win Rate** | {trade.get('win_rate', 0.0):.2%} |
| **Total Trades** | {n_trades} |
| **Total Costs** | {total_costs:.4f} ({total_costs*100:.2f}%) |

---

## Statistical Metrics (Prediction Quality)

| Metric | Value |
|--------|-------|
| **RMSE** | {stat.get('rmse', 0.0):.6f} |
| **MAE** | {stat.get('mae', 0.0):.6f} |
| **MAPE** | {stat.get('mape', 0.0):.2f}% |
| **R²** | {stat.get('r2', 0.0):.4f} |
| **Directional Accuracy** | {stat.get('directional_accuracy', 0.0):.2%} |
| **F1 (Ternary)** | {stat.get('f1_ternary', 0.0):.4f} |

---

## Trading Metrics (Portfolio Performance)

| Metric | Value |
|--------|-------|
| **Sharpe Ratio** | {trade.get('sharpe', 0.0):.2f} |
| **Sortino Ratio** | {trade.get('sortino', 0.0):.2f} |
| **Calmar Ratio** | {trade.get('calmar', 0.0):.2f} |
| **Profit Factor** | {trade.get('profit_factor', 0.0):.2f} |
| **CAGR** | {trade.get('cagr', 0.0):.2%} |
| **Max Drawdown** | {trade.get('max_drawdown', 0.0):.2%} |
| **Information Ratio** | {trade.get('information_ratio', 0.0):.2f} |

---

## Model Information

**Type:** {results.model_metadata.get('model_type', 'N/A')}  
**Architecture:** {results.model_metadata.get('architecture', 'N/A')}  
**Framework:** {results.model_metadata.get('framework', 'N/A')}

---

## Backtest Configuration

```yaml
{yaml.dump(results.backtest_config, default_flow_style=False)}
```

---

## Conclusion

Backtest completed successfully with {n_trades} trades over {len(results.dates) if results.dates is not None else len(results.predictions)} trading days.

**Final Portfolio Value:** {results.equity_curve[-1]:.2f}  
**Total Return:** {total_return:.2%}  
**Risk-Adjusted Return (Sharpe):** {trade.get('sharpe', 0.0):.2f}
"""

    with open(output_path, "w") as f:
        f.write(report)


def load_backtest_results(
    results_dir: Path,
) -> BacktestResults:
    """
    Load saved backtest results from disk.

    Args:
        results_dir: Directory containing saved results

    Returns:
        BacktestResults instance

    Raises:
        FileNotFoundError: If required files missing
    """
    results_dir = Path(results_dir)

    logger.info(f"Loading backtest results from {results_dir}")

    # Load metrics
    metrics_path = results_dir / "backtest_results.json"
    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)

    # Load time series
    equity_path = results_dir / "equity_curve.csv"
    equity_df = pd.read_csv(equity_path)

    # Load metadata
    metadata_path = results_dir / "metadata.yaml"
    with open(metadata_path, "r") as f:
        metadata = yaml.safe_load(f)

    # Reconstruct BacktestResults
    results = BacktestResults(
        model_type=metrics_data["model_type"],
        ticker=metrics_data["ticker"],
        timestamp=metrics_data["timestamp"],
        predictions=equity_df["prediction"].values,
        actual_returns=equity_df["actual_return"].values,
        dates=pd.to_datetime(equity_df["date"]) if "date" in equity_df.columns else None,
        signals=equity_df["signal"].values,
        strategy_returns=equity_df["strategy_return"].values,
        equity_curve=equity_df["equity"].values,
        trade_costs=equity_df["trade_cost"].values,
        statistical_metrics=metrics_data["statistical_metrics"],
        trading_metrics=metrics_data["trading_metrics"],
        model_metadata=metadata["model_metadata"],
        backtest_config=metadata["backtest_config"],
    )

    logger.info("✓ Backtest results loaded successfully")

    return results
