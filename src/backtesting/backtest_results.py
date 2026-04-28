from datetime import datetime
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import yaml

import logging
import sys


# Add project root to path
# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting.backtester import BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class BacktestResults:
    model_type: str
    ticker: str
    timestamp: str

    predictions: np.ndarray
    actual_returns: np.ndarray
    dates: Optional[pd.DatetimeIndex]

    backtest_result: BacktestResult

    signals: np.ndarray
    strategy_returns: np.ndarray
    equity_curve: np.ndarray
    trade_costs: np.ndarray

    model_metadata: Dict
    backtest_config: Dict
    statistical_metrics: Dict[str, float]
    trading_metrics: Dict[str, float]

    @staticmethod
    def from_engine(
        *,
        model_type: str,
        ticker: str,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        dates: Optional[pd.DatetimeIndex],
        backtest_result: "BacktestResult",
        signals: np.ndarray,
        strategy_returns: np.ndarray,
        trade_costs: np.ndarray,
        model_metadata: Dict,
        backtest_config: Dict,
        statistical_metrics: Dict[str, float],
        trading_metrics: Dict[str, float],
    ) -> "BacktestResults":
        return BacktestResults(
            model_type=model_type,
            ticker=ticker,
            timestamp=datetime.utcnow().isoformat(),
            predictions=predictions,
            actual_returns=actual_returns,
            dates=dates,
            backtest_result=backtest_result,
            signals=signals,
            strategy_returns=strategy_returns,
            equity_curve=backtest_result.equity_curve,
            trade_costs=trade_costs,
            model_metadata=model_metadata,
            backtest_config=backtest_config,
            statistical_metrics=statistical_metrics,
            trading_metrics=trading_metrics,
        )


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
            "total_return": float(
                results.equity_curve[-1] / results.equity_curve[0] - 1
            ),
            "final_capital": float(results.equity_curve[-1]),
            "n_trades": int(np.sum(np.abs(np.diff(results.signals)) > 0)),
            "n_days": (
                len(results.dates)
                if results.dates is not None
                else len(results.predictions)
            ),
            "total_costs": float(np.sum(results.trade_costs)),
        },
    }

    metrics_path = output_dir / "backtest_results.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    logger.info(f"   Metrics saved to {metrics_path.name}")

    # 2. Save time series (CSV)
    # Find the minimum length to ensure all arrays are aligned
    min_len = min(
        len(results.predictions),
        len(results.actual_returns),
        len(results.signals),
        len(results.strategy_returns),
        len(results.equity_curve),
        len(results.trade_costs),
    )

    if results.dates is not None:
        dates_col = results.dates[:min_len]
    else:
        dates_col = pd.RangeIndex(min_len)

    logger.info(f"Aligning time series to length {min_len}")

    equity_df = pd.DataFrame(
        {
            "date": dates_col,
            "prediction": results.predictions[:min_len],
            "actual_return": results.actual_returns[:min_len],
            "signal": results.signals[:min_len],
            "strategy_return": results.strategy_returns[:min_len],
            "equity": results.equity_curve[:min_len],
            "trade_cost": results.trade_costs[:min_len],
        }
    )

    equity_path = output_dir / "equity_curve.csv"
    equity_df.to_csv(equity_path, index=False)
    logger.info(f"   Equity curve saved to {equity_path.name}")

    # 3. Save predictions separately (CSV)
    predictions_df = pd.DataFrame(
        {
            "date": dates_col,
            "prediction": results.predictions[:min_len],
            "actual_return": results.actual_returns[:min_len],
        }
    )

    pred_path = output_dir / "predictions.csv"
    predictions_df.to_csv(pred_path, index=False)
    logger.info(f"   Predictions saved to {pred_path.name}")

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
    logger.info(f"   Metadata saved to {metadata_path.name}")

    # 5. Generate summary report (Markdown)
    report_path = output_dir / "backtest_report.md"
    generate_backtest_report(results, report_path)
    logger.info(f"   Report saved to {report_path.name}")

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
