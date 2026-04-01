# ── Standard library imports ───────────────────────────────────────────
import sys
from pathlib import Path

# ── Third-party imports ───────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
from hydra import initialize, compose

# ── Add project root to sys.path for local module imports ─────────────
root_path = Path(__file__).parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

# ── Hydra configuration ──────────────────────────────────────────────
cfg_path = root_path / "configs"
initialize(config_path=str(cfg_path), version_base="1.3")
cfg = compose(config_name="config")

# ── Local imports ─────────────────────────────────────────────────────
from src.features.build_features import calculate_indicators

# ── 05_model_results_analysis.py ─────────────────────────────────────
# Purpose: Analyze the predictive performance of the LSTM on the out-of-sample test set

# ── Load results CSV ───────────────────────────────────────────────────
results_path = Path(cfg.paths.reports.results) / "predictions.csv"
results_df = pd.read_csv(results_path)
results_df["timestamp"] = pd.to_datetime(results_df["timestamp"])
results_df.set_index("timestamp", inplace=True)

# ── Actual vs Predicted Mid-Price ─────────────────────────────────────
fig_path_1 = Path(cfg.paths.reports.plots) / "05_actual_vs_predicted.png"
plt.figure(figsize=(15, 6))
plt.plot(results_df["actual"], label="Actual Mid-Price", alpha=0.7)
plt.plot(results_df["predicted"], label="PSO-LSTM Prediction", linestyle="--")
plt.title("Actual vs. Predicted Mid-Price Movement")
plt.legend()
plt.tight_layout()
plt.savefig(fig_path_1)
plt.close()  # Close figure to free memory

# ── Financial Performance ─────────────────────────────────────────────
fig_path_2 = Path(cfg.paths.reports.plots) / "05_cumulative_strategy_returns.png"
cum_returns = (1 + results_df["strategy_returns"]).cumprod()
cum_returns.plot(title="Cumulative Strategy Returns (Out-of-Sample)")
plt.tight_layout()
plt.savefig(fig_path_2)
plt.close()

# ── Key Metrics ───────────────────────────────────────────────────────
print(f"Final Sharpe Ratio: {results_df['sharpe'].iloc[-1]:.2f}")
print(f"Max Drawdown: {results_df['mdd'].min():.2%}")
