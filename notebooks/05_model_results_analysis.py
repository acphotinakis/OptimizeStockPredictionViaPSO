import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from src.features.build_features import calculate_indicators

from hydra import initialize, compose

# Add project root to sys.path
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

# Hydra config
cfg_path = root_path / "configs"
initialize(config_path=str(cfg_path), version_base="1.3")
cfg = compose(config_name="config")


# 05_model_results_analysis.py
# Purpose: Analyze the predictive performance of the LSTM on the out-of-sample test set

import pandas as pd
import matplotlib.pyplot as plt

# ── Load results CSV ───────────────────────────────────────────────────
results_df = pd.read_csv("../reports/results/predictions.csv")
results_df["timestamp"] = pd.to_datetime(results_df["timestamp"])
results_df.set_index("timestamp", inplace=True)

# ── Actual vs Predicted Mid-Price ─────────────────────────────────────
plt.figure(figsize=(15, 6))
plt.plot(results_df["actual"], label="Actual Mid-Price", alpha=0.7)
plt.plot(results_df["predicted"], label="PSO-LSTM Prediction", linestyle="--")
plt.title("Actual vs. Predicted Mid-Price Movement")
plt.legend()
plt.show()

# ── Financial Performance ─────────────────────────────────────────────
cum_returns = (1 + results_df["strategy_returns"]).cumprod()
cum_returns.plot(title="Cumulative Strategy Returns (Out-of-Sample)")
plt.show()

print(f"Final Sharpe Ratio: {results_df['sharpe'].iloc[-1]:.2f}")
print(f"Max Drawdown: {results_df['mdd'].min():.2%}")
