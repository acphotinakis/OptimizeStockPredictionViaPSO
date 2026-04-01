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

# 06_pso_optimization_analysis.py
# Purpose: Visualize the convergence behavior of the Improved Particle Swarm Optimization

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load PSO history ───────────────────────────────────────────────────
pso_log = pd.read_csv("../models/pso_best/pso_history.csv")

# ── Swarm Convergence Curve ───────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(pso_log["iteration"], pso_log["best_fitness"], marker="o")
plt.xlabel("Iteration")
plt.ylabel("Fitness (Sharpe Ratio)")
plt.title("Swarm Convergence Curve")
plt.grid(True)
plt.show()

# ── Parameter Sensitivity ────────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(pso_log["hidden_size"], ax=ax[0], bins=15, kde=True)
ax[0].set_title("Distribution of Hidden Layers")

sns.scatterplot(data=pso_log, x="learning_rate", y="best_fitness", ax=ax[1])
ax[1].set_xscale("log")
ax[1].set_title("Learning Rate vs. Fitness")

plt.show()
