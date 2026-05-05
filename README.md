# PSO-LSTM Stock Return Forecasting

A hybrid forecasting system that pairs a 2-layer PyTorch LSTM with Improved
Particle Swarm Optimization (IPSO) for hyperparameter search, alongside an
XGBoost lag-feature baseline trained on the same supervised target. The
project tackles the well-known difficulty of next-bar return prediction in
high-noise, non-stationary equity time series, where naive grid search is
prohibitively expensive and small architectural choices dominate the bias /
variance / generalisation tradeoff. PSO is used because the LSTM
hyperparameter surface is non-convex and discrete in places (batch size),
which favours a population-based, gradient-free search over manual tuning.
The IPSO variant from Ji et al. 2021 mitigates premature convergence via a
non-linear tanh inertia schedule and an adaptive mutation operator; the
fitness function follows Deng & Peng 2025 in penalising the network's
mean-squared weight magnitude (MSW) on top of validation MSE; and the
overall search-space layout follows Zeng et al. 2025. A CSCI-633 academic
project (RIT, Spring 2026).

---

## Key design decisions

- **Split-first architecture.** The chronological 70/10/20 train/val/test
  split is computed in `pipelines/run_build_features.py` *before* any
  stateful transformer is fit. Wavelet thresholds, the multi-stage feature
  selector, the `FrozenMinMaxScaler` (features), and the
  `FrozenStandardScaler` (target) all `fit` on train only and are pickled
  into a single `frozen_pipeline.pkl`; val and test only ever see
  `transform()`.
- **NYSE-RTH-aligned 1-minute bars.** Every ticker is reindexed to the
  shared NYSE regular-trading-hours calendar (SPY benchmark) with causal
  forward-fill capped at 5 bars (`MAX_GAP_FILL_BARS`), making cross-ticker
  features safe and timestamps directly comparable across symbols.
- **1-bar-ahead forward log-return target, z-scored.** The supervised
  target is `y[t] = log(close[t+1] / close[t])`, then standardised with a
  train-only `FrozenStandardScaler` (mean 0, std 1). Under z-scored
  targets, a constant-mean predictor has MSE near 1.0, which gives a clean
  reference value when reading training curves.
- **IPSO with tanh inertia and adaptive mutation.** Improved PSO per Ji et
  al. 2021: per-iteration inertia
  `omega(t) = w_max - (w_max - w_min) * tanh(4 t / T)` and a mutation
  probability that decays from 0.30 to 0 over the run. Each particle's
  fitness training runs for a *fixed 100 epochs* - the `epochs` PSO
  dimension is retained for spec compatibility but its decoded value is
  ignored.
- **Spec-compliant composite fitness.** Phase 1 minimises
  `F(x) = gamma * MSE + (1 - gamma) * msw_scale * MSW` with `gamma = 0.9`
  and an `msw_scale = 0.01` calibration factor that brings the
  weight-magnitude term into the same order as the scaled-target MSE.
  Bias parameters are *excluded* from MSW per Deng & Peng 2025; only
  recurrent and dense weight matrices contribute.
- **Modern PyTorch AMP, 6D PSO search space.** The LSTM trainer uses
  `torch.amp.GradScaler` when `use_amp: true` and CUDA is available
  (silent fp32 fallback on CPU). The PSO swarm searches a 6D vector of
  `lstm_units_1`, `lstm_units_2`, `dropout_rate`, `learning_rate` (log
  scale), `batch_size` (discrete `{32, 64}`), and `epochs` (decoded but
  unused).
- **Apples-to-apples XGBoost baseline.** The XGBoost pipeline consumes
  the same `X_*.npy` / `y_*.npy` arrays as the LSTM but reshapes them
  into lag features inside `run_xgboost.py` (`feature_type: "lag_based"`)
  so it predicts the *same* 1-bar-ahead z-scored return - no flattened
  sequence shortcut.

---

## System architecture

```
+-------------------+     +----------------------------+     +------------------------+
|  Alpaca OHLCV     | --> | Ingest + clean +           | --> | data/processed/        |
|  (per-ticker)     |     | NYSE-RTH align (SPY cal.)  |     | <TF>/<TICKER>.parquet  |
+-------------------+     +----------------------------+     +-----------+------------+
                                                                         |
                                                                         v
+----------------------------------------------------------------------------------+
| Split-first feature build (per-ticker, per-split)                                |
|   - 70/10/20 chronological split                                                 |
|   - wavelet denoise -> indicators -> cross-ticker context                        |
|   - 4-stage selector (variance / corr / VIF / MI), fit on TRAIN only             |
|   - FrozenMinMaxScaler (features) + FrozenStandardScaler (target), TRAIN only    |
+----------------------------------+-----------------------------------------------+
                                   |
                                   v
                       data/features_v2/<TICKER>/
                       X_{train,val,test}.npy, y_{train,val,test}.npy,
                       frozen_pipeline.pkl
                                   |
        +--------------------------+--------------------------+
        v                          v                          v
+----------------+        +-------------------+      +-------------------+
| Baseline LSTM  |        |     PSO-LSTM      |      |     XGBoost       |
| run_lstm.py    |        | train_pso_lstm.py |      | run_xgboost.py    |
| (fixed config) |        | IPSO -> retrain   |      | (lag features)    |
+--------+-------+        +---------+---------+      +---------+---------+
         |                          |                          |
         +--------------------------+--------------------------+
                                    v
                           +-------------------+
                           | run_backtest.py   |
                           | session-aware     |
                           | trading sim       |
                           +---------+---------+
                                     v
                          metrics.json, equity.csv,
                          trade_log.csv, plots/*
```

---

## Repo layout

```
OptimizeStockPredictionViaPSO/
  config/
    default_config.yaml          single source of truth (data, features, models, pso, backtest)
    symbol_universe.yaml         optional explicit peer mapping
    tickers.txt                  ticker universe (one symbol per line)
  pipelines/                     CLI entry points (run from repo root)
    data_ingest_align_clean.py   Alpaca download -> clean -> NYSE-align
    run_build_features.py        split-first feature build, freezes pipeline
    run_lstm.py                  baseline LSTM training
    train_pso_lstm.py            Phase 1 IPSO search + Phase 2 final fit
    run_xgboost.py               XGBoost lag-feature baseline
    run_backtest.py              session-aware backtest dispatcher
  src/
    data/                        Alpaca client, OHLCV cleaner, NYSE alignment, windowing
    features/                    indicators, wavelet, 4-stage selector, scalers, target def
    models/                      LSTMModel/Trainer, XGBoostModel/Trainer, set_seeds
    optimizer/                   StandardPSO, IPSO, particle, SpecCompliantFitness
    backtesting/                 Backtester, BacktestResults, SessionEvent
    evaluation/                  metrics, canonical_split, plotting
    utils/                       config_loader, logger, seed, data_storage
  docs/
    QUICKSTART.md                end-to-end run instructions (canonical how-to)
    feature / model / evaluation specs
    research_notes.md            PSO bibliography
  constants.py                   DEFAULT_CONFIG_PATH, MAX_GAP_FILL_BARS
  requirements.txt
```

`data/`, `logs/`, `results/`, `plans/`, `tests/`, and `venv/` are
gitignored.

---

## Quickstart

The canonical end-to-end how-to lives in
[`docs/QUICKSTART.md`](docs/QUICKSTART.md), including per-stage CLI flags,
output paths, single-GPU sizing tables, and a troubleshooting section.
The minimal smoke-test path (one ticker, short window, CPU) is:

```bash
# 1. Ingest, clean, NYSE-align (reads dates + tickers from config)
python pipelines/data_ingest_align_clean.py \
    --config config/default_config.yaml \
    --tickers config/tickers.txt --skip-existing

# 2. Build split-first features for one timeframe
python pipelines/run_build_features.py \
    --config config/default_config.yaml \
    --tickers config/tickers.txt \
    --processed-dir data/processed --output data/features_v2 \
    --timeframe 1Min --target-method log_return --workers 1

# 3. Train baseline LSTM (use --device cpu without CUDA)
python pipelines/run_lstm.py \
    --config config/default_config.yaml \
    --ticker AAPL --timeframe 1Min \
    --data-path data/features_v2/AAPL --device cpu --run-id smoke

# 4. Backtest the trained model on the held-out 20% test slice
python pipelines/run_backtest.py \
    --config config/default_config.yaml \
    --model_type lstm_baseline \
    --model_path results/experiments/AAPL_1Min_lstm_baseline_smoke/train \
    --ticker AAPL --timeframe 1Min \
    --data-path data/features_v2/AAPL --processed-dir data/processed \
    --run-id smoke
```

For a full run, restore the canonical date range in the config, populate
`config/tickers.txt`, and substitute `--device cuda` and a larger
`--workers` value where appropriate.

---

## Configuration

Everything is driven by `config/default_config.yaml`, parsed into typed
dataclasses by `src/utils/config_loader.py`. Any new YAML key must also
exist on the corresponding dataclass - `load_config()` rejects unknown
keys. The seven top-level blocks are:

| Block | Purpose | Notable knobs |
|---|---|---|
| `data` | universe + calendar | `start_date`, `end_date`, `freq` (default `1Min`), `benchmark_ticker`, `session_start/end`, `max_ffill_bars` |
| `features` | engineering pipeline | `wavelet.{enabled, level}`, `selector.{variance_threshold, correlation_threshold, vif_threshold, mi_quantile_threshold}`, `cross_ticker.{spy_features, peer_features, peer_count}`, `windowing.{lookback, prediction_horizon}` |
| `lstm_baseline` | fixed-hp LSTM | `lstm_units_1`, `lstm_units_2`, `dropout_rate` (default `0.05`), `loss` (`mse_directional`), `directional_loss_weight` (default `0.1`), `batch_size`, `early_stopping.patience`, `use_amp`, `grad_clip` |
| `xgboost` | lag-feature baseline | `max_depth`, `learning_rate`, `n_estimators`, `early_stopping_rounds`, `feature_type: lag_based`, `tree_method: hist` |
| `pso` | IPSO search | `n_particles` (20), `n_iterations` (50), `inertia_min/max` (0.4 / 0.9), `c1`, `c2`, `n_workers` (4), `subsample_train_fraction` (0.25), `fitness.gamma` (0.9), `search_space.*` |
| `backtesting` | trading sim | `transaction_cost`, `slippage`, `position_fraction`, `stop_loss`, `daily_loss_limit`, `signal_generation`, `initial_capital` |
| `evaluation` | split + protocol | `splits.{train_pct, val_pct, test_pct}` (0.7 / 0.1 / 0.2), `pipeline.{fit_once, freeze_forever}` |

The `enforcement:` block at the top of the YAML documents the
data-leakage invariants (`allow_retraining: false`,
`allow_pipeline_refitting: false`, `allow_test_access_training: false`,
`require_frozen_state: true`, `require_chronological_splits: true`).
These are not currently enforced as runtime asserts in every pipeline,
but they describe the design contract the codebase is built around - do
not violate them.

---

## Models

### Baseline LSTM (`pipelines/run_lstm.py`)

A 2-layer `nn.LSTM` (`lstm_units_1` -> `lstm_units_2`) followed by a
linear head, trained with Adam, gradient clipping at 1.0, and modern
PyTorch AMP when CUDA is available. The default loss is
`mse_directional`: standard MSE plus a sign-disagreement penalty
weighted by `directional_loss_weight = 0.1`, designed to discourage the
optimiser from collapsing onto the constant-mean predictor (which has
near-zero variance and yields MSE ~1 under z-scored targets but no
trading signal). Early stopping on `val_loss` with patience 10. All
hyperparameters come from the `lstm_baseline` block - none are searched.

### PSO-LSTM (`pipelines/train_pso_lstm.py`)

A two-phase protocol. **Phase 1**: IPSO over the 6D search space (Ji et
al. 2021) with 20 particles for 50 iterations. Each particle decodes to
LSTM hyperparameters, gets a *fixed 100 epochs* of training on the
PSO-train slice (the most-recent `subsample_train_fraction` of the 90/10
re-split of train+val), and is scored on the never-subsampled PSO-val
slice using `SpecCompliantFitness`:

```
F(x) = 0.9 * MSE_val + 0.1 * 0.01 * MSW(model)
```

with MSW computed only over weight matrices (biases excluded, per
Deng & Peng 2025). **Phase 2**: a single LSTM is retrained from scratch
with the global-best particle's hyperparameters on the *full* 80%
train+val slice, again for 100 epochs, with no early stopping. The held-
out 20% test slice is touched only by the backtester. `--skip-pso`
re-uses an existing Phase 1 result.

### XGBoost baseline (`pipelines/run_xgboost.py`)

`xgboost.XGBRegressor` with `tree_method: hist`, `max_depth: 6`,
`n_estimators: 500`, and early stopping on val. Crucially, the LSTM
windowed inputs (`(N, lookback, F)`) are reshaped into 2D lag features
inside the XGBoost pipeline so the model predicts the same 1-bar-ahead
z-scored log return that the LSTMs predict. This is what makes the head-
to-head comparison meaningful - both models see the same supervised
target on the same chronological split.

---

## Evaluation

A *single* 70/10/20 chronological split is established once in
`run_build_features.py` (`src/evaluation/canonical_split.py`). There is
no walk-forward path in this codebase - the earlier walk-forward
prototype was removed, and any references in the older README are
out of date. All test-set numbers come from `pipelines/run_backtest.py`,
which dispatches on `--model-type` and reuses the `load_trained_model`
helpers exported by `pipelines/run_lstm.py` and `pipelines/run_xgboost.py`
to rebuild the trained model from its `models/<run-id>/` artefacts. It
then runs inference on the held-out 20% slice and feeds predictions into
the session-aware `Backtester`.

The backtester recognises NYSE session boundaries via `SessionEvent`
(returns `OPEN` / `CLOSE` / `NONE` from `_check_session_boundary`),
charges per-leg `transaction_cost` plus `slippage` on each entry and
exit, supports a configurable `stop_loss`, `position_fraction`, and
`daily_loss_limit`, and writes:

| Output | Contents |
|---|---|
| `metrics.json` | full statistical + trading metric panel |
| `equity_curve.csv` | per-bar equity, position, return |
| `trade_log.csv` | per-trade entry/exit, PnL, hold time |
| `plots/` | equity curve, drawdown, returns histogram, signal overlay |

Computed metrics:

| Family | Metrics |
|---|---|
| Statistical | RMSE, MAE, MAPE, R-squared, directional accuracy (with `exclude_zeros=True` and `threshold=1e-8`), F1 / AUC over the ternary `{down, flat, up}` labelling |
| Trading | Sharpe (annualised at `sqrt(252 * 390)` for 1-minute bars), Sortino, CAGR, max drawdown, Calmar, profit factor, win rate, information ratio vs. benchmark |

Note: `metrics.py:ANNUALISE_1MIN = sqrt(252 * 390)` and
`n_bars_per_year = 252 * 390` in CAGR are calibrated for the default
1-minute frequency. If you change `data.freq`, adjust those constants
accordingly.

---

## Reproducibility

`set_all_seeds(seed)` in `src/utils/seed.py` (re-exported from
`src.models`) seeds Python's `random`, NumPy, PyTorch CPU/CUDA,
`PYTHONHASHSEED`, sets `cudnn.deterministic = True`, disables
`cudnn.benchmark`, and (in the broader pipeline-level wrapper) calls
`torch.use_deterministic_algorithms(True, warn_only=True)`. The PSO
swarm uses `random_seed: 42` and `deterministic: true` by default;
per-particle seeds are derived as `global_seed * 1000 + particle_idx`
via `get_rng()` so each fitness evaluation is independently
reproducible.

**Caveat for paper-quality runs.** When `pso.n_workers > 1`, particle
fitness evaluations run in `ProcessPoolExecutor` subprocesses; the
swarm RNG forks identically into each child, breaking strict
reproducibility (this is also flagged in the QUICKSTART troubleshooting
section). For reported numbers, set `pso.n_workers: 1` and accept the
slower wall-time. The default of `4` is tuned for development throughput
on a 24 GB+ GPU, not for canonical reproducibility.

The `xgboost.n_jobs: -1` default likewise trades reproducibility for
speed; set `n_jobs: 1` for cross-machine bit-identical XGBoost runs.

---

## Dependencies

Python 3.10+. Key libraries:

| Package | Min version | Purpose |
|---|---|---|
| `torch` | `>=2.2` | LSTM, AMP, deterministic ops |
| `xgboost` | `>=2.0` | gradient-boosted baseline |
| `numpy`, `pandas` | `>=1.26`, `>=2.1` | numerics, data frames |
| `scikit-learn` | `>=1.4` | scalers, MI, VIF utilities |
| `scipy` | `>=1.12` | wavelet support routines |
| `pywavelets` (`pywt`) | - | Haar wavelet denoising |
| `alpaca-py`, `alpaca-trade-api` | `>=0.30`, `>=3.0` | OHLCV ingestion |
| `pandas-market-calendars` | `>=4.3` | NYSE RTH calendar |
| `pandas-ta`, `ta-lib` | latest | technical indicators |
| `pyarrow`, `tables` | `>=15.0`, `>=3.9` | parquet / HDF5 storage |
| `pyyaml`, `python-dotenv` | `>=6.0`, `>=1.0` | config and `.env` |
| `prettytable`, `tqdm` | `>=3.10`, `>=4.66` | CLI tables and progress |
| `matplotlib`, `seaborn` | `>=3.8`, `>=0.13` | plots |

The full pinned list lives in [`requirements.txt`](requirements.txt).
A working `.env` at the repo root must contain `ALPACA_API_KEY`,
`ALPACA_SECRET_KEY` (or the legacy `ALPACA_API_SECRET`), and
`ALPACA_BASE_URL` - see `.env.example`.

---

## References

The full bibliography with per-paper PSO-design notes is in
[`docs/research_notes.md`](docs/research_notes.md). Primary references:

- **Ji, Liew & Yang (2021).** *Application of LSTM Model based on
  Particle Swarm Optimization Algorithm in Stock Market Trend
  Prediction.* IEEE Access. Source of the IPSO algorithm: non-linear
  tanh inertia weight and adaptive mutation factor used in this
  project's `IPSO` class.
- **Zeng et al. (2025).** *Enhancing stock index prediction: A hybrid
  LSTM-PSO model for improved forecasting accuracy.* Source of the 20
  particles / 50 iterations swarm sizing and the multi-layer LSTM
  search-space layout used here.
- **Deng & Peng (2025).** *A Novel Improved Particle Swarm Optimization
  for LSTM.* Source of the composite fitness
  `gamma * MSE + (1 - gamma) * msw_scale * MSW`, with biases excluded
  from MSW - implemented in `src/optimizer/fitness.py`.
- **Lanbouri & Achchab.** *Stock Market Prediction on High-Frequency
  Data Using Long-Short Term Memory.* Reference for high-frequency
  intraday LSTM design; does not use PSO and is included as the
  manually-tuned-LSTM contrast point.

---

## Disclaimer

This project is for **educational and research purposes only** as part
of CSCI-633 (Biologically-Inspired Intelligent Systems) at RIT. Nothing
here is investment advice. Backtested performance on a single
chronological split is not a reliable predictor of live trading
performance, and the Alpaca data pipeline, transaction-cost model, and
slippage model are simplified relative to a production execution
environment. Do not use this code to make real-money trades.
