# Quickstart

End-to-end commands for the PSO-LSTM stock-prediction pipeline. Run every
command from the **repo root** with the `633` conda environment active.

## 0. Prerequisites

```bash
conda activate 633
pip install -r requirements.txt        # one-time dependency sync
```

`.env` at the repo root must contain Alpaca credentials. Either name for the
secret works:

```bash
ALPACA_API_KEY=PK...
ALPACA_SECRET_KEY=...                  # or ALPACA_API_SECRET=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

`config/tickers.txt` lists the ticker universe (one symbol per line, `#` to
comment). `config/default_config.yaml` is the single source of truth for date
range, frequency, and all model hyperparameters.

The default frequency is `1Min`. Examples below use `1Min`; substitute
`1Day`/`5Min`/`15Min`/`1Hour` as needed (the same string passes through every
pipeline).

---

## 1. Ingest + clean + NYSE-align

Downloads OHLCV from Alpaca, validates, fills short gaps, and reindexes each
ticker to the NYSE RTH minute calendar. Reads dates and frequency from
`config/default_config.yaml`.

```bash
python pipelines/data_ingest_align_clean.py \
    --config config/default_config.yaml \
    --tickers config/tickers.txt \
    --skip-existing
```

Writes `data/raw/<TF>/<TICKER>.parquet`, `data/cleaned/<TF>/<TICKER>.parquet`,
`data/aligned/<TF>/<TICKER>.parquet`, `data/processed/<TF>/<TICKER>.parquet`.
`--skip-existing` lets you resume after a partial run.

---

## 2. Build features (one timeframe per invocation)

Computes 45+ technical indicators, splits chronologically 70/10/20 with
target/features computed per-split, fits the feature scaler + 3-stage
feature selector on **train only**, standardises the target with
`FrozenStandardScaler` (mean=0, std=1 on train), and serialises the frozen
pipeline.

```bash
python pipelines/run_build_features.py \
    --config config/default_config.yaml \
    --tickers config/tickers.txt \
    --processed-dir data/processed \
    --output data/features_v2 \
    --timeframe 1Min \
    --target-method log_return \
    --workers 4
```

`--workers N` parallelises across tickers via `ProcessPoolExecutor`. Each
worker uses ~300–500 MB at 1Min over multi-year history; size to system
memory. Sequential default is `--workers 1`.

| Machine | Recommended `--workers` |
|---|---|
| 16 GB Mac | 4 |
| 32 GB workstation | 8 |
| 64 GB+ server | `min(n_tickers, $(nproc))` |

Writes per ticker:

```
data/features_v2/<TICKER>/
    X_train.npy      y_train.npy            # y_*.npy is z-scored, NOT in [-1, 1]
    X_val.npy        y_val.npy
    X_test.npy       y_test.npy
    test_index.npy
    frozen_pipeline.pkl                     # {feature_scaler, target_scaler, feature_selector, metadata}
    feature_scaler.joblib                   # FrozenMinMaxScaler  ([-1, 1])
    target_scaler.joblib                    # FrozenStandardScaler (mean 0, std 1)
    feature_selector.joblib  selected_features.json
    logs/build_features.log                 # per-worker log when --workers > 1
```

> **Important:** the output path has no timeframe component, so running
> multiple timeframes in one invocation would overwrite each other. Run this
> step once per timeframe you need.

---

## 3. Train

Pick one. Each writes to
`results/experiments/<TICKER>_<TF>_<MODEL>_<RUN_ID>/`.

### 3a. Baseline LSTM (fixed hyperparameters from config)

```bash
python pipelines/run_lstm.py \
    --config config/default_config.yaml \
    --ticker AAPL \
    --timeframe 1Min \
    --data-path data/features_v2/AAPL \
    --device cuda \
    --run-id baseline-001
```

Drop `--device cuda` (use `cpu`) on machines without CUDA. The LSTM uses
modern AMP (`torch.amp.GradScaler`) when `lstm_baseline.use_amp=true` and
CUDA is available; on CPU it silently falls back to fp32.

**Anti-collapse defaults** are now baked into the config:
- `loss: "mse_directional"` adds a sign-disagreement penalty on top of MSE
  (controlled by `directional_loss_weight: 0.1`).
- `dropout_rate: 0.05` (was 0.2) — high dropout was poisoning train metrics
  vs. the dropout-off val pass and dragging the optimiser into the
  constant-mean local minimum.
- The target was z-scored at stage 2, so a constant-mean predictor now has
  MSE = 1.0; any improvement below that is real signal.

### 3b. PSO-LSTM (Phase 1 IPSO search → Phase 2 final fit)

```bash
python pipelines/train_pso_lstm.py \
    --config config/default_config.yaml \
    --data-path data/features_v2/AAPL \
    --output-dir results/pso_lstm/AAPL_1Min
```

Phase 1: 20 particles × 50 iterations of LSTM training. Phase 2: final fit
on combined train+val (80%) with the best hyperparameters and the exact PSO
epoch count (no early stopping).

To re-use a prior Phase 1 search:
```bash
python pipelines/train_pso_lstm.py \
    --config config/default_config.yaml \
    --data-path data/features_v2/AAPL \
    --output-dir results/pso_lstm/AAPL_1Min \
    --skip-pso
```

### 3c. XGBoost (lag-feature baseline)

```bash
python pipelines/run_xgboost.py \
    --config config/default_config.yaml \
    --ticker AAPL \
    --timeframe 1Min \
    --data-path data/features_v2/AAPL \
    --run-id xgb-001
```

XGBoost consumes the same `X_train.npy` / `y_train.npy` files as the LSTM
but reshapes them into lag features inside the pipeline, so the comparison
is apples-to-apples (both predict the same 1-bar-ahead forward return).

---

## 4. Backtest

All three model types are wired. The dispatcher loads the model, runs
inference on the held-out 20% test slice, drives the session-aware
`Backtester`, and writes metrics + equity curve + trade log + plots under
`results/experiments/<TICKER>_<TF>_<MODEL>_<RUN_ID>/backtest/`.

### 4a. Baseline LSTM

```bash
python pipelines/run_backtest.py \
    --config config/default_config.yaml \
    --model_type lstm_baseline \
    --model_path results/experiments/AAPL_1Min_lstm_baseline_baseline-001/train \
    --ticker AAPL --timeframe 1Min \
    --data-path data/features_v2/AAPL \
    --processed-dir data/processed \
    --run-id baseline-001
```

`--model_path` is the **directory** that contains `baseline_lstm_model.pt`
and `model_config.json` (i.e. the `train/` subdir of the experiment).

### 4b. PSO-LSTM

```bash
python pipelines/run_backtest.py \
    --config config/default_config.yaml \
    --model_type pso_lstm \
    --model_path results/pso_lstm/AAPL_1Min \
    --ticker AAPL --timeframe 1Min \
    --data-path data/features_v2/AAPL \
    --processed-dir data/processed \
    --run-id pso-001
```

`--model_path` here is the Phase-2 output directory containing
`pso_lstm_model.pt` and `model_config.yaml`.

### 4c. XGBoost

```bash
python pipelines/run_backtest.py \
    --config config/default_config.yaml \
    --model_type xgboost \
    --model_path results/experiments/AAPL_1Min_xgboost_xgb-001/train \
    --ticker AAPL --timeframe 1Min \
    --data-path data/features_v2/AAPL \
    --processed-dir data/processed \
    --run-id xgb-001
```

`--model_path` directory must contain `xgboost_model.json`,
`feature_names.json`, and `feature_importance.json`.

---

## What to expect from training

After stage 3a runs you can sanity-check that the optimiser is finding real
signal rather than collapsing to the constant-mean predictor (the failure
mode the new defaults guard against):

```bash
python - <<'PY'
import json
from pathlib import Path
run = Path("results/experiments/AAPL_1Min_lstm_baseline_<RUN_ID>/train")
h = json.load(open(run / "training_history.json"))["history"]
print("variance_ratio per epoch:", [f"{v:.3f}" for v in h["variance_ratio"]])
print("val loss per epoch:      ", [f"{v:.4f}" for v in h["val_loss"]])
print("val DA per epoch:        ", [f"{m['directional_accuracy']:.3f}" if m['directional_accuracy'] is not None else 'nan' for m in h["val_metrics"]])
PY
```

Healthy signs:
- `val_loss` steadily decreasing (with z-scored targets, MSE for the
  constant predictor is ~1.0; anything below 1.0 is real signal).
- `variance_ratio` climbing from near-zero toward 0.3+ over the first
  ~20 epochs (predictions are no longer flat).
- `val_metrics[…].directional_accuracy` rising above 0.50.

Trouble signs:
- `variance_ratio` stuck below 0.05 → predictions are still collapsing.
  Try `directional_loss_weight: 0.5` in `default_config.yaml` or switch
  loss to `huber`.
- `val_loss` strictly increasing → likely overfitting. Reduce
  `lstm_units_*` or raise dropout to ~0.1.
- `val_metrics[…].directional_accuracy` exactly 0.50 → constant-predict
  collapse confirmed.

---

## Smoke-test path

Before committing to the full run, shrink everything:

1. In `config/default_config.yaml`, swap to the commented-in short window
   (`start_date: "2026-01-05"`, `end_date: "2026-04-05"`).
2. Trim `config/tickers.txt` to one ticker plus the SPY benchmark.
3. Run the four stages:

```bash
python pipelines/data_ingest_align_clean.py
python pipelines/run_build_features.py --timeframe 1Min --target-method log_return --workers 1
python pipelines/run_lstm.py --ticker AAPL --timeframe 1Min \
    --data-path data/features_v2/AAPL --device cpu --run-id smoke
python pipelines/run_backtest.py --model_type lstm_baseline \
    --model_path results/experiments/AAPL_1Min_lstm_baseline_smoke/train \
    --ticker AAPL --timeframe 1Min --data-path data/features_v2/AAPL \
    --run-id smoke
```

If that succeeds, restore the full date range, bump `--workers` to fit
your machine, and re-run with `--skip-existing` on stage 1 to fill in
only the missing chunks.

---

## Common issues

- **`ModuleNotFoundError`**: re-activate `633` and `pip install -r requirements.txt`.
  New deps in this branch: `prettytable`, `alpaca-py`, `pandas-market-calendars`.
- **`Missing Alpaca API credentials`**: `.env` must be at the repo root and
  the process must be started from there (the ingestor calls `load_dotenv()`
  with no path arg).
- **Empty trainer features**: rerun stage 2 with `--timeframe` matching the
  one used downstream — running multiple timeframes in one stage 2
  invocation would have overwritten earlier ones.
- **Stage 2 OOM under `--workers > 1`**: each worker holds the full ticker
  in memory at once. Drop `--workers` until the run fits, or shrink the
  date range.
- **Stage 2 worker silently fails**: per-worker logs are at
  `data/features_v2/<TICKER>/logs/build_features.log`. The parent log only
  records the dispatch summary; failures are listed as
  `<TICKER>: failed (<error>)`.
- **Val metrics flat / RMSE near zero**: see "What to expect from training"
  above. The new directional loss + standardised target + low dropout
  defaults are designed to avoid this; if you see it anyway, `variance_ratio
  < 0.05` confirms collapse and the troubleshooting block has next steps.
- **Backtest model-path errors**: each model type expects a different
  directory layout (see §4a-c). LSTM baseline → the `train/` subdir;
  PSO-LSTM → the Phase-2 output dir; XGBoost → the `train/` subdir with
  its three JSON sidecars.
