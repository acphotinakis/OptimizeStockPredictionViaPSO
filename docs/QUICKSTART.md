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
target/features computed per-split, fits the scaler + 3-stage feature
selector on **train only**, and serialises the frozen pipeline.

```bash
python pipelines/run_build_features.py \
    --config config/default_config.yaml \
    --tickers config/tickers.txt \
    --processed-dir data/processed \
    --output data/features_v2 \
    --timeframe 1Min \
    --target-method log_return
```

Writes per ticker:

```
data/features_v2/<TICKER>/
    X_train.npy      y_train.npy
    X_val.npy        y_val.npy
    X_test.npy       y_test.npy
    test_index.npy
    frozen_pipeline.pkl
    feature_scaler.joblib    target_scaler.joblib
    feature_selector.joblib  selected_features.json
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

Currently wired for `lstm_baseline`. The CLI accepts `pso_lstm` and
`xgboost` but raises `NotImplementedError` until those branches are added
to `pipelines/run_backtest.py:main()`.

```bash
python pipelines/run_backtest.py \
    --config config/default_config.yaml \
    --model_type lstm_baseline \
    --model_path results/experiments/AAPL_1Min_lstm_baseline_baseline-001/train \
    --ticker AAPL \
    --timeframe 1Min \
    --data-path data/features_v2/AAPL \
    --processed-dir data/processed \
    --run-id baseline-001
```

`--model_path` is the **train directory** (it must contain
`baseline_lstm_model.pt` and `model_config.json`).

Writes `metrics.json`, an equity-curve CSV, a trade log, and plots under
`results/experiments/<TICKER>_<TF>_lstm_baseline_<RUN_ID>/backtest/`.

---

## Smoke-test path

Before committing to the full run, shrink everything:

1. In `config/default_config.yaml`, swap to the commented-in short window
   (`start_date: "2026-01-05"`, `end_date: "2026-04-05"`).
2. Trim `config/tickers.txt` to one ticker plus the SPY benchmark.
3. Run the four stages:

```bash
python pipelines/data_ingest_align_clean.py
python pipelines/run_build_features.py --timeframe 1Min --target-method log_return
python pipelines/run_lstm.py --ticker AAPL --timeframe 1Min \
    --data-path data/features_v2/AAPL --device cpu --run-id smoke
python pipelines/run_backtest.py --model_type lstm_baseline \
    --model_path results/experiments/AAPL_1Min_lstm_baseline_smoke/train \
    --ticker AAPL --timeframe 1Min --data-path data/features_v2/AAPL \
    --run-id smoke
```

If that succeeds, restore the full date range and re-run with
`--skip-existing` on stage 1 to fill in only the missing chunks.

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
- **Backtest `NotImplementedError` for `pso_lstm`/`xgboost`**: the dispatcher
  in `pipelines/run_backtest.py:main()` only wires `lstm_baseline`. Extend
  the `elif` block using the existing `LSTMAdapter`/`XGBoostAdapter` in
  `src/evaluation/model_loader.py` as the pattern.
