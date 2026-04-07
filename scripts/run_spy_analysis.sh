#!/bin/bash
# Complete workflow for SPY LSTM analysis with signal generation and plotting

set -e  # Exit on error

TICKER="SPY"
SEED=42
RESULTS_DIR="results"

echo "=========================================="
echo "SPY LSTM Analysis Pipeline"
echo "=========================================="
echo ""

# Check if SPY features exist
if [ ! -d "data/features/${TICKER}" ]; then
    echo "✗ Error: SPY features not found in data/features/${TICKER}"
    echo ""
    echo "Please run feature engineering first:"
    echo "  python scripts/build_features.py --config config/default_config.yaml"
    exit 1
fi

echo "Step 1/4: Training LSTM model for ${TICKER}..."
python scripts/run_lstm_baseline.py \
    --ticker ${TICKER} \
    --mode train \
    --seed ${SEED} \
    --results-dir ${RESULTS_DIR}

if [ $? -ne 0 ]; then
    echo "✗ Training failed"
    exit 1
fi
echo "✓ Training complete"
echo ""

echo "Step 2/4: Running validation..."
python scripts/run_lstm_baseline.py \
    --ticker ${TICKER} \
    --mode val \
    --seed ${SEED} \
    --results-dir ${RESULTS_DIR}

if [ $? -ne 0 ]; then
    echo "✗ Validation failed"
    exit 1
fi
echo "✓ Validation complete"
echo ""

echo "Step 3/4: Testing and generating signals..."
python scripts/run_lstm_baseline.py \
    --ticker ${TICKER} \
    --mode test \
    --seed ${SEED} \
    --results-dir ${RESULTS_DIR}

if [ $? -ne 0 ]; then
    echo "✗ Testing failed"
    exit 1
fi
echo "✓ Testing complete"
echo ""

echo "Step 4/4: Generating visualization..."
python plots/plot_lstm.py \
    --ticker ${TICKER} \
    --seed ${SEED} \
    --results-dir ${RESULTS_DIR}

if [ $? -ne 0 ]; then
    echo "✗ Plotting failed"
    exit 1
fi
echo "✓ Plotting complete"
echo ""

echo "=========================================="
echo "✓ SPY Analysis Complete!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  - Model: ${RESULTS_DIR}/lstm_baseline_model_${TICKER}_train_seed${SEED}.pth"
echo "  - Predictions: ${RESULTS_DIR}/lstm_aligned_${TICKER}_test_seed${SEED}.csv"
echo "  - Signals: ${RESULTS_DIR}/lstm_baseline_test_signals_${TICKER}_test_seed${SEED}.npy"
echo "  - Plot: ${RESULTS_DIR}/plots/lstm_ohlcv_${TICKER}_seed${SEED}.png"
echo "  - Summary: ${RESULTS_DIR}/plots/lstm_ohlcv_${TICKER}_seed${SEED}.txt"
echo ""
