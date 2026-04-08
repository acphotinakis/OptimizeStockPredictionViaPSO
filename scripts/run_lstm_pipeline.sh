#!/bin/bash
# Complete LSTM training, validation, and testing pipeline

set -e

TICKER="${1:-SPY}"
SEED="${2:-42}"

echo "=========================================="
echo "LSTM Pipeline"
echo "Ticker: ${TICKER}"
echo "Seed: ${SEED}"
echo "=========================================="
echo ""

# Check if features exist
if [ ! -d "data/features/${TICKER}" ]; then
    echo "✗ Error: Features not found for ${TICKER}"
    echo ""
    echo "Please run feature engineering first:"
    echo "  bash scripts/run_feature_engineering.sh"
    exit 1
fi

echo "Step 1/3: Training LSTM model..."
python pipelines/run_lstm_baseline.py \
    --ticker ${TICKER} \
    --mode train \
    --seed ${SEED}

echo "✓ Training complete"
echo ""

echo "Step 2/3: Running validation..."
python pipelines/run_lstm_baseline.py \
    --ticker ${TICKER} \
    --mode val \
    --seed ${SEED}

echo "✓ Validation complete"
echo ""

echo "Step 3/3: Testing on test set..."
python pipelines/run_lstm_baseline.py \
    --ticker ${TICKER} \
    --mode test \
    --seed ${SEED}

echo "✓ Testing complete"
echo ""

echo "=========================================="
echo "✓ LSTM Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  - Model: results/lstm_baseline_model_${TICKER}_train_seed${SEED}.pth"
echo "  - Predictions: results/lstm_aligned_${TICKER}_test_seed${SEED}.csv"
echo "  - Plots: results/plots/"
