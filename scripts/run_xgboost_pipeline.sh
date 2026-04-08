#!/bin/bash
# Complete XGBoost training, validation, and testing pipeline

set -e

TICKER="${1:-SPY}"
SEED="${2:-42}"
TRAIN_MODE="${3:-default}"

echo "=========================================="
echo "XGBoost Pipeline"
echo "Ticker: ${TICKER}"
echo "Seed: ${SEED}"
echo "Train Mode: ${TRAIN_MODE}"
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

echo "Step 1/3: Training XGBoost model..."
python pipelines/run_xgboost.py \
    --ticker ${TICKER} \
    --mode train \
    --train-mode ${TRAIN_MODE} \
    --seed ${SEED}

echo "✓ Training complete"
echo ""

echo "Step 2/3: Running validation..."
python pipelines/run_xgboost.py \
    --ticker ${TICKER} \
    --mode val \
    --seed ${SEED}

echo "✓ Validation complete"
echo ""

echo "Step 3/3: Testing on test set..."
python pipelines/run_xgboost.py \
    --ticker ${TICKER} \
    --mode test \
    --seed ${SEED}

echo "✓ Testing complete"
echo ""

echo "=========================================="
echo "✓ XGBoost Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  - Model: results/xgb_model_${TICKER}_train_seed${SEED}.ubj"
echo "  - Metrics: results/xgb_test_metrics_${TICKER}_test_seed${SEED}.json"
echo "  - Plots: results/plots/"
