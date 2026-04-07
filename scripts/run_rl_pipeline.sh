#!/bin/bash
# Complete RL training and evaluation pipeline

set -e

TICKER="${1:-SPY}"
EPISODES="${2:-500}"
SEED=42

echo "=========================================="
echo "RL Trading Agent Pipeline"
echo "Ticker: $TICKER"
echo "Episodes: $EPISODES"
echo "=========================================="
echo ""

# Check if features exist
if [ ! -d "data/features/${TICKER}" ]; then
    echo "✗ Error: Features not found for ${TICKER}"
    echo ""
    echo "Please run feature engineering first:"
    echo "  python scripts/build_features.py --config config/default_config.yaml"
    exit 1
fi

echo "Step 1/2: Training RL agent..."
python scripts/train_rl_agent.py \
    --ticker ${TICKER} \
    --episodes ${EPISODES} \
    --eval-freq 50 \
    --save-freq 100 \
    --seed ${SEED}

if [ $? -ne 0 ]; then
    echo "✗ Training failed"
    exit 1
fi
echo "✓ Training complete"
echo ""

echo "Step 2/2: Evaluating on test set..."
python scripts/evaluate_rl_agent.py \
    --ticker ${TICKER}

if [ $? -ne 0 ]; then
    echo "✗ Evaluation failed"
    exit 1
fi
echo "✓ Evaluation complete"
echo ""

echo "=========================================="
echo "✓ RL Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  - Model: results/rl/checkpoints/rl_agent_${TICKER}_best.pth"
echo "  - Training curves: results/rl/plots/rl_training_curves_${TICKER}.png"
echo "  - Final evaluation: results/rl/plots/rl_final_evaluation_${TICKER}.png"
echo "  - Test comparison: results/rl/plots/rl_test_comparison_${TICKER}.png"
echo "  - Training history: results/rl/rl_training_history_${TICKER}.csv"
echo "  - Test results: results/rl/rl_test_results_${TICKER}.json"
echo ""
