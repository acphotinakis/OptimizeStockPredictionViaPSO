#!/bin/bash
# Complete workflow for 6-ticker PSO-LSTM analysis with signal generation and plotting

set -e  # Exit on error

SEED=42
RESULTS_DIR="results"

# Define tickers organized by sector
TECH_TICKERS=("AAPL" "MSFT" "GOOGL" "NVDA" "TSLA")
BENCHMARK="SPY"

ALL_TICKERS=("${TECH_TICKERS[@]}" "${BENCHMARK}")

echo "=========================================="
echo "6-Ticker PSO-LSTM Analysis Pipeline"
echo "=========================================="
echo ""

for TICKER in "${ALL_TICKERS[@]}"; do
    echo "------------------------------------------"
    echo "Processing ${TICKER}..."
    echo "------------------------------------------"
    
    # Check if features exist
    if [ ! -d "data/features/${TICKER}" ]; then
        echo "✗ Error: Features not found for ${TICKER} in data/features/${TICKER}"
        echo "Please run feature engineering first:"
        echo "  python pipelines/build_features.py --config config/default_config.yaml"
        exit 1
    fi

    echo "Step 1/4: Training LSTM model for ${TICKER}..."
    python pipelines/run_lstm_baseline.py \
        --ticker ${TICKER} \
        --mode train \
        --seed ${SEED} \
        --results-dir ${RESULTS_DIR}

    echo "✓ Training complete"
    echo ""

    echo "Step 2/4: Running validation for ${TICKER}..."
    python pipelines/run_lstm_baseline.py \
        --ticker ${TICKER} \
        --mode val \
        --seed ${SEED} \
        --results-dir ${RESULTS_DIR}

    echo "✓ Validation complete"
    echo ""

    echo "Step 3/4: Testing and generating signals for ${TICKER}..."
    python pipelines/run_lstm_baseline.py \
        --ticker ${TICKER} \
        --mode test \
        --seed ${SEED} \
        --results-dir ${RESULTS_DIR}

    echo "✓ Testing complete"
    echo ""

    echo "Step 4/4: Generating visualization for ${TICKER}..."
    python plots/plot_lstm.py \
        --ticker ${TICKER} \
        --seed ${SEED} \
        --results-dir ${RESULTS_DIR}

    echo "✓ Plotting complete"
    echo ""

    echo "Results for ${TICKER} saved in:"
    echo "  - Model: ${RESULTS_DIR}/lstm_baseline_model_${TICKER}_train_seed${SEED}.pth"
    echo "  - Predictions: ${RESULTS_DIR}/lstm_aligned_${TICKER}_test_seed${SEED}.csv"
    echo "  - Signals: ${RESULTS_DIR}/lstm_baseline_test_signals_${TICKER}_test_seed${SEED}.npy"
    echo "  - Plot: ${RESULTS_DIR}/plots/lstm_ohlcv_${TICKER}_seed${SEED}.png"
    echo "  - Summary: ${RESULTS_DIR}/plots/lstm_ohlcv_${TICKER}_seed${SEED}.txt"
    echo ""
done

echo "=========================================="
echo "✓ 6-Ticker PSO-LSTM Analysis Complete!"
echo "=========================================="
