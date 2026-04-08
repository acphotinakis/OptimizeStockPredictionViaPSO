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

    echo "Running LSTM pipeline for ${TICKER}..."
    bash scripts/run_lstm_pipeline.sh ${TICKER} ${SEED}
    
    echo "✓ Pipeline complete for ${TICKER}"
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
