#!/bin/bash
# Feature engineering pipeline: build features from aligned data

set -e

CONFIG="${1:-config/default_config.yaml}"
TICKERS="${2:-config/tickers.txt}"
N_JOBS="${3:-1}"

echo "=========================================="
echo "Feature Engineering Pipeline"
echo "=========================================="
echo ""

echo "Building features with ${N_JOBS} parallel jobs..."
python pipelines/build_features.py \
    --config ${CONFIG} \
    --input data/processed/aligned_universe.parquet \
    --output data/features \
    --tickers ${TICKERS} \
    --n-jobs ${N_JOBS}

echo ""
echo "=========================================="
echo "✓ Feature Engineering Complete!"
echo "=========================================="
echo ""
echo "Outputs:"
echo "  - Features: data/features/"
