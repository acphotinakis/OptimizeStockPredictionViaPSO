#!/bin/bash
# Data ingestion pipeline: download, clean, and align stock data

set -e

CONFIG="${1:-config/default_config.yaml}"
TICKERS="${2:-config/tickers.txt}"

echo "=========================================="
echo "Data Ingestion Pipeline"
echo "=========================================="
echo ""

echo "Step 1/3: Downloading raw data..."
python pipelines/ingest_data.py \
    --mode ingest \
    --config ${CONFIG} \
    --tickers ${TICKERS} \
    --raw-output data/raw

echo "✓ Download complete"
echo ""

echo "Step 2/3: Cleaning data..."
python pipelines/ingest_data.py \
    --mode clean \
    --config ${CONFIG} \
    --tickers ${TICKERS} \
    --raw-output data/raw \
    --cleaned-output data/cleaned

echo "✓ Cleaning complete"
echo ""

echo "Step 3/3: Aligning data to SPY timestamps..."
python pipelines/ingest_data.py \
    --mode align \
    --config ${CONFIG} \
    --tickers ${TICKERS} \
    --cleaned-output data/cleaned \
    --processed-output data/processed

echo "✓ Alignment complete"
echo ""

echo "=========================================="
echo "✓ Data Ingestion Complete!"
echo "=========================================="
echo ""
echo "Outputs:"
echo "  - Raw data: data/raw/"
echo "  - Cleaned data: data/cleaned/"
echo "  - Aligned data: data/processed/"
