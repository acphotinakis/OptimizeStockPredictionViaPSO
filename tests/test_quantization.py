#!/usr/bin/env python3
"""
scripts/test_quantization.py

Test script to verify quantization implementation.
Measures memory usage, model size, and accuracy impact.

Usage:
    python scripts/test_quantization.py --ticker AAPL
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.lstm.lstm_model import LSTMModel, LSTMTrainer
from backup.quantized_lstm import QuantizedLSTMModel
from src.utils.logger import setup_logger
from src.utils.memory_profiler import MemoryProfiler, MemoryMonitor
from src.utils.memory_manager import MemoryManager
from src.utils.seed import set_all_seeds

logger = logging.getLogger(__name__)


def test_data_quantization():
    """Test Phase 1: Data quantization (float64 --> float32)."""
    logger.info("\n" + "=" * 60)
    logger.info("Phase 1: Data Quantization Test")
    logger.info("=" * 60)

    # Create sample data
    n_samples = 10000
    n_features = 100

    # Float64 (default)
    data_f64 = np.random.randn(n_samples, n_features).astype(np.float64)
    size_f64 = data_f64.nbytes / 1e6  # MB

    # Float32 (quantized)
    data_f32 = data_f64.astype(np.float32)
    size_f32 = data_f32.nbytes / 1e6  # MB

    reduction = (1 - size_f32 / size_f64) * 100

    logger.info("Float64 size: %.2f MB", size_f64)
    logger.info("Float32 size: %.2f MB", size_f32)
    logger.info("Memory reduction: %.1f%%", reduction)

    # Check numerical precision
    max_error = np.max(np.abs(data_f64 - data_f32))
    logger.info("Max numerical error: %.2e", max_error)

    assert reduction > 45, "Expected ~50% reduction"
    logger.info("✓ Phase 1 test passed")


def test_model_quantization():
    """Test Phase 2: Model quantization (float32 --> int8)."""
    logger.info("\n" + "=" * 60)
    logger.info("Phase 2: Model Quantization Test")
    logger.info("=" * 60)

    # Create a small LSTM model
    input_size = 50
    hidden_units = 128
    num_layers = 2

    model = LSTMModel(
        input_size=input_size,
        num_layers=num_layers,
        hidden_units=hidden_units,
        dropout=0.2,
    )

    # Get original model size
    original_size = QuantizedLSTMModel._get_model_size(model)
    logger.info("Original model size: %.2f MB", original_size)

    # Quantize
    quantized_wrapper = QuantizedLSTMModel(model)
    quantized_model = quantized_wrapper.quantize()

    # Get quantized model size
    quantized_size = QuantizedLSTMModel._get_model_size(quantized_model)
    logger.info("Quantized model size: %.2f MB", quantized_size)

    reduction = (1 - quantized_size / original_size) * 100
    logger.info("Model size reduction: %.1f%%", reduction)

    # Test inference
    dummy_input = np.random.randn(10, 30, input_size).astype(np.float32)

    model.eval()
    with torch.no_grad():
        original_output = model(torch.FloatTensor(dummy_input)).numpy()

    quantized_output = quantized_wrapper.predict(dummy_input)

    # Check accuracy
    mae = np.mean(np.abs(original_output.flatten() - quantized_output))
    logger.info("Mean absolute error (original vs quantized): %.6f", mae)

    assert reduction > 50, "Expected ~75% reduction"
    assert mae < 0.01, "Quantization error too large"
    logger.info("✓ Phase 2 test passed")


def test_mixed_precision_training():
    """Test Phase 3: Mixed precision training."""
    logger.info("\n" + "=" * 60)
    logger.info("Phase 3: Mixed Precision Training Test")
    logger.info("=" * 60)

    if not torch.cuda.is_available():
        logger.info("GPU not available, skipping mixed precision test")
        return

    # Create synthetic training data
    n_samples = 1000
    seq_len = 30
    input_size = 50

    X_train = np.random.randn(n_samples, seq_len, input_size).astype(np.float32)
    y_train = np.random.randn(n_samples).astype(np.float32)
    X_val = np.random.randn(200, seq_len, input_size).astype(np.float32)
    y_val = np.random.randn(200).astype(np.float32)

    # Train with AMP
    model_amp = LSTMModel(
        input_size=input_size,
        num_layers=2,
        hidden_units=64,
        dropout=0.2,
    )

    with MemoryMonitor("Training with AMP"):
        trainer_amp = LSTMTrainer(
            model=model_amp,
            lr=0.001,
            max_epochs=5,
            batch_size=64,
            use_amp=True,
        )
        trainer_amp.fit(X_train, y_train, X_val, y_val)

    peak_mem_amp = MemoryProfiler.get_peak_memory()
    logger.info("Peak GPU memory with AMP: %.2f GB", peak_mem_amp["gpu_peak_gb"])

    # Train without AMP
    MemoryManager.clear_all()
    MemoryProfiler.reset_peak_memory()

    model_fp32 = LSTMModel(
        input_size=input_size,
        num_layers=2,
        hidden_units=64,
        dropout=0.2,
    )

    with MemoryMonitor("Training without AMP"):
        trainer_fp32 = LSTMTrainer(
            model=model_fp32,
            lr=0.001,
            max_epochs=5,
            batch_size=64,
            use_amp=False,
        )
        trainer_fp32.fit(X_train, y_train, X_val, y_val)

    peak_mem_fp32 = MemoryProfiler.get_peak_memory()
    logger.info("Peak GPU memory without AMP: %.2f GB", peak_mem_fp32["gpu_peak_gb"])

    if peak_mem_amp["gpu_peak_gb"] > 0 and peak_mem_fp32["gpu_peak_gb"] > 0:
        reduction = (
            1 - peak_mem_amp["gpu_peak_gb"] / peak_mem_fp32["gpu_peak_gb"]
        ) * 100
        logger.info("Memory reduction with AMP: %.1f%%", reduction)

    logger.info("✓ Phase 3 test passed")


def test_gradient_checkpointing():
    """Test Phase 3: Gradient checkpointing."""
    logger.info("\n" + "=" * 60)
    logger.info("Phase 3: Gradient Checkpointing Test")
    logger.info("=" * 60)

    # Create synthetic data
    n_samples = 500
    seq_len = 60  # Longer sequence
    input_size = 100

    X_train = np.random.randn(n_samples, seq_len, input_size).astype(np.float32)
    y_train = np.random.randn(n_samples).astype(np.float32)
    X_val = np.random.randn(100, seq_len, input_size).astype(np.float32)
    y_val = np.random.randn(100).astype(np.float32)

    # With checkpointing
    model_cp = LSTMModel(
        input_size=input_size,
        num_layers=3,
        hidden_units=128,
        dropout=0.2,
        use_checkpointing=True,
    )

    with MemoryMonitor("Training with checkpointing"):
        trainer_cp = LSTMTrainer(
            model=model_cp,
            lr=0.001,
            max_epochs=3,
            batch_size=32,
        )
        trainer_cp.fit(X_train, y_train, X_val, y_val)

    logger.info("✓ Gradient checkpointing test passed")


def test_memory_profiler():
    """Test Phase 4: Memory profiling utilities."""
    logger.info("\n" + "=" * 60)
    logger.info("Phase 4: Memory Profiler Test")
    logger.info("=" * 60)

    # Test get_memory_usage
    mem = MemoryProfiler.get_memory_usage()
    logger.info(
        "Current memory: CPU %.2f GB | GPU %.2f GB", mem["cpu_gb"], mem["gpu_gb"]
    )

    # Test MemoryMonitor context manager
    with MemoryMonitor("Test operation"):
        data = np.random.randn(10000, 1000)
        _ = data @ data.T

    # Test profile decorator
    @MemoryProfiler.profile
    def dummy_function():
        return np.random.randn(5000, 5000)

    _ = dummy_function()

    logger.info("✓ Phase 4 test passed")


def main():
    parser = argparse.ArgumentParser(description="Test quantization implementation")
    parser.add_argument(
        "--phase",
        type=str,
        choices=["all", "data", "model", "training", "checkpointing", "profiler"],
        default="all",
        help="Which phase to test",
    )
    args = parser.parse_args()

    setup_logger(log_file="logs/test_quantization.log", level="INFO")
    set_all_seeds(42)

    logger.info("=" * 60)
    logger.info("Quantization Implementation Tests")
    logger.info("=" * 60)

    MemoryProfiler.log_memory("Initial")

    try:
        if args.phase in ["all", "data"]:
            test_data_quantization()

        if args.phase in ["all", "model"]:
            test_model_quantization()

        if args.phase in ["all", "training"]:
            test_mixed_precision_training()

        if args.phase in ["all", "checkpointing"]:
            test_gradient_checkpointing()

        if args.phase in ["all", "profiler"]:
            test_memory_profiler()

        logger.info("\n" + "=" * 60)
        logger.info("✓ All tests passed!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("Test failed: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
