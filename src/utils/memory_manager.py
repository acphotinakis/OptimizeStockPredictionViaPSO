"""
src/utils/memory_manager.py

Automatic memory management utilities for training loops.
Phase 4: Configuration & Utilities (from QUANTIZATION.md)
"""

from __future__ import annotations

import gc
import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MemoryManager:
    """Automatic memory management for training loops."""
    
    @staticmethod
    def clear_all() -> None:
        """Clear all caches and run garbage collection.
        
        Call this between PSO particle evaluations or between epochs
        to free up unused memory.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @staticmethod
    def get_optimal_batch_size(
        model: nn.Module,
        input_shape: Tuple[int, int],
        max_batch: int = 512,
        min_batch: int = 8,
    ) -> int:
        """Binary search for optimal batch size that fits in GPU memory.
        
        Args:
            model: PyTorch model to test.
            input_shape: (sequence_length, feature_dim) tuple.
            max_batch: Maximum batch size to try.
            min_batch: Minimum acceptable batch size.
        
        Returns:
            Optimal batch size that fits in memory.
        """
        device = next(model.parameters()).device
        if device.type == "cpu":
            return max_batch  # No GPU memory constraint
        
        logger.info("Searching for optimal batch size (max=%d)...", max_batch)
        batch_size = max_batch
        
        while batch_size >= min_batch:
            try:
                # Test with dummy data
                dummy_input = torch.randn(batch_size, *input_shape).to(device)
                model.train()
                output = model(dummy_input)
                loss = output.sum()
                loss.backward()
                
                # Success - this batch size works
                MemoryManager.clear_all()
                logger.info("Optimal batch size found: %d", batch_size)
                return batch_size
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Try smaller batch
                    batch_size = batch_size // 2
                    MemoryManager.clear_all()
                    logger.debug("Batch size %d too large, trying %d", batch_size * 2, batch_size)
                else:
                    raise
        
        logger.warning("Could not find optimal batch size, using minimum: %d", min_batch)
        return min_batch
    
    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        """Get detailed GPU memory statistics.
        
        Returns:
            Dict with memory statistics in GB.
        """
        if not torch.cuda.is_available():
            return {}
        
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "max_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
            "total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        }
    
    @staticmethod
    def log_memory_stats(prefix: str = "") -> None:
        """Log detailed GPU memory statistics."""
        stats = MemoryManager.get_memory_stats()
        if not stats:
            return
        
        utilization = (stats["allocated_gb"] / stats["total_gb"]) * 100
        
        logger.info(
            "%sGPU Memory: %.2f / %.2f GB (%.1f%% used) | "
            "Peak: %.2f GB | Reserved: %.2f GB",
            f"{prefix} " if prefix else "",
            stats["allocated_gb"],
            stats["total_gb"],
            utilization,
            stats["max_allocated_gb"],
            stats["reserved_gb"],
        )
    
    @staticmethod
    def check_memory_available(required_gb: float) -> bool:
        """Check if sufficient GPU memory is available.
        
        Args:
            required_gb: Required memory in GB.
        
        Returns:
            True if sufficient memory available.
        """
        if not torch.cuda.is_available():
            return True  # No GPU constraint
        
        stats = MemoryManager.get_memory_stats()
        available = stats["total_gb"] - stats["allocated_gb"]
        
        if available < required_gb:
            logger.warning(
                "Insufficient GPU memory: %.2f GB available, %.2f GB required",
                available, required_gb
            )
            return False
        
        return True
