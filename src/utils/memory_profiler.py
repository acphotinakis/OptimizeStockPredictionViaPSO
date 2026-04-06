"""
src/utils/memory_profiler.py

Memory profiling utilities for tracking CPU and GPU memory usage.
Phase 4: Configuration & Utilities (from QUANTIZATION.md)
"""

from __future__ import annotations

import gc
import logging
from functools import wraps
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class MemoryProfiler:
    """Track CPU and GPU memory usage."""
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage.
        
        Returns:
            Dict with 'cpu_gb' and 'gpu_gb' keys.
        """
        # CPU memory
        try:
            import psutil
            cpu_mem = psutil.Process().memory_info().rss / 1e9  # GB
        except ImportError:
            cpu_mem = 0.0
        
        # GPU memory
        gpu_mem = 0.0
        gpu_allocated = 0.0
        gpu_reserved = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                gpu_allocated = torch.cuda.memory_allocated() / 1e9  # GB
                gpu_reserved = torch.cuda.memory_reserved() / 1e9  # GB
                gpu_mem = gpu_allocated
        except ImportError:
            pass
        
        return {
            "cpu_gb": cpu_mem,
            "gpu_gb": gpu_mem,
            "gpu_allocated_gb": gpu_allocated,
            "gpu_reserved_gb": gpu_reserved,
        }
    
    @staticmethod
    def log_memory(prefix: str = "") -> None:
        """Log current memory usage.
        
        Args:
            prefix: Optional prefix for log message.
        """
        mem = MemoryProfiler.get_memory_usage()
        msg = f"{prefix} Memory: CPU {mem['cpu_gb']:.2f} GB"
        if mem['gpu_gb'] > 0:
            msg += f" | GPU {mem['gpu_gb']:.2f} GB (allocated) / {mem['gpu_reserved_gb']:.2f} GB (reserved)"
        logger.info(msg)
    
    @staticmethod
    def profile(func):
        """Decorator to profile memory usage of a function.
        
        Usage:
            @MemoryProfiler.profile
            def my_function():
                # Your code here
                pass
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            mem_before = MemoryProfiler.get_memory_usage()
            result = func(*args, **kwargs)
            mem_after = MemoryProfiler.get_memory_usage()
            
            cpu_delta = mem_after["cpu_gb"] - mem_before["cpu_gb"]
            gpu_delta = mem_after["gpu_gb"] - mem_before["gpu_gb"]
            
            logger.info(
                "[%s] Memory: CPU %.2f GB → %.2f GB (Δ%+.2f GB) | "
                "GPU %.2f GB → %.2f GB (Δ%+.2f GB)",
                func.__name__,
                mem_before["cpu_gb"], mem_after["cpu_gb"], cpu_delta,
                mem_before["gpu_gb"], mem_after["gpu_gb"], gpu_delta
            )
            return result
        return wrapper
    
    @staticmethod
    def get_peak_memory() -> Dict[str, float]:
        """Get peak memory usage since last reset.
        
        Returns:
            Dict with peak memory values.
        """
        result = {"cpu_peak_gb": 0.0, "gpu_peak_gb": 0.0}
        
        try:
            import torch
            if torch.cuda.is_available():
                result["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        except ImportError:
            pass
        
        return result
    
    @staticmethod
    def reset_peak_memory() -> None:
        """Reset peak memory tracking."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                logger.debug("Reset GPU peak memory stats")
        except ImportError:
            pass


class MemoryMonitor:
    """Context manager for monitoring memory usage.
    
    Usage:
        with MemoryMonitor("Training LSTM"):
            trainer.fit(X_train, y_train, X_val, y_val)
    """
    
    def __init__(self, operation_name: str = "Operation"):
        self.operation_name = operation_name
        self.mem_before: Optional[Dict] = None
        self.mem_after: Optional[Dict] = None
    
    def __enter__(self):
        MemoryProfiler.reset_peak_memory()
        self.mem_before = MemoryProfiler.get_memory_usage()
        logger.info("[%s] Starting - Memory: CPU %.2f GB | GPU %.2f GB",
                   self.operation_name,
                   self.mem_before["cpu_gb"],
                   self.mem_before["gpu_gb"])
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.mem_after = MemoryProfiler.get_memory_usage()
        peak = MemoryProfiler.get_peak_memory()
        
        cpu_delta = self.mem_after["cpu_gb"] - self.mem_before["cpu_gb"]
        gpu_delta = self.mem_after["gpu_gb"] - self.mem_before["gpu_gb"]
        
        logger.info(
            "[%s] Complete - Memory: CPU Δ%+.2f GB | GPU Δ%+.2f GB (peak %.2f GB)",
            self.operation_name, cpu_delta, gpu_delta, peak["gpu_peak_gb"]
        )
