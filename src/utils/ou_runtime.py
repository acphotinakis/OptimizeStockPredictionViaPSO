from __future__ import annotations

import random
import numpy as np
import torch


def configure_torch_runtime(device: torch.device, deterministic: bool = False, amp: bool = False) -> None:
    """Set safe PyTorch runtime options for faster GPU execution when available."""
    if device.type == "cuda":
        if not deterministic:
            torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
        print(f"AMP enabled: {bool(amp)}")


def resolve_device(device_arg: str = "auto") -> torch.device:
    """Choose CPU/CUDA/MPS device."""
    device_arg = (device_arg or "auto").lower()
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False.")
    if device_arg == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("--device mps was requested, but MPS is not available.")
    return torch.device(device_arg)


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
