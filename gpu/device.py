"""
Unified device detection for PyTorch.

Priority: CUDA → MPS (Apple Silicon) → CPU.
"""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_info() -> dict:
    device = get_device()
    info = {"device": str(device), "torch_version": torch.__version__}

    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1)
        info["cuda_version"] = torch.version.cuda or "unknown"
    elif device.type == "mps":
        info["gpu_name"] = "Apple Silicon (Metal Performance Shaders)"
        info["note"] = "Unified memory architecture"

    return info
