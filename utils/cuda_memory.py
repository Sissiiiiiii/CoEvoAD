from __future__ import annotations

from typing import Any, Dict, Optional

import torch


_MIB = 1024 * 1024


def _mib(value_bytes: int) -> float:
    return float(value_bytes) / _MIB


def reset_cuda_peak_memory(device: Optional[int] = None) -> bool:
    """Reset PyTorch CUDA peak counters for the current process."""
    if not torch.cuda.is_available():
        return False
    torch.cuda.reset_peak_memory_stats(device)
    return True


def collect_cuda_peak_memory(device: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Collect PyTorch CUDA memory counters for the current process."""
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize(device)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    current_allocated = int(torch.cuda.memory_allocated(device))
    current_reserved = int(torch.cuda.memory_reserved(device))
    return {
        "device": device,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "current_allocated_bytes": current_allocated,
        "current_reserved_bytes": current_reserved,
        "peak_allocated_mib": _mib(peak_allocated),
        "peak_reserved_mib": _mib(peak_reserved),
        "current_allocated_mib": _mib(current_allocated),
        "current_reserved_mib": _mib(current_reserved),
    }


def log_cuda_peak_memory(logger, label: str, device: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Log CUDA peak counters in a stable, grep-friendly format."""
    stats = collect_cuda_peak_memory(device)
    if stats is None:
        logger.info("CUDA_PEAK_MEMORY|label=%s|available=0", label)
        return None

    logger.info(
        "CUDA_PEAK_MEMORY|label=%s|device=%s|peak_allocated_mib=%.2f|"
        "peak_reserved_mib=%.2f|current_allocated_mib=%.2f|current_reserved_mib=%.2f",
        label,
        "current" if device is None else device,
        stats["peak_allocated_mib"],
        stats["peak_reserved_mib"],
        stats["current_allocated_mib"],
        stats["current_reserved_mib"],
    )
    return stats
