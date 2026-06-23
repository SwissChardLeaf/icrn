"""Test utilities (JAX device policy and GPU subprocess runners)."""

from icrn._testing.gpu_subprocess import (
    GPU_STRESS_MIN_VRAM_BYTES,
    gpu_available,
    gpu_memory_stats_supported,
    gpu_vram_at_least,
    gpu_vram_bytes,
    run_gpu_test,
)

__all__ = [
    "GPU_STRESS_MIN_VRAM_BYTES",
    "gpu_available",
    "gpu_memory_stats_supported",
    "gpu_vram_at_least",
    "gpu_vram_bytes",
    "run_gpu_test",
]
