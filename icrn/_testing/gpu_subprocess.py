"""Run test code on GPU in an isolated subprocess."""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Callable
from typing import Any

_MP_CONTEXT = mp.get_context("spawn")

_GPU_AVAILABLE: bool | None = None
_GPU_VRAM_BYTES: int | None = None

# Minimum VRAM for GPU checkpoint memory stress tests (opt-in via env var).
GPU_STRESS_MIN_VRAM_BYTES = 6 * 1024**3


def ensure_gpu_jax_env() -> None:
    """
    Configure the current process to use GPU JAX (call before importing jax).
    """
    os.environ["ICRN_TEST_JAX_DEVICE"] = "gpu"
    os.environ.pop("JAX_PLATFORMS", None)


def _probe_gpu_worker(queue: mp.Queue) -> None:
    ensure_gpu_jax_env()
    import jax

    queue.put(any(device.platform == "gpu" for device in jax.devices()))


def gpu_available() -> bool:
    """Return whether a GPU JAX device is available (probed in a subprocess)."""
    global _GPU_AVAILABLE
    if _GPU_AVAILABLE is not None:
        return _GPU_AVAILABLE

    queue: mp.Queue = _MP_CONTEXT.Queue()
    process = _MP_CONTEXT.Process(target=_probe_gpu_worker, args=(queue,))
    process.start()
    process.join()
    if process.exitcode != 0:
        _GPU_AVAILABLE = False
        return _GPU_AVAILABLE

    _GPU_AVAILABLE = bool(queue.get())
    return _GPU_AVAILABLE


def _gpu_test_worker(test_fn: Callable[[], Any], queue: mp.Queue) -> None:
    ensure_gpu_jax_env()
    try:
        test_fn()
    except BaseException as error:
        queue.put(error)
    else:
        queue.put(None)


def gpu_memory_stats_supported() -> bool:
    """True when GPU ``memory_stats()`` exposes ``peak_bytes_in_use``."""
    if not gpu_available():
        return False

    queue: mp.Queue = _MP_CONTEXT.Queue()
    process = _MP_CONTEXT.Process(
        target=_gpu_memory_stats_worker,
        args=(queue,),
    )
    process.start()
    process.join()
    if process.exitcode != 0:
        return False
    return bool(queue.get())


def _gpu_memory_stats_worker(queue: mp.Queue) -> None:
    ensure_gpu_jax_env()
    import jax

    stats = jax.devices("gpu")[0].memory_stats()
    supported = isinstance(stats, dict) and "peak_bytes_in_use" in stats
    queue.put(supported)


def _gpu_vram_worker(queue: mp.Queue) -> None:
    ensure_gpu_jax_env()
    import jax

    stats = jax.devices("gpu")[0].memory_stats()
    if isinstance(stats, dict) and "bytes_limit" in stats:
        queue.put(int(stats["bytes_limit"]))
    else:
        queue.put(None)


def gpu_vram_bytes() -> int | None:
    """Return total GPU VRAM (bytes) from ``memory_stats()['bytes_limit']``."""
    global _GPU_VRAM_BYTES
    if not gpu_available():
        return None
    if _GPU_VRAM_BYTES is not None:
        return _GPU_VRAM_BYTES

    queue: mp.Queue = _MP_CONTEXT.Queue()
    process = _MP_CONTEXT.Process(target=_gpu_vram_worker, args=(queue,))
    process.start()
    process.join()
    if process.exitcode != 0:
        return None

    vram = queue.get()
    if vram is None:
        return None

    _GPU_VRAM_BYTES = int(vram)
    return _GPU_VRAM_BYTES


def gpu_vram_at_least(min_bytes: int) -> bool:
    """True when the GPU reports at least ``min_bytes`` of VRAM."""
    vram = gpu_vram_bytes()
    return vram is not None and vram >= min_bytes


def run_gpu_test(test_fn: Callable[[], Any]) -> None:
    """Run ``test_fn`` in a fresh subprocess with GPU JAX enabled."""
    queue: mp.Queue = _MP_CONTEXT.Queue()
    process = _MP_CONTEXT.Process(
        target=_gpu_test_worker,
        args=(test_fn, queue),
    )
    process.start()
    process.join()
    if process.exitcode != 0 and queue.empty():
        raise RuntimeError(
            f"GPU test subprocess failed with exit code {process.exitcode}"
        )

    error = queue.get()
    if error is not None:
        raise error
