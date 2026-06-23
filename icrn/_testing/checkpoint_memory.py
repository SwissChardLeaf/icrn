"""Shared helpers for checkpoint memory tests (CPU and GPU workers)."""

from __future__ import annotations

import gc
from typing import Any

import numpy as np


def matmul_step(key_state, non_state, dt):
    """One solver step: ``tanh(W @ x)`` on a 2D state."""
    import jax.numpy as jnp

    key, state = key_state
    x = state["A"]
    y = jnp.tanh(non_state["W"] @ x)
    return key, state | {"A": y}


def make_matmul_loss(checkpoint_length, n, steps, seed=0, device=None):
    """Build a scalar loss through ``_loop_with_checkpointing``."""
    import jax
    import jax.numpy as jnp

    from icrn._internal._loop import _loop_with_checkpointing

    key = jax.random.key(seed)
    state = {"A": jnp.zeros((n, n))}
    times = np.array([steps * 0.01])
    dt = 0.01

    W = jax.random.normal(key, (n, n)) * 0.01
    if device is not None:
        state = jax.device_put(state, device)
        W = jax.device_put(W, device)

    def loss(W):
        out = _loop_with_checkpointing(
            matmul_step,
            times,
            key,
            state,
            {"W": W},
            dt,
            checkpoint_length=checkpoint_length,
        )
        return out["A"][-1].sum()

    return loss, W


def read_peak_rss_bytes() -> int:
    """Return peak resident set size (bytes) for the current process."""
    with open("/proc/self/status") as status_file:
        for line in status_file:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("VmHWM not found in /proc/self/status")


def read_peak_gpu_bytes() -> int:
    """Return peak GPU memory use (bytes) for the current process."""
    import jax

    stats = jax.devices("gpu")[0].memory_stats()
    if not isinstance(stats, dict) or "peak_bytes_in_use" not in stats:
        raise RuntimeError("GPU peak memory stats are not available")
    return stats["peak_bytes_in_use"]


def backward_peak_rss_bytes(
    checkpoint_length, n, steps, seed=0, mp_context=None
) -> int:
    """Peak host RSS for a backward pass in an isolated CPU subprocess."""
    import multiprocessing as mp

    if mp_context is None:
        mp_context = mp.get_context("spawn")

    queue: mp.Queue = mp_context.Queue()
    process = mp_context.Process(
        target=_backward_worker_cpu,
        args=(checkpoint_length, n, steps, seed, queue),
    )
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(
            f"CPU backward worker failed with exit code {process.exitcode}"
        )
    return queue.get()


def backward_peak_gpu_bytes(
    checkpoint_length, n, steps, seed=0, mp_context=None
) -> int:
    """Peak GPU bytes for a backward pass in an isolated GPU subprocess."""
    import multiprocessing as mp

    if mp_context is None:
        mp_context = mp.get_context("spawn")

    queue: mp.Queue = mp_context.Queue()
    process = mp_context.Process(
        target=_backward_worker_gpu,
        args=(checkpoint_length, n, steps, seed, queue),
    )
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(
            f"GPU backward worker failed with exit code {process.exitcode}"
        )
    return queue.get()


def _backward_worker_cpu(checkpoint_length, n, steps, seed, queue: Any) -> None:
    import jax

    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        loss, W = make_matmul_loss(
            checkpoint_length, n, steps, seed=seed, device=cpu
        )
        grad = jax.grad(loss)(W)
        jax.block_until_ready(grad)
    gc.collect()
    queue.put(read_peak_rss_bytes())


def _backward_worker_gpu(checkpoint_length, n, steps, seed, queue: Any) -> None:
    from icrn._testing.gpu_subprocess import ensure_gpu_jax_env

    ensure_gpu_jax_env()
    import jax

    device = jax.devices("gpu")[0]
    loss, W = make_matmul_loss(
        checkpoint_length, n, steps, seed=seed, device=device
    )
    grad = jax.grad(loss)(W)
    jax.block_until_ready(grad)
    queue.put(read_peak_gpu_bytes())
