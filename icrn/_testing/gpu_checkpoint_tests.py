"""
GPU checkpoint memory test bodies (run in subprocess via ``run_gpu_test``).
"""

from __future__ import annotations


def checkpoint_grad_matches_no_checkpoint() -> None:
    import jax
    import jax.numpy as jnp

    from icrn._testing.checkpoint_memory import make_matmul_loss

    device = jax.devices("gpu")[0]
    n = 128
    steps = 200
    loss_none, W = make_matmul_loss(None, n, steps, device=device)
    loss_ckpt, _ = make_matmul_loss(8, n, steps, device=device)

    grad_none = jax.grad(loss_none)(W)
    grad_ckpt = jax.grad(loss_ckpt)(W)
    jax.block_until_ready((grad_none, grad_ckpt))

    assert grad_none.device == device
    assert grad_ckpt.device == device
    assert jnp.allclose(grad_none, grad_ckpt, rtol=1e-4, atol=1e-4)


def checkpoint_reduces_backward_peak_memory() -> None:
    from icrn._testing.checkpoint_memory import backward_peak_gpu_bytes

    n = 256
    steps = 350
    checkpoint_length = 8

    peak_none = backward_peak_gpu_bytes(None, n, steps)
    peak_ckpt = backward_peak_gpu_bytes(checkpoint_length, n, steps)

    assert peak_ckpt < peak_none
    assert peak_ckpt < 0.7 * peak_none


def checkpoint_large_problem_fits_with_checkpoints() -> None:
    from icrn._testing.checkpoint_memory import backward_peak_gpu_bytes

    # Smaller than the CPU stress test (512/500): GPU VRAM is tighter and the
    # uncheckpointed backward pass must complete on consumer cards (e.g. 8 GiB).
    n = 384
    steps = 400
    checkpoint_length = 8
    min_savings_bytes = 250 * 1024 * 1024

    peak_none = backward_peak_gpu_bytes(None, n, steps)
    peak_ckpt = backward_peak_gpu_bytes(checkpoint_length, n, steps)

    assert peak_ckpt < peak_none
    assert peak_ckpt < 0.5 * peak_none
    assert peak_none - peak_ckpt > min_savings_bytes
