import multiprocessing as mp
import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from ..utils.dict_utils import dict_allclose
from ._loop import (
    _loop_with_checkpointing,
    _scan_segment,
    _split_pre_computed_state_segments,
    _times_to_steps,
)

# Helpers for checkpoint memory tests (TestLoopCheckpointMemoryCPU/GPU). Each
# backward pass runs in a fresh subprocess so peak memory reflects that run
# alone, not prior allocations from comparing unchecked vs checkpointed modes.

GPU_AVAILABLE = any(device.platform == "gpu" for device in jax.devices())

_HAS_PROC_STATUS = os.path.isfile("/proc/self/status")
# Use spawn (not fork) so child processes start with a clean address space.
# Fork would inherit the parent's JAX allocations and skew memory comparisons.
_MP_CONTEXT = mp.get_context("spawn")


def _cpu_device():
    """Return the CPU device for checkpoint memory tests."""
    return jax.devices("cpu")[0]


def _gpu_memory_stats_supported():
    """True when GPU ``memory_stats()`` exposes ``peak_bytes_in_use``."""
    if not GPU_AVAILABLE:
        return False
    stats = jax.devices("gpu")[0].memory_stats()
    return isinstance(stats, dict) and "peak_bytes_in_use" in stats


def _matmul_step(key_state, non_state, dt):
    """One solver step: ``tanh(W @ x)`` on a 2D state.

    Matmul + nonlinearity makes reverse mode retain many scan intermediates
    when checkpointing is disabled, which is what the memory tests exercise.
    """
    key, state = key_state
    x = state["A"]
    y = jnp.tanh(non_state["W"] @ x)
    return key, state | {"A": y}


def _make_matmul_loss(checkpoint_length, n, steps, seed=0, device=None):
    """Build a scalar loss through ``_loop_with_checkpointing``.

    Runs ``steps`` fixed steps (``dt=0.01``, single output time at the end).
    ``checkpoint_length=None`` uses one uncheckpointed scan; an integer splits
    the trajectory and wraps each segment with ``jax.checkpoint``.

    Parameters
    ----------
    device : jax.Device, optional
        When set, places the initial state and weight matrix on this device
        (CPU tests use ``_cpu_device()``, GPU tests use a GPU device).

    Returns
    -------
    loss : callable
        Maps the weight matrix ``W`` to a scalar sum of the final state.
    W : jax.Array
        Initial weight matrix for ``jax.grad``.
    """
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
            _matmul_step,
            times,
            key,
            state,
            {"W": W},
            dt,
            checkpoint_length=checkpoint_length,
        )
        return out["A"][-1].sum()

    return loss, W


def _read_peak_rss_bytes():
    """Return peak resident set size (bytes) for the current process.

    Reads ``VmHWM`` from ``/proc/self/status`` (Linux). Only available when
    ``_HAS_PROC_STATUS`` is true.
    """
    with open("/proc/self/status") as status_file:
        for line in status_file:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("VmHWM not found in /proc/self/status")


def _backward_worker(checkpoint_length, n, steps, seed, queue):
    """Subprocess target: run one CPU backward pass and report peak RSS."""
    import gc

    cpu = _cpu_device()
    with jax.default_device(cpu):
        loss, W = _make_matmul_loss(
            checkpoint_length, n, steps, seed=seed, device=cpu
        )
        grad = jax.grad(loss)(W)
        jax.block_until_ready(grad)
    gc.collect()
    queue.put(_read_peak_rss_bytes())


def _read_peak_gpu_bytes():
    """Return peak GPU memory use (bytes) for the current process.

    Uses ``peak_bytes_in_use`` from the first GPU device's ``memory_stats()``.
    """
    stats = jax.devices("gpu")[0].memory_stats()
    if not isinstance(stats, dict) or "peak_bytes_in_use" not in stats:
        raise RuntimeError("GPU peak memory stats are not available")
    return stats["peak_bytes_in_use"]


def _backward_worker_gpu(checkpoint_length, n, steps, seed, queue):
    """Subprocess target: run one GPU backward pass and report peak device memory."""
    device = jax.devices("gpu")[0]
    loss, W = _make_matmul_loss(
        checkpoint_length, n, steps, seed=seed, device=device
    )
    grad = jax.grad(loss)(W)
    jax.block_until_ready(grad)
    queue.put(_read_peak_gpu_bytes())


def _backward_peak_gpu_bytes(checkpoint_length, n, steps, seed=0):
    """Peak GPU memory (bytes) for ``jax.grad(loss)(W)`` in an isolated subprocess."""
    queue = _MP_CONTEXT.Queue()
    process = _MP_CONTEXT.Process(
        target=_backward_worker_gpu,
        args=(checkpoint_length, n, steps, seed, queue),
    )
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(
            "GPU backward worker failed with exit code "
            f"{process.exitcode}"
        )
    return queue.get()


def _backward_peak_rss_bytes(checkpoint_length, n, steps, seed=0):
    """Peak RSS (bytes) for ``jax.grad(loss)(W)`` in an isolated subprocess.

    Spawns a new process via ``_MP_CONTEXT`` (spawn), runs the backward pass,
    and returns that process's high-water-mark RSS. Isolation keeps unchecked
    and checkpointed measurements comparable.
    """
    queue = _MP_CONTEXT.Queue()
    process = _MP_CONTEXT.Process(
        target=_backward_worker,
        args=(checkpoint_length, n, steps, seed, queue),
    )
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(
            "backward worker failed with exit code "
            f"{process.exitcode}"
        )
    return queue.get()


class TestTimesToSteps(unittest.TestCase):
    def test_times_to_steps_single_segment(self):
        times = jnp.array([1, 1.5, 2, 2.2])
        dt = 1

        segment_steps, segment_dt_fractions = _times_to_steps(times, dt, 5)

        self.assertIsInstance(segment_steps, list)
        self.assertIsInstance(segment_dt_fractions, list)
        self.assertEqual(len(segment_steps), 1)
        self.assertEqual(len(segment_dt_fractions), 1)
        self.assertEqual(len(segment_steps[0]), 4)
        self.assertEqual(len(segment_dt_fractions[0]), 4)

        self.assertTrue(jnp.allclose(segment_steps[0], jnp.array([1, 1, 2, 2])))
        self.assertTrue(
            jnp.allclose(segment_dt_fractions[0], jnp.array([0, 0.5, 0, 0.2]))
        )

        times = jnp.array([1, 1.1, 1.5, 2.2, 2.4])
        dt = 0.5

        segment_steps, segment_dt_fractions = _times_to_steps(times, dt, 5)

        self.assertIsInstance(segment_steps, list)
        self.assertIsInstance(segment_dt_fractions, list)
        self.assertEqual(len(segment_steps), 1)
        self.assertEqual(len(segment_dt_fractions), 1)
        self.assertEqual(len(segment_steps[0]), 5)
        self.assertEqual(len(segment_dt_fractions[0]), 5)

        self.assertTrue(
            jnp.allclose(segment_steps[0], jnp.array([2, 2, 3, 4, 4]))
        )
        self.assertTrue(
            jnp.allclose(
                segment_dt_fractions[0], jnp.array([0, 0.2, 0, 0.4, 0.8])
            )
        )

    def test_times_to_steps_multiple_segments(self):
        times = jnp.array([0.1, 1, 2.5, 4.9, 5, 5.1, 6, 7.5, 9.9])
        dt = 1.0

        segment_steps, segment_dt_fractions = _times_to_steps(times, dt, 5)

        self.assertIsInstance(segment_steps, list)
        self.assertIsInstance(segment_dt_fractions, list)
        self.assertEqual(len(segment_steps), 2)
        self.assertEqual(len(segment_dt_fractions), 2)

        self.assertTrue(
            jnp.allclose(segment_steps[0], jnp.array([0, 1, 2, 4, 5]))
        )
        self.assertTrue(
            jnp.allclose(
                segment_dt_fractions[0], jnp.array([0.1, 0, 0.5, 0.9, 0.0])
            )
        )
        self.assertTrue(jnp.allclose(segment_steps[1], jnp.array([0, 1, 2, 4])))
        self.assertTrue(
            jnp.allclose(
                segment_dt_fractions[1], jnp.array([0.1, 0.0, 0.5, 0.9])
            )
        )

    def test_times_to_steps_multiple_segments2(self):
        times = jnp.array([1, 2.2, 3, 4.2])
        length = 3
        dt = 1.0

        segment_steps, segment_dt_fractions = _times_to_steps(times, dt, length)

        self.assertIsInstance(segment_steps, list)
        self.assertIsInstance(segment_dt_fractions, list)
        self.assertEqual(len(segment_steps), 2)
        self.assertEqual(len(segment_dt_fractions), 2)

        self.assertTrue(jnp.allclose(segment_steps[0], jnp.array([1, 2, 3])))
        self.assertTrue(
            jnp.allclose(segment_dt_fractions[0], jnp.array([0, 0.2, 0]))
        )
        self.assertTrue(jnp.allclose(segment_steps[1], jnp.array([1])))
        self.assertTrue(jnp.allclose(segment_dt_fractions[1], jnp.array([0.2])))


class TestSplitPreComputedStateSegments(unittest.TestCase):
    def test_segments_have_checkpoint_length_plus_one(self):
        pre_computed_state = {"C": jnp.arange(6)}
        checkpoint_length = 3

        segments = _split_pre_computed_state_segments(
            pre_computed_state, checkpoint_length
        )

        self.assertIsInstance(segments, list)
        self.assertEqual(len(segments), 2)
        for segment in segments:
            self.assertEqual(segment["C"].shape[0], checkpoint_length + 1)
        # closed intervals: segment 0 holds points 0..3, segment 1 holds 3..6
        # (point 6 is past the input, so it is zero-padded)
        self.assertTrue(jnp.allclose(segments[0]["C"], jnp.array([0, 1, 2, 3])))
        self.assertTrue(jnp.allclose(segments[1]["C"], jnp.array([3, 4, 5, 0])))

    def test_adjacent_segments_share_boundary_states(self):
        pre_computed_state = {"C": jnp.arange(5)}

        segments = _split_pre_computed_state_segments(pre_computed_state, 2)

        self.assertEqual(len(segments), 3)
        self.assertTrue(jnp.allclose(segments[0]["C"], jnp.array([0, 1, 2])))
        self.assertTrue(jnp.allclose(segments[1]["C"], jnp.array([2, 3, 4])))
        # final segment zero-padded at the end
        self.assertTrue(jnp.allclose(segments[2]["C"], jnp.array([4, 0, 0])))

        # the end of each segment is the start of the next
        for i in range(len(segments) - 1):
            self.assertTrue(
                jnp.allclose(segments[i]["C"][-1], segments[i + 1]["C"][0])
            )

    def test_single_segment_when_length_exceeds_steps(self):
        pre_computed_state = {"C": jnp.arange(3)}

        segments = _split_pre_computed_state_segments(pre_computed_state, 5)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["C"].shape[0], 6)
        self.assertTrue(
            jnp.allclose(segments[0]["C"], jnp.array([0, 1, 2, 0, 0, 0]))
        )

    def test_multiple_keys_and_trailing_dims(self):
        pre_computed_state = {
            "C": jnp.arange(10).reshape(5, 2),
            "D": jnp.arange(5).astype(float),
        }

        segments = _split_pre_computed_state_segments(pre_computed_state, 2)

        self.assertEqual(len(segments), 3)
        for segment in segments:
            self.assertEqual(set(segment.keys()), {"C", "D"})
            self.assertEqual(segment["C"].shape, (3, 2))
            self.assertEqual(segment["D"].shape, (3,))

        self.assertTrue(
            jnp.allclose(segments[0]["C"], jnp.array([[0, 1], [2, 3], [4, 5]]))
        )
        self.assertTrue(
            jnp.allclose(segments[1]["C"], jnp.array([[4, 5], [6, 7], [8, 9]]))
        )
        self.assertTrue(
            jnp.allclose(segments[2]["C"], jnp.array([[8, 9], [0, 0], [0, 0]]))
        )
        # shared boundary on the leaf with trailing dims
        self.assertTrue(jnp.allclose(segments[0]["C"][-1], segments[1]["C"][0]))
        self.assertTrue(
            jnp.allclose(segments[2]["D"], jnp.array([4.0, 0.0, 0.0]))
        )


class TestScan(unittest.TestCase):
    def setUp(self):
        def step_add_dt(key_state, non_state, dt):
            key, state = key_state
            state["A"] += dt
            return key, state

        self.step_add_dt = step_add_dt

        def step_add_k(key_state, non_state, dt):
            key, state = key_state
            state["A"] += non_state["k"]
            return key, state

        self.step_add_k = step_add_k

        def step_random_uniform(key_state, non_state, dt):
            key, state = key_state
            new_key, key = jax.random.split(key)
            state["A"] += jax.random.uniform(key, shape=state["A"].shape)
            return new_key, state

        self.step_random_uniform = step_random_uniform

    def test_step_add_dt(self):
        state = {
            "A": jnp.array(0),
        }
        dt = 1
        key = jax.random.key(0)
        final_state, hist = _scan_segment(
            self.step_add_dt, key, state, None, dt, length=5
        )

        self.assertTrue(jnp.allclose(final_state[0], key))
        self.assertTrue(jnp.allclose(final_state[1]["A"], jnp.array([5])))
        self.assertTrue(jnp.allclose(hist["A"], jnp.array([0, 1, 2, 3, 4, 5])))

    def test_step_add_k(self):
        state = {
            "A": jnp.array([0.0, 1.0]),
        }
        non_state = {"k": jnp.array([0.1, 0.2]), "c": jnp.array([0.3, 0.4])}
        key = jax.random.key(0)
        final_state, hist = _scan_segment(
            self.step_add_k, key, state, non_state, 1.0, length=5
        )

        self.assertTrue(jnp.allclose(final_state[0], key))
        self.assertTrue(
            jnp.allclose(final_state[1]["A"], jnp.array([0.5, 2.0]))
        )
        self.assertTrue(
            jnp.allclose(
                hist["A"],
                jnp.array(
                    [
                        [0.0, 1.0],
                        [0.1, 1.2],
                        [0.2, 1.4],
                        [0.3, 1.6],
                        [0.4, 1.8],
                        [0.5, 2.0],
                    ]
                ),
            )
        )

    def test_step_random_uniform(self):
        state = {
            "A": jnp.array(0.0),
        }
        key = jax.random.key(0)
        final_state, hist = _scan_segment(
            self.step_random_uniform, key, state, None, 1.0, length=5
        )

        target_key = key
        target_state = state
        target_hist = [target_state["A"]]
        for _ in range(5):
            target_key, use_key = jax.random.split(target_key)
            target_state["A"] += jax.random.uniform(
                use_key, shape=target_state["A"].shape
            )
            target_hist.append(target_state["A"])

        target_hist = jnp.array(target_hist)

        self.assertTrue(jnp.allclose(final_state[0], target_key))
        self.assertTrue(jnp.allclose(final_state[1]["A"], target_state["A"]))
        self.assertTrue(jnp.allclose(hist["A"], target_hist))


class TestLoopWithCheckpointing(unittest.TestCase):
    def setUp(self):
        def add_one(key_state, non_state, dt):
            key, state = key_state
            state["A"] += 1
            return key, state

        self.add_one = add_one

        def multiply_by_k(key_state, non_state, dt):
            key, state = key_state
            state["A"] += dt * non_state["k"] * state["A"]
            return key, state

        self.multiply_by_k = multiply_by_k

        def func(key_state, non_state, dt):
            key, state = key_state
            new_key, key = jax.random.split(key)
            state["A"] -= 1
            state["A"] -= jax.random.uniform(key, shape=state["A"].shape)
            return new_key, state

        self.func = func

    def test_add_one(self):
        times = jnp.array([1, 2.2, 3, 4.2])
        length = 3
        dt = 1.0
        key = jax.random.key(0)
        state = {
            "A": jnp.array(0),
            "B": jnp.array(0),
        }

        interpolated_hist = _loop_with_checkpointing(
            self.add_one, times, key, state, None, dt, length
        )

        target_hist = {
            "A": jnp.array([1, 2.2, 3, 4.2]),
            "B": jnp.array([0, 0, 0, 0]),
        }

        self.assertTrue(dict_allclose(interpolated_hist, target_hist))

        times = jnp.array([0, 1, 2.2, 3, 4.2])
        interpolated_hist = _loop_with_checkpointing(
            self.add_one, times, key, state, None, dt, length
        )
        target_hist = {
            "A": jnp.array([0, 1, 2.2, 3, 4.2]),
            "B": jnp.array([0, 0, 0, 0, 0]),
        }

        self.assertTrue(dict_allclose(interpolated_hist, target_hist))

        interpolated_hist_no_checkpoint = _loop_with_checkpointing(
            self.add_one, times, key, state, None, dt, checkpoint_length=None
        )
        target_hist_no_checkpoint = {
            "A": jnp.array([0, 1, 2.2, 3, 4.2]),
            "B": jnp.array([0, 0, 0, 0, 0]),
        }
        self.assertTrue(
            dict_allclose(
                interpolated_hist_no_checkpoint, target_hist_no_checkpoint
            )
        )

        interpolated_hist_change_length = _loop_with_checkpointing(
            self.add_one, times, key, state, None, dt, checkpoint_length=2
        )
        target_hist_change_length = {
            "A": jnp.array([0, 1, 2.2, 3, 4.2]),
            "B": jnp.array([0, 0, 0, 0, 0]),
        }
        self.assertTrue(
            dict_allclose(
                interpolated_hist_change_length, target_hist_change_length
            )
        )

    def test_pre_computed_state_wrong_length_raises(self):
        times = jnp.array([0.0, 1.0])
        dt = 0.1
        key = jax.random.key(0)
        state = {"A": jnp.array(0.0), "C": jnp.array(0.0)}
        # max_step = ceil(1.0 / 0.1) = 10, so the required leading length is 11
        bad_pre_computed_state = {"C": jnp.zeros(5)}

        with self.assertRaises(ValueError):
            _loop_with_checkpointing(
                self.add_one,
                times,
                key,
                state,
                None,
                dt,
                checkpoint_length=4,
                pre_computed_state=bad_pre_computed_state,
            )

    def test_pre_computed_state_correct_length_runs(self):
        times = jnp.array([0.0, 1.0])
        dt = 0.1
        key = jax.random.key(0)
        state = {"A": jnp.array(0.0), "C": jnp.array(0.0)}
        good_pre_computed_state = {"C": jnp.zeros(11)}

        # should not raise
        _loop_with_checkpointing(
            self.add_one,
            times,
            key,
            state,
            None,
            dt,
            checkpoint_length=4,
            pre_computed_state=good_pre_computed_state,
        )

    def test_multiply_by_k(self):
        times = jnp.array([1, 2, 2.2, 3, 4, 4.4, 4.5])
        length = 8
        dt = 0.5
        key = jax.random.key(0)
        state = {
            "A": jnp.array(1.0),
            "B": jnp.array(0),
        }
        non_state = {
            "k": jnp.array(2.0),
        }

        interpolated_hist = _loop_with_checkpointing(
            self.multiply_by_k, times, key, state, non_state, dt, length
        )

        target_hist = {
            "A": jnp.array([4.0, 16, 22.4, 64, 256, 460.8, 512]),
            "B": jnp.array([0, 0, 0, 0, 0, 0, 0]),
        }

        print(_times_to_steps(times, dt, length))

        self.assertTrue(dict_allclose(interpolated_hist, target_hist))

    def test_func(self):
        times = jnp.array([0, 0.5, 1.9, 4, 5.2, 6, 7.1, 8, 9.1])
        length = 4
        dt = 1.0
        key = jax.random.key(0)
        state = {"A": jnp.array(2.0), "B": jnp.array(2.0)}
        non_state = None

        target_hist = {
            "A": jnp.array(
                [
                    2.0000000e00,
                    1.4963531e00,
                    -1.1551231e-03,
                    -2.2048483e00,
                    -3.6442435e00,
                    -4.7814059e00,
                    -6.2855392e00,
                    -7.3914685e00,
                    -8.9495916e00,
                ]
            ),
            "B": jnp.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
        }

        interpolated_hist = _loop_with_checkpointing(
            self.func, times, key, state, non_state, dt, length
        )

        self.assertTrue(dict_allclose(interpolated_hist, target_hist))


class TestLoopCheckpointMemoryCPU(unittest.TestCase):
    """CPU tests: ``checkpoint_length`` lowers backward-pass host RSS.

    Arrays are placed on the CPU explicitly (via ``jax.default_device``) so
    these tests still measure host RSS when a GPU is available. Peak RSS
    comparisons run each backward pass in a fresh subprocess (see module
    helpers above) because ``VmHWM`` is cumulative for the lifetime of a
    process.
    """

    def test_checkpoint_length_grad_matches_no_checkpoint(self):
        """Checkpointing must not change reverse-mode gradients on CPU."""
        cpu = _cpu_device()
        n = 128
        steps = 200
        with jax.default_device(cpu):
            loss_none, W = _make_matmul_loss(None, n, steps, device=cpu)
            loss_ckpt, _ = _make_matmul_loss(8, n, steps, device=cpu)

            grad_none = jax.grad(loss_none)(W)
            grad_ckpt = jax.grad(loss_ckpt)(W)
            jax.block_until_ready((grad_none, grad_ckpt))

        self.assertEqual(grad_none.device, cpu)
        self.assertEqual(grad_ckpt.device, cpu)
        self.assertTrue(
            jnp.allclose(grad_none, grad_ckpt, rtol=1e-4, atol=1e-4)
        )

    @unittest.skipUnless(
        _HAS_PROC_STATUS,
        "peak RSS measurement requires /proc/self/status",
    )
    def test_checkpoint_length_reduces_backward_peak_memory(self):
        """Segmented ``jax.checkpoint`` uses less peak RSS than no checkpoint."""
        n = 256
        steps = 350
        checkpoint_length = 8

        peak_none = _backward_peak_rss_bytes(None, n, steps)
        peak_ckpt = _backward_peak_rss_bytes(
            checkpoint_length, n, steps
        )

        self.assertLess(peak_ckpt, peak_none)
        self.assertLess(peak_ckpt, 0.7 * peak_none)

    @unittest.skipUnless(
        os.environ.get("ICRN_MEMORY_STRESS_TEST"),
        "set ICRN_MEMORY_STRESS_TEST=1 to run",
    )
    @unittest.skipUnless(
        _HAS_PROC_STATUS,
        "peak RSS measurement requires /proc/self/status",
    )
    def test_checkpoint_length_large_problem_fits_with_checkpoints(self):
        """Large problem: checkpointed backward uses much less RSS (manual/CI opt-in)."""
        n = 512
        steps = 500
        checkpoint_length = 8

        peak_none = _backward_peak_rss_bytes(None, n, steps)
        peak_ckpt = _backward_peak_rss_bytes(
            checkpoint_length, n, steps
        )

        self.assertLess(peak_ckpt, peak_none)
        self.assertLess(peak_ckpt, 0.5 * peak_none)
        self.assertGreater(peak_none - peak_ckpt, 500 * 1024 * 1024)


@unittest.skipUnless(GPU_AVAILABLE, "no GPU available")
class TestLoopCheckpointMemoryGPU(unittest.TestCase):
    """GPU tests: ``checkpoint_length`` lowers backward-pass device memory.

    Mirrors ``TestLoopCheckpointMemoryCPU`` but places arrays on the GPU and
    reads ``peak_bytes_in_use`` from ``device.memory_stats()`` in isolated
    subprocesses.
    """

    def test_checkpoint_length_grad_matches_no_checkpoint(self):
        """Checkpointing must not change reverse-mode gradients on GPU."""
        device = jax.devices("gpu")[0]
        n = 128
        steps = 200
        loss_none, W = _make_matmul_loss(None, n, steps, device=device)
        loss_ckpt, _ = _make_matmul_loss(8, n, steps, device=device)

        grad_none = jax.grad(loss_none)(W)
        grad_ckpt = jax.grad(loss_ckpt)(W)
        jax.block_until_ready((grad_none, grad_ckpt))

        self.assertEqual(grad_none.device, device)
        self.assertEqual(grad_ckpt.device, device)
        self.assertTrue(
            jnp.allclose(grad_none, grad_ckpt, rtol=1e-4, atol=1e-4)
        )

    @unittest.skipUnless(
        _gpu_memory_stats_supported(),
        "GPU device memory_stats peak_bytes_in_use is not available",
    )
    def test_checkpoint_length_reduces_backward_peak_memory(self):
        """Segmented ``jax.checkpoint`` uses less peak GPU memory."""
        n = 256
        steps = 350
        checkpoint_length = 8

        peak_none = _backward_peak_gpu_bytes(None, n, steps)
        peak_ckpt = _backward_peak_gpu_bytes(
            checkpoint_length, n, steps
        )

        self.assertLess(peak_ckpt, peak_none)
        self.assertLess(peak_ckpt, 0.7 * peak_none)

    @unittest.skipUnless(
        os.environ.get("ICRN_MEMORY_STRESS_TEST"),
        "set ICRN_MEMORY_STRESS_TEST=1 to run",
    )
    @unittest.skipUnless(
        _gpu_memory_stats_supported(),
        "GPU device memory_stats peak_bytes_in_use is not available",
    )
    def test_checkpoint_length_large_problem_fits_with_checkpoints(self):
        """Large GPU problem: checkpointed backward uses much less device memory."""
        n = 512
        steps = 500
        checkpoint_length = 8

        peak_none = _backward_peak_gpu_bytes(None, n, steps)
        peak_ckpt = _backward_peak_gpu_bytes(
            checkpoint_length, n, steps
        )

        self.assertLess(peak_ckpt, peak_none)
        self.assertLess(peak_ckpt, 0.5 * peak_none)
        self.assertGreater(peak_none - peak_ckpt, 500 * 1024 * 1024)
