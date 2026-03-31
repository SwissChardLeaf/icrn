import unittest
import jax.numpy as jnp
from ._scan import _scan_linear_interpolation, _scan_by_segments_with_checkpointing


class TestScanLinearInterpolation(unittest.TestCase):
    def test_scan_linear_interpolation_only_state(self):
        def step_f(state, non_state, time, dt, key):
            return state + dt

        state = jnp.array([0, 1])
        non_state = None
        times = jnp.array([0, 1, 1.5, 2, 2.2])
        dt = 1
        key = None
        result = _scan_linear_interpolation(step_f, times, state, non_state, dt, key)

        self.assertEqual(
            jnp.allclose(
                result, jnp.array([0, 1], [1, 2], [1.5, 2.5], [2, 3], [2.2, 3.2])
            )
        )

        # same problem but with dict

    def test_scan_linear_interpolation_only_non_state(self):
        def step_f(state, non_state, time, dt, key):
            return non_state + non_state

        state = jnp.array([0, 0])
        non_state = jnp.array([0, 0])
        times = jnp.array([0, 1])
        dt = 1
        key = jnp.array([0])
        result = _scan_linear_interpolation(step_f, times, state, non_state, dt, key)
