import unittest
import jax.numpy as jnp
import jax
from ._loop import _times_to_steps, _inner_scan

class TestTimesToSteps(unittest.TestCase):
    def test_times_to_steps(self):
        times = jnp.array([0, 1, 1.5, 2, 2.2])
        dt = 1

        steps, dt_fractions = _times_to_steps(times, dt)

        self.assertTrue(jnp.allclose(steps, jnp.array([0, 1, 1, 2, 2])))
        self.assertTrue(jnp.allclose(dt_fractions, jnp.array([0, 0, 0.5, 0, 0.2])))

        times = jnp.array([1, 1.1, 1.5, 2.2, 2.4])
        dt = 0.5

        steps, dt_fractions = _times_to_steps(times, dt)

        self.assertTrue(jnp.allclose(steps, jnp.array([2, 2, 3, 4, 4])))
        self.assertTrue(jnp.allclose(dt_fractions, jnp.array([0, 0.2, 0, 0.4, 0.8])))

class TestScan(unittest.TestCase):
    def setUp(self):
        self.step_add_dt = lambda key, state, x, dt: ((key, state + dt), state + dt)
        self.step_add_k = lambda key, state, x, non_state: ((key, state + non_state["k"]), state + non_state["k"])

        def step_random_uniform(key, state, x):
            new_key, key = jax.random.split(key)
            new_state = state + jax.random.uniform(key, shape=state.shape)
            return (new_key, new_state), new_state

        self.step_random_uniform = step_random_uniform

        self.step_xs = lambda key, state, x: ((key, state + x), state + x)

    def test_step_add_dt(self):
        state = jnp.array(0)
        dt = 1
        key = jax.random.key(0)
        final_state, hist = _scan(self.step_add_dt, key, state, dt, length=5)

        self.assertTrue(jnp.allclose(final_state[0], key))
        self.assertTrue(jnp.allclose(final_state[1], jnp.array([5])))
        self.assertTrue(jnp.allclose(hist, jnp.array([1, 2, 3, 4, 5])))

    def test_step_add_k(self):
        state = jnp.array([0.0, 1.0])
        non_state = {
            "k": jnp.array([0.1, 0.2]),
            "c": jnp.array([0.3, 0.4])
        }
        key = jax.random.key(0)
        final_state, hist = _inner_scan(self.step_add_k, key, state, non_state, length=5)

        self.assertTrue(jnp.allclose(final_state[0], key))
        self.assertTrue(jnp.allclose(final_state[1], jnp.array([0.5, 2.0])))
        self.assertTrue(jnp.allclose(hist, jnp.array([[0.1, 1.2], [0.2, 1.4], [0.3, 1.6], [0.4, 1.8], [0.5, 2.0]])))

    def test_step_random_uniform(self):
        state = jnp.array([0.0, 1.0])
        key = jax.random.key(0)
        final_state, hist = _inner_scan(self.step_random_uniform, key, state, length=5)

        target_key = key
        target_state = state
        target_hist = []
        for _ in range(5):
            target_key, use_key = jax.random.split(target_key)
            target_state += jax.random.uniform(use_key, shape=target_state.shape)
            target_hist.append(target_state)

        target_hist = jnp.array(target_hist)

        self.assertTrue(jnp.allclose(final_state[0], target_key))
        self.assertTrue(jnp.allclose(final_state[1], target_state))
        self.assertTrue(jnp.allclose(hist, target_hist))

    def test_step_xs(self):
        state = jnp.array([0.0, 1.0])
        key = None
        xs = jnp.arange(5).astype(float)

        final_state, hist = _inner_scan(self.step_xs, key, state, xs=xs)
        self.assertTrue(final_state[0] == None)
        self.assertTrue(jnp.allclose(final_state[1], jnp.array([10.0, 11.0])))
        self.assertTrue(jnp.allclose(hist, jnp.array([[0.0, 1.0], [1.0, 2.0], [3.0, 4.0], [6.0, 7.0], [10.0, 11.0]])))

class TestScanCheckpointing(unittest.TestCase):
    pass

class TestScan(unittest.TestCase):
    def setUp(self):
        self.step_f = lambda state, non_state, time, dt, key: (state + dt, key)

    def test_only_state(self):
        state = jnp.array([0, 0])
        non_state = jnp.array([0, 0])
        times = jnp.array([0, 1])
        dt = 1
        key = jnp.array([0])
        result = _scan(self.step_f, state, non_state, dt, key, 1)

        self.assertEqual(
            jnp.allclose(result, jnp.array([0, 1]))
        )

class TestScanCheckpointing(unittest.TestCase):
    def setUp(self):
        self.step_f = lambda state, non_state, time, dt, key: (state + dt, key)

    def test_only_state(self):
        state = jnp.array([0, 0])
        non_state = jnp.array([0, 0])
        times = jnp.array([0, 1])
        dt = 1
        key = jnp.array([0])
        result = _scan_checkpointing(self.step_f, times, state, non_state, dt, key)

        self.assertEqual(
            jnp.allclose(result, jnp.array([0, 1]))
        )



    def test_only_non_state():
        pass

    def test_scan_by_segments_with_checkpointing_with_dict(self):
        pass

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
