import unittest
from jax._src.distributed import State
import jax.numpy as jnp
import jax
from ._loop import _times_to_steps, _scan, _loop_with_checkpointing
from ..utils.dict_utils import dict_allclose

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
        self.assertTrue(jnp.allclose(segment_dt_fractions[0], jnp.array([0, 0.5, 0, 0.2])))

        times = jnp.array([1, 1.1, 1.5, 2.2, 2.4])
        dt = 0.5

        segment_steps, segment_dt_fractions = _times_to_steps(times, dt, 5)

        self.assertIsInstance(segment_steps, list)
        self.assertIsInstance(segment_dt_fractions, list)
        self.assertEqual(len(segment_steps), 1)
        self.assertEqual(len(segment_dt_fractions), 1)
        self.assertEqual(len(segment_steps[0]), 5)
        self.assertEqual(len(segment_dt_fractions[0]), 5)

        self.assertTrue(jnp.allclose(segment_steps[0], jnp.array([2, 2, 3, 4, 4])))
        self.assertTrue(jnp.allclose(segment_dt_fractions[0], jnp.array([0, 0.2, 0, 0.4, 0.8])))

    def test_times_to_steps_multiple_segments(self):
        times = jnp.array([0.1, 1, 2.5, 4.9, 5, 5.1, 6, 7.5, 9.9])
        dt = 1.0

        segment_steps, segment_dt_fractions = _times_to_steps(times, dt, 5)

        self.assertIsInstance(segment_steps, list)
        self.assertIsInstance(segment_dt_fractions, list)
        self.assertEqual(len(segment_steps), 2)
        self.assertEqual(len(segment_dt_fractions), 2)

        self.assertTrue(jnp.allclose(segment_steps[0], jnp.array([0, 1, 2, 4, 5])))
        self.assertTrue(jnp.allclose(segment_dt_fractions[0], jnp.array([0.1, 0, 0.5, 0.9, 0.0])))
        self.assertTrue(jnp.allclose(segment_steps[1], jnp.array([0, 1, 2, 4])))
        self.assertTrue(jnp.allclose(segment_dt_fractions[1], jnp.array([0.1, 0.0, 0.5, 0.9])))

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
        self.assertTrue(jnp.allclose(segment_dt_fractions[0], jnp.array([0, 0.2, 0])))
        self.assertTrue(jnp.allclose(segment_steps[1], jnp.array([1])))
        self.assertTrue(jnp.allclose(segment_dt_fractions[1], jnp.array([0.2])))

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
        final_state, hist = _scan(self.step_add_dt, key, state, None, dt, length=5)

        self.assertTrue(jnp.allclose(final_state[0], key))
        self.assertTrue(jnp.allclose(final_state[1]["A"], jnp.array([5])))
        self.assertTrue(jnp.allclose(hist["A"], jnp.array([0, 1, 2, 3, 4, 5])))

    def test_step_add_k(self):
        state = {
            "A": jnp.array([0.0, 1.0]),
        }
        non_state = {
            "k": jnp.array([0.1, 0.2]),
            "c": jnp.array([0.3, 0.4])
        }
        key = jax.random.key(0)
        final_state, hist = _scan(self.step_add_k, key, state, non_state, 1.0, length=5)

        self.assertTrue(jnp.allclose(final_state[0], key))
        self.assertTrue(jnp.allclose(final_state[1]["A"], jnp.array([0.5, 2.0])))
        self.assertTrue(jnp.allclose(hist["A"], jnp.array([[0.0, 1.0], [0.1, 1.2], [0.2, 1.4], [0.3, 1.6], [0.4, 1.8], [0.5, 2.0]])))

    def test_step_random_uniform(self):
        state = {
            "A": jnp.array(0.0),
        }
        key = jax.random.key(0)
        final_state, hist = _scan(self.step_random_uniform, key, state, None, 1.0, length=5)

        target_key = key
        target_state = state
        target_hist = [target_state["A"]]
        for _ in range(5):
            target_key, use_key = jax.random.split(target_key)
            target_state["A"] += jax.random.uniform(use_key, shape=target_state["A"].shape)
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

        interpolated_hist = _loop_with_checkpointing(self.add_one, times, key, state, None, dt, length)

        target_hist = {
            "A": jnp.array([1, 2.2, 3, 4.2]),
            "B": jnp.array([0, 0, 0, 0]),
        }

        self.assertTrue(dict_allclose(interpolated_hist, target_hist))

        times = jnp.array([0, 1, 2.2, 3, 4.2])
        interpolated_hist = _loop_with_checkpointing(self.add_one, times, key, state, None, dt, length)
        target_hist = {
            "A": jnp.array([0, 1, 2.2, 3, 4.2]),
            "B": jnp.array([0, 0, 0, 0, 0]),
        }

        self.assertTrue(dict_allclose(interpolated_hist, target_hist))

        interpolated_hist_no_checkpoint = _loop_with_checkpointing(self.add_one, times, key, state, None, dt, checkpoint_length=None)
        target_hist_no_checkpoint = {
            "A": jnp.array([0, 1, 2.2, 3, 4.2]),
            "B": jnp.array([0, 0, 0, 0, 0]),
        }
        self.assertTrue(dict_allclose(interpolated_hist_no_checkpoint, target_hist_no_checkpoint))

        interpolated_hist_change_length = _loop_with_checkpointing(self.add_one, times, key, state, None, dt, checkpoint_length=2)
        target_hist_change_length = {
            "A": jnp.array([0, 1, 2.2, 3, 4.2]),
            "B": jnp.array([0, 0, 0, 0, 0]),
        }
        self.assertTrue(dict_allclose(interpolated_hist_change_length, target_hist_change_length))

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

        interpolated_hist = _loop_with_checkpointing(self.multiply_by_k, times, key, state, non_state, dt, length)

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
        state = {
            "A": jnp.array(2.0),
            "B": jnp.array(2.0)
        }
        non_state = None
        
        target_hist = {
            "A": jnp.array([
                2.0000000e+00, 1.4963531e+00, -1.1551231e-03, -2.2048483e+00, \
                -3.6442435e+00, -4.7814059e+00, -6.2855392e+00, \
                -7.3914685e+00, -8.9495916e+00]),
            "B": jnp.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        }

        interpolated_hist = _loop_with_checkpointing(self.func, times, key, state, non_state, dt, length)

        self.assertTrue(dict_allclose(interpolated_hist, target_hist))