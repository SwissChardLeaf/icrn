import unittest

import jax
import jax.numpy as jnp

from ..utils.dict_utils import dict_allclose
from ._loop import (
    _loop_with_checkpointing,
    _scan_segment,
    _split_pre_computed_state_segments,
    _times_to_steps,
)


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
