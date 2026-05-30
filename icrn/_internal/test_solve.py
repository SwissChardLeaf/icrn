import unittest

import jax
import jax.numpy as jnp
from jax.experimental import checkify

from ..operator import AbstractOperator
from ..utils.dict_utils import dict_allclose
from ._solve import _ops_to_f, _solve_with_f, _to_mod_op


class TestModOp(unittest.TestCase):
    def test_to_mod_op_deterministic(self):
        class TestOperator(AbstractOperator):
            def __init__(self, mode):
                self.mode = mode

            def update_state(self, state, non_state, dt):
                state["A"] -= 1
                return state

            def get_mode(self):
                return self.mode

            def get_is_stochastic(self):
                return False

        strict_op = TestOperator("strict")
        relu_op = TestOperator("relu")
        no_mode_op = TestOperator(None)

        strict_mod_op_f = _to_mod_op(strict_op)
        relu_mod_op_f = _to_mod_op(relu_op)
        no_mode_mod_op_f = _to_mod_op(no_mode_op)

        state = {"A": jnp.array(2.0), "B": jnp.array(2.0)}
        key = jax.random.key(0)

        checked = checkify.checkify(strict_mod_op_f)

        err, (key, state) = checked((key, state), None, 1)
        checkify.check_error(err)
        self.assertEqual(key, jax.random.key(0))
        self.assertTrue(
            dict_allclose(state, {"A": jnp.array(1.0), "B": jnp.array(2.0)})
        )

        err, (key, state) = checked((key, state), None, 1.1)
        checkify.check_error(err)
        self.assertEqual(key, jax.random.key(0))
        self.assertTrue(
            dict_allclose(state, {"A": jnp.array(0.0), "B": jnp.array(2.0)})
        )

        err, (key, state) = checked((key, state), None, 1.1)
        with self.assertRaises(Exception):
            checkify.check_error(err)

        state = {"A": jnp.array(2.0), "B": jnp.array(2.0)}
        key = jax.random.key(0)

        key, state = relu_mod_op_f((key, state), None, 1)
        self.assertEqual(key, jax.random.key(0))
        self.assertTrue(
            dict_allclose(state, {"A": jnp.array(1.0), "B": jnp.array(2.0)})
        )

        key, state = relu_mod_op_f((key, state), None, 1)
        self.assertEqual(key, jax.random.key(0))
        self.assertTrue(
            dict_allclose(state, {"A": jnp.array(0.0), "B": jnp.array(2.0)})
        )

        key, state = relu_mod_op_f((key, state), None, 1)
        self.assertEqual(key, jax.random.key(0))
        self.assertTrue(
            dict_allclose(state, {"A": jnp.array(0.0), "B": jnp.array(2.0)})
        )

        state = {"A": jnp.array(2.0), "B": jnp.array(2.0)}
        key = jax.random.key(0)

        key, state = no_mode_mod_op_f((key, state), None, 1)
        self.assertEqual(key, jax.random.key(0))
        self.assertTrue(
            dict_allclose(state, {"A": jnp.array(1.0), "B": jnp.array(2.0)})
        )

        key, state = no_mode_mod_op_f((key, state), None, 1)
        self.assertEqual(key, jax.random.key(0))
        self.assertTrue(
            dict_allclose(state, {"A": jnp.array(0.0), "B": jnp.array(2.0)})
        )

        key, state = no_mode_mod_op_f((key, state), None, 1)
        self.assertEqual(key, jax.random.key(0))
        self.assertTrue(
            dict_allclose(state, {"A": jnp.array(-1.0), "B": jnp.array(2.0)})
        )

    def test_to_mod_op_stochastic(self):
        class TestOperator(AbstractOperator):
            def __init__(self, mode):
                self.mode = mode

            def update_state(self, key_state, non_state, dt):
                key, state = key_state
                new_key, key = jax.random.split(key)
                state["A"] -= jax.random.uniform(key, shape=state["A"].shape)
                return new_key, state

            def get_mode(self):
                return self.mode

            def get_is_stochastic(self):
                return True

        strict_op = TestOperator("strict")
        relu_op = TestOperator("relu")
        no_mode_op = TestOperator(None)

        strict_mod_op_f = _to_mod_op(strict_op)
        relu_mod_op_f = _to_mod_op(relu_op)
        no_mode_mod_op_f = _to_mod_op(no_mode_op)

        key = jax.random.key(0)
        manual_vals = []
        val = jnp.array(2.0)
        while val > 0:
            new_key, key = jax.random.split(key)
            val -= jax.random.uniform(key, shape=val.shape)
            manual_vals.append((new_key, val))
            key = new_key

        state = {"A": jnp.array(2.0), "B": jnp.array(2.0)}
        key = jax.random.key(0)
        for man_key, man_val in manual_vals:
            key, state = no_mode_mod_op_f((key, state), None, 1)
            self.assertEqual(key, man_key)
            self.assertTrue(
                dict_allclose(state, {"A": man_val, "B": jnp.array(2.0)})
            )

        state = {"A": jnp.array(2.0), "B": jnp.array(2.0)}
        key = jax.random.key(0)
        for man_key, man_val in manual_vals:
            if man_val < 0:
                with self.assertRaises(ValueError):
                    key, state = strict_mod_op_f((key, state), None, 1)

                break
            else:
                key, state = strict_mod_op_f((key, state), None, 1)
                self.assertEqual(key, man_key)
                self.assertTrue(
                    dict_allclose(state, {"A": man_val, "B": jnp.array(2.0)})
                )

        state = {"A": jnp.array(2.0), "B": jnp.array(2.0)}
        key = jax.random.key(0)
        for man_key, man_val in manual_vals:
            key, state = relu_mod_op_f((key, state), None, 1)
            self.assertEqual(key, man_key)
            if man_val < 0:
                self.assertTrue(
                    dict_allclose(
                        state, {"A": jnp.array(0.0), "B": jnp.array(2.0)}
                    )
                )
            else:
                self.assertTrue(
                    dict_allclose(state, {"A": man_val, "B": jnp.array(2.0)})
                )


class TestSolveWithOps(unittest.TestCase):
    def setUp(self):
        class DeterministicOperator(AbstractOperator):
            def __init__(self, mode):
                self.mode = mode

            def update_state(self, state, non_state, dt):
                state["A"] -= 1.0
                return state

            def get_mode(self):
                return self.mode

            def get_is_stochastic(self):
                return False

        class StochasticOperator(AbstractOperator):
            def __init__(self, mode):
                self.mode = mode

            def update_state(self, key_state, non_state, dt):
                key, state = key_state
                new_key, key = jax.random.split(key)
                state["A"] -= jax.random.uniform(key, shape=state["A"].shape)
                return new_key, state

            def get_mode(self):
                return self.mode

            def get_is_stochastic(self):
                return True

        self.deterministic_op = DeterministicOperator
        self.stochastic_op = StochasticOperator

        def func(key_state, non_state, dt):
            key, state = key_state
            new_key, key = jax.random.split(key)
            state["A"] -= 1
            state["A"] -= jax.random.uniform(key, shape=state["A"].shape)
            return new_key, state

        self.func = func

    def test_ops_to_f(self):
        ops = [self.deterministic_op(None), self.stochastic_op(None)]

        state = {"A": jnp.array(2.0), "B": jnp.array(2.0)}
        key = jax.random.key(0)

        test_f = _ops_to_f(ops)

        out_key, out_state = test_f((key, state), None, 1)
        new_key, use_key = jax.random.split(key)
        self.assertEqual(out_key, new_key)
        self.assertTrue(
            dict_allclose(
                out_state,
                {
                    "A": 1.0
                    - jax.random.uniform(use_key, shape=state["A"].shape),
                    "B": jnp.array(2.0),
                },
            )
        )

    def test_solve_with_f(self):
        state = {"A": jnp.array(2.0), "B": jnp.array(2.0)}
        key = jax.random.key(0)
        times = jnp.array([0, 0.5, 1.9, 4, 5.2, 6, 7.1, 8, 9.1])
        dt = 1.0

        manual_dt_vals = [jnp.array(2.0)]
        for i in range(10):
            key, state = self.func((key, state), None, 1)
            manual_dt_vals.append(state["A"])

        target_interpolated_hist_vals = []
        for t in times:
            step = int(jnp.floor(t))
            dt_fraction = (t % dt) / dt

            target_val = (
                manual_dt_vals[step] * (1 - dt_fraction)
                + manual_dt_vals[step + 1] * dt_fraction
            )
            target_interpolated_hist_vals.append(target_val)

        target_interpolated_hist_vals = jnp.array(target_interpolated_hist_vals)

        state = {"A": jnp.array(2.0), "B": jnp.array(2.0)}
        key = jax.random.key(0)
        checkpoint_length = 4
        int_hist = _solve_with_f(
            self.func, state, None, dt, key, times, checkpoint_length
        )

        print(int_hist["A"])
        print(target_interpolated_hist_vals)

        self.assertTrue(
            jnp.allclose(int_hist["A"], target_interpolated_hist_vals)
        )

    def test_solve_with_ops(self):
        none_ops = [self.deterministic_op(None), self.stochastic_op(None)]
        relu_ops = [self.deterministic_op("relu"), self.stochastic_op("relu")]
        strict_ops = [
            self.deterministic_op("strict"),
            self.stochastic_op("strict"),
        ]

        none_f = _ops_to_f(none_ops)
        _ops_to_f(relu_ops)
        _ops_to_f(strict_ops)

        state = {"A": jnp.array(2.0), "B": jnp.array(2.0)}
        key = jax.random.key(0)

        out_key, out_state = none_f((key, state), None, 1)
        new_key, use_key = jax.random.split(key)
        self.assertEqual(out_key, new_key)
