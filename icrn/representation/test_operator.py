import unittest
from ._operator import AbstractOperator
import jax.numpy as jnp
import jax
from ..utils.dict_utils import dict_allclose

class TestExtendingAbstractOperator(unittest.TestCase):
    def test_subtracting(self):
        class TestOperator(AbstractOperator):
            def update_state(self, key, state, non_state, dt):
                state["A"] -= non_state["a"] * dt
                state["B"] -= non_state["b"] * dt
                return key, state

        op_no_mode = TestOperator(None)
        state = {"A": jnp.array(2.0), "B": jnp.array(2.0)}
        non_state = {"a": jnp.array(1.0), "b": jnp.array(2.0)}
        key = jax.random.key(0)
        
        key, state = op_no_mode.update_with_checks(key, state, non_state, 1)
        self.assertEqual(key, jax.random.key(0))
        self.assertTrue(dict_allclose(state, {"A": jnp.array(1.0), "B": jnp.array(0.0)}))

        key, state = op_no_mode.update_with_checks(key, state, non_state, 1)
        self.assertEqual(key, jax.random.key(0))
        self.assertTrue(dict_allclose(state, {"A": jnp.array(0.0), "B": jnp.array(-2.0)}))

        key, state = op_no_mode.update_with_checks(key, state, non_state, 1)
        self.assertEqual(key, jax.random.key(0))
        self.assertTrue(dict_allclose(state, {"A": jnp.array(-1.0), "B": jnp.array(-4.0)}))

        op_mode_relu = TestOperator("relu")
        state = {"A": jnp.array(2.0), "B": jnp.array(2.0)}
        non_state = {"a": jnp.array(1.0), "b": jnp.array(2.0)}
        key = jax.random.key(0)
        
        key, state = op_mode_relu.update_with_checks(key, state, non_state, 1)
        self.assertEqual(key, jax.random.key(0))
        self.assertTrue(dict_allclose(state, {"A": jnp.array(1.0), "B": jnp.array(0.0)}))

        key, state = op_mode_relu.update_with_checks(key, state, non_state, 1)
        self.assertEqual(key, jax.random.key(0))
        self.assertTrue(dict_allclose(state, {"A": jnp.array(0.0), "B": jnp.array(0.0)}))

        key, state = op_mode_relu.update_with_checks(key, state, non_state, 1)
        self.assertEqual(key, jax.random.key(0))
        self.assertTrue(dict_allclose(state, {"A": jnp.array(0.0), "B": jnp.array(0.0)}))

        op_mode_strict = TestOperator("strict")
        state = {"A": jnp.array(2.0), "B": jnp.array(2.0)}
        non_state = {"a": jnp.array(1.0), "b": jnp.array(2.0)}
        key = jax.random.key(0)
        
        key, state = op_mode_strict.update_with_checks(key, state, non_state, 1)
        self.assertEqual(key, jax.random.key(0))
        self.assertTrue(dict_allclose(state, {"A": jnp.array(1.0), "B": jnp.array(0.0)}))

        with self.assertRaises(ValueError):
            key, state = op_mode_strict.update_with_checks(key, state, non_state, 1)

class TestReactionsOperator(unittest.TestCase):
    def setup(self):
        A, B, C = many_species("A, B, C")
        alpha, beta = many_rate_constants("alpha, beta")
        i, j = many_index_symbols("i, j")

        self.rxns = [
            rxn(A + B, C, alpha),
            rxn(C, A + B, beta),
        ]
        

class TestFastReactionsOperator(unittest.TestCase):
    pass

class TestSpectralDiffusionOperator(unittest.TestCase):
    pass

class TestConvolutionalDiffusionOperator(unittest.TestCase):
    pass