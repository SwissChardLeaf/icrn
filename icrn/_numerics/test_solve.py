# import unittest

# op_no_mode = TestOperator()
#         state = {"A": jnp.array(2.0), "B": jnp.array(2.0)}
#         non_state = {"a": jnp.array(1.0), "b": jnp.array(2.0)}
#         key = jax.random.key(0)
        
#         key, state = op_no_mode.update_state(key, state, non_state, 1)
#         self.assertEqual(key, jax.random.key(0))
#         self.assertTrue(dict_allclose(state, {"A": jnp.array(1.0), "B": jnp.array(0.0)}))

#         key, state = op_no_mode.update_state(key, state, non_state, 1)
#         self.assertEqual(key, jax.random.key(0))
#         self.assertTrue(dict_allclose(state, {"A": jnp.array(0.0), "B": jnp.array(-2.0)}))

#         key, state = op_no_mode.update_state(key, state, non_state, 1)
#         self.assertEqual(key, jax.random.key(0))
#         self.assertTrue(dict_allclose(state, {"A": jnp.array(-1.0), "B": jnp.array(-4.0)}))

#         op_mode_relu = TestOperator("relu")
#         state = {"A": jnp.array(2.0), "B": jnp.array(2.0)}
#         non_state = {"a": jnp.array(1.0), "b": jnp.array(2.0)}
#         key = jax.random.key(0)
        
#         key, state = op_mode_relu.update_state(key, state, non_state, 1)
#         self.assertEqual(key, jax.random.key(0))
#         self.assertTrue(dict_allclose(state, {"A": jnp.array(1.0), "B": jnp.array(0.0)}))

#         key, state = op_mode_relu.update_state(key, state, non_state, 1)
#         self.assertEqual(key, jax.random.key(0))
#         self.assertTrue(dict_allclose(state, {"A": jnp.array(0.0), "B": jnp.array(0.0)}))

#         key, state = op_mode_relu.update_state(key, state, non_state, 1)
#         self.assertEqual(key, jax.random.key(0))
#         self.assertTrue(dict_allclose(state, {"A": jnp.array(0.0), "B": jnp.array(0.0)}))

#         op_mode_strict = TestOperator("strict")
#         state = {"A": jnp.array(2.0), "B": jnp.array(2.0)}
#         non_state = {"a": jnp.array(1.0), "b": jnp.array(2.0)}
#         key = jax.random.key(0)
        
#         key, state = op_mode_strict.update_with_checks(key, state, non_state, 1)
#         self.assertEqual(key, jax.random.key(0))
#         self.assertTrue(dict_allclose(state, {"A": jnp.array(1.0), "B": jnp.array(0.0)}))

#         with self.assertRaises(ValueError):
#             key, state = op_mode_strict.update_with_checks(key, state, non_state, 1)