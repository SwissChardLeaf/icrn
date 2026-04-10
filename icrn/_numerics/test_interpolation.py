import unittest
import jax.numpy as jnp
from ._interpolation import _linear_interpolation, _hermitian_interpolation

class TestLinearInterpolation(unittest.TestCase):
    def test_linear_interpolation(self):
        # state is y = y0 * e^(-at)
        dt = 0.5
        y0 = 2.0
        a = 2.0
        ys = {
            "A" : y0 * jnp.exp(-a * jnp.arange(20) *dt)
        }

        times = jnp.array([0, 0.25, 1.2, 3.5, 8.1])
        steps = jnp.array([0, 0, 2, 7, 16])
        dt_fractions = jnp.array([0.0, 0.5, 0.4, 0.0, 0.2])

        result = _linear_interpolation(steps, dt_fractions, ys)
        target = jnp.array([
            y0,
            y0 + (y0 * jnp.exp(-a * 0.5) - y0) * 0.5,
            y0 * jnp.exp(-a * 1) + (y0 * jnp.exp(-a * 1.5) - y0 * jnp.exp(-a * 1)) * 0.4,
            y0 * jnp.exp(-a * 3.5), 
            y0 * jnp.exp(-a * 8.0) + (y0 * jnp.exp(-a * 8.5) - y0 * jnp.exp(-a * 8.0)) * 0.2
        ])

        self.assertTrue(jnp.allclose(result["A"], target))

class TestHermitianInterpolation(unittest.TestCase):
    def test_hermitian_interpolation(self):
        pass