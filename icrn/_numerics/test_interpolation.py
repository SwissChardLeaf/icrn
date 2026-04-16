import unittest
import jax.numpy as jnp
from ._interpolation import _linear_interpolation, _hermitian_interpolation

class TestLinearInterpolation(unittest.TestCase):
    def test_exponential_decay(self):
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

    def test_multi_dimensional_decay(self):
        dt = 1.0
        y0 = jnp.array([1.0, 2.0, 3.0])
        a = 1.0
        ys = {
            "A" : jnp.array([y0 * jnp.exp(-a * i *dt) for i in range(10)])
        }

        times = jnp.array([0, 0.25, 1.2, 3.5, 8.1])
        steps = jnp.array([0, 0, 1, 3, 8])
        dt_fractions = jnp.array([0.0, 0.25, 0.2, 0.5, 0.1])

        result = _linear_interpolation(steps, dt_fractions, ys)
        target = jnp.array([
            y0,
            y0 + (y0 * jnp.exp(-a * 1) - y0) * 0.25,
            y0 * jnp.exp(-a * 1) + (y0 * jnp.exp(-a * 2) - y0 * jnp.exp(-a * 1)) * 0.2,
            y0 * jnp.exp(-a * 3) + (y0 * jnp.exp(-a * 4) - y0 * jnp.exp(-a * 3)) * 0.5,
            y0 * jnp.exp(-a * 8) + (y0 * jnp.exp(-a * 9) - y0 * jnp.exp(-a * 8)) * 0.1
        ])

        print(result["A"])

        self.assertTrue(jnp.allclose(result["A"], target))

class TestHermitianInterpolation(unittest.TestCase):
    def test_hermitian_interpolation(self):
        pass