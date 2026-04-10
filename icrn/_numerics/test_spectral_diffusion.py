import unittest
from jax import numpy as jnp
from jax.numpy.fft import fftfreq

from icrn._numerics._spectral_diffusion import _compute_lap_op

class TestComputeLapOp(unittest.TestCase):
    def test_1D_compute_lap_op(self):
        result = _compute_lap_op((5,), (1,))
        manual_result = -(fftfreq(5, d=1) * 2 * jnp.pi)**2

        self.assertTrue(jnp.allclose(result, manual_result))

    def test_2D_compute_lap_op(self):
        result = _compute_lap_op((5, 5), (1, 1))

        kx = fftfreq(5, d=1) * 2 * jnp.pi
        ky = fftfreq(5, d=1) * 2 * jnp.pi
        manual_result = -(kx[None, :]**2 + ky[:, None]**2)
        self.assertTrue(jnp.allclose(result, manual_result))

    def test_3D_compute_lap_op(self):
        result = _compute_lap_op((5, 5, 5), (1, 1, 1))
        kx = fftfreq(5, d=1) * 2 * jnp.pi
        ky = fftfreq(5, d=1) * 2 * jnp.pi
        kz = fftfreq(5, d=1) * 2 * jnp.pi
        manual_result = -(kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2)
        self.assertTrue(jnp.allclose(result, manual_result))

def TestSpectralDiffuse(unittest.TestCase):
    pass