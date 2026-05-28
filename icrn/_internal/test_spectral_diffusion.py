import unittest
from jax import numpy as jnp
from jax.numpy.fft import fftfreq

from ..symbols import many_species
from ._spectral_diffusion import (
    _compute_lap_op,
    _spectral_species_diffuse,
    _spectral_diffuse,
)
from ..utils.dict_utils import dict_allclose


class TestComputeLapOp(unittest.TestCase):
    def test_1D_compute_lap_op(self):
        computed_result = _compute_lap_op((5,), (1,))
        target_result = -((fftfreq(5, d=1) * 2 * jnp.pi) ** 2)

        self.assertTrue(jnp.allclose(computed_result, target_result))

        computed_result = _compute_lap_op((4, 3), (1, 1))
        target_result = jnp.array(
            [
                [-0.0, -4.3864913, -4.3864913],
                [-2.4674013, -6.8538923, -6.8538923],
                [-9.869605, -14.256096, -14.256096],
                [-2.4674013, -6.8538923, -6.8538923],
            ]
        )

        self.assertTrue(jnp.allclose(computed_result, target_result))

    def test_2D_compute_lap_op(self):
        computed_result = _compute_lap_op((5, 5), (1, 1))

        kx = fftfreq(5, d=1) * 2 * jnp.pi
        ky = fftfreq(5, d=1) * 2 * jnp.pi
        target_result = -(kx[None, :] ** 2 + ky[:, None] ** 2)
        self.assertTrue(jnp.allclose(computed_result, target_result))

        computed_result = _compute_lap_op((2, 3), (0.5, 2.0))
        target_result = jnp.array(
            [
                [-0.0, -1.0966228, -1.0966228],
                [-39.47842, -40.575043, -40.575043],
            ]
        )
        self.assertTrue(jnp.allclose(computed_result, target_result))

    def test_3D_compute_lap_op(self):
        result = _compute_lap_op((5, 5, 5), (1, 1, 1))
        kx = fftfreq(5, d=1) * 2 * jnp.pi
        ky = fftfreq(5, d=1) * 2 * jnp.pi
        kz = fftfreq(5, d=1) * 2 * jnp.pi
        manual_result = -(
            kx[:, None, None] ** 2
            + ky[None, :, None] ** 2
            + kz[None, None, :] ** 2
        )
        self.assertTrue(jnp.allclose(result, manual_result))

        computed_result = _compute_lap_op((1, 4), (1, 1))
        target_result = jnp.array([[-0.0, -2.4674013, -9.869605, -2.4674013]])
        self.assertTrue(jnp.allclose(computed_result, target_result))

    def test_compute_lap_op_different_dspace(self):
        computed_result = _compute_lap_op((5,), (1.5,))
        kx = fftfreq(5, d=1.5) * 2 * jnp.pi
        manual_result = -(kx**2)
        self.assertTrue(jnp.allclose(computed_result, manual_result))

        computed_result = _compute_lap_op((5, 5, 5), (1, 2, 3))
        kx = fftfreq(5, d=1) * 2 * jnp.pi
        ky = fftfreq(5, d=2) * 2 * jnp.pi
        kz = fftfreq(5, d=3) * 2 * jnp.pi
        manual_result = -(
            kx[:, None, None] ** 2
            + ky[None, :, None] ** 2
            + kz[None, None, :] ** 2
        )
        self.assertTrue(jnp.allclose(computed_result, manual_result))


class TestSpectralSpeciesDiffuse(unittest.TestCase):
    def test_1D_spectral_species_diffuse_analytical(self):
        n_points = 100
        lap_op = _compute_lap_op((n_points,), (2 * jnp.pi / n_points,))
        time = 1.0
        kd = jnp.array(1.0)
        target = 1.0 + jnp.cos(
            jnp.linspace(0, 2 * jnp.pi, n_points, endpoint=False)
        ) * jnp.exp(-kd * time)

        max_errs = []

        for dt in [1e-0, 1e-1, 1e-2, 1e-3]:
            state = 1.0 + jnp.cos(
                jnp.linspace(0, 2 * jnp.pi, n_points, endpoint=False)
            )

            for _ in range(jnp.ceil(time / dt).astype(int)):
                state = _spectral_species_diffuse(state, kd, lap_op, dt)

            max_errs.append(jnp.max(jnp.abs(state - target)))

        self.assertTrue(jnp.all(jnp.diff(jnp.array(max_errs)) < 0))
        self.assertTrue(max_errs[-1] < 1e-3)

    def test_1D_and_2D_agreement(self):
        state_1D = jnp.linspace(0, 2 * jnp.pi, 10, endpoint=False)
        state_2D = state_1D[None, :]
        lap_op_1D = _compute_lap_op((10,), (1,))
        lap_op_2D = _compute_lap_op((1, 10), (1, 1))

        kd = jnp.array(1.0)
        dt = 1.0

        result_1D = _spectral_species_diffuse(state_1D, kd, lap_op_1D, dt)
        result_2D = _spectral_species_diffuse(state_2D, kd, lap_op_2D, dt)

        self.assertTrue(jnp.allclose(result_1D, result_2D[0]))

    def test_2D_spectral_species_diffuse(self):
        initial_state = jnp.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        target_state = jnp.array(
            [
                [0.03760443, 0.03843227, 0.03896552, 0.03843227, 0.03760443],
                [0.03843226, 0.0406484, 0.04309084, 0.0406484, 0.03843226],
                [0.03896552, 0.04309085, 0.05130513, 0.04309085, 0.03896552],
                [0.03843226, 0.0406484, 0.04309084, 0.0406484, 0.03843226],
                [0.03760443, 0.03843226, 0.03896552, 0.03843226, 0.03760443],
            ]
        )

        lap_op = _compute_lap_op((5, 5), (1, 1))
        kd = jnp.array(10.0)
        dt = 2.0

        computed_state = _spectral_species_diffuse(
            initial_state, kd, lap_op, dt
        )
        self.assertTrue(jnp.allclose(computed_state, target_state))

    def test_2D_spectral_species_diffuse_analytical(self):
        n_points = 100
        lap_op = _compute_lap_op(
            (n_points, n_points), (2 * jnp.pi / n_points, 2 * jnp.pi / n_points)
        )
        time = 1.0
        kd = jnp.array(1.0)

        x = jnp.linspace(0, 2 * jnp.pi, n_points, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, n_points, endpoint=False)
        X, Y = jnp.meshgrid(x, y)

        target = (
            1.0
            + jnp.cos(X) * jnp.exp(-kd * time)
            + jnp.cos(Y) * jnp.exp(-kd * time)
        )

        max_errs = []

        for dt in [1e-0, 1e-1, 1e-2]:
            state = 1.0 + jnp.cos(X) + jnp.cos(Y)

            for _ in range(jnp.ceil(time / dt).astype(int)):
                state = _spectral_species_diffuse(state, kd, lap_op, dt)

            max_errs.append(jnp.max(jnp.abs(state - target)))

        self.assertTrue(jnp.all(jnp.diff(jnp.array(max_errs)) < 0))
        self.assertTrue(max_errs[-1] < 1e-2)

    def test_2D_spectral_species_diffuse_multidimensional_species(self):
        initial_state1 = jnp.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        initial_state2 = jnp.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        initial_state = jnp.zeros((5, 5, 3, 2))

        initial_state = initial_state.at[..., 0, 0].set(initial_state1)
        initial_state = initial_state.at[..., 0, 1].set(initial_state2)
        initial_state = initial_state.at[..., 1, 0].set(initial_state1)
        initial_state = initial_state.at[..., 1, 1].set(initial_state2)
        initial_state = initial_state.at[..., 2, 0].set(initial_state1)
        initial_state = initial_state.at[..., 2, 1].set(initial_state2)

        self.assertEqual(initial_state.shape, (5, 5, 3, 2))

        lap_op = _compute_lap_op((5, 5), (1, 1))
        kd = jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        dt = 1.0

        target_state = jnp.zeros((5, 5, 3, 2))
        target_state = target_state.at[..., 0, 0].set(
            _spectral_species_diffuse(
                initial_state1, jnp.array(1.0), lap_op, dt
            )
        )
        target_state = target_state.at[..., 0, 1].set(
            _spectral_species_diffuse(
                initial_state2, jnp.array(1.0), lap_op, dt
            )
        )
        target_state = target_state.at[..., 1, 0].set(
            _spectral_species_diffuse(
                initial_state1, jnp.array(2.0), lap_op, dt
            )
        )
        target_state = target_state.at[..., 1, 1].set(
            _spectral_species_diffuse(
                initial_state2, jnp.array(2.0), lap_op, dt
            )
        )
        target_state = target_state.at[..., 2, 0].set(
            _spectral_species_diffuse(
                initial_state1, jnp.array(3.0), lap_op, dt
            )
        )
        target_state = target_state.at[..., 2, 1].set(
            _spectral_species_diffuse(
                initial_state2, jnp.array(3.0), lap_op, dt
            )
        )

        computed_state = _spectral_species_diffuse(
            initial_state, kd, lap_op, dt
        )
        self.assertTrue(jnp.allclose(computed_state, target_state))

    def test_3D_spectral_species_diffusion(self):
        pass


class TestSpectralDiffuse(unittest.TestCase):
    def test_spectral_diffuse(self):
        A, B, C = many_species("A, B, C")

        def make_initial_state(a):
            res = jnp.zeros((5, 5))
            res = res.at[a // 5, a % 5].set(1.0)
            return res

        A_initial_state = make_initial_state(0)

        B_initial_state = jnp.zeros((5, 5, 2))
        B_initial_state = B_initial_state.at[..., 0].set(make_initial_state(0))
        B_initial_state = B_initial_state.at[..., 1].set(make_initial_state(1))

        C_initial_state = jnp.zeros((5, 5, 3, 2))
        C_initial_state = C_initial_state.at[..., 0, 0].set(
            make_initial_state(0)
        )
        C_initial_state = C_initial_state.at[..., 0, 1].set(
            make_initial_state(1)
        )
        C_initial_state = C_initial_state.at[..., 1, 0].set(
            make_initial_state(2)
        )
        C_initial_state = C_initial_state.at[..., 1, 1].set(
            make_initial_state(3)
        )
        C_initial_state = C_initial_state.at[..., 2, 0].set(
            make_initial_state(4)
        )
        C_initial_state = C_initial_state.at[..., 2, 1].set(
            make_initial_state(5)
        )

        initial_state = {
            A: A_initial_state,
            B: B_initial_state,
            C: C_initial_state,
        }

        diffusion_constant_vals = {
            A: jnp.array(1.0),
            B: jnp.array([1.0, 2.0]),
            C: jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        }

        lap_op = _compute_lap_op((5, 5), (1, 1))
        dt = 1.0

        A_target_state = _spectral_species_diffuse(
            A_initial_state, diffusion_constant_vals[A], lap_op, dt
        )
        B_target_state = _spectral_species_diffuse(
            B_initial_state, diffusion_constant_vals[B], lap_op, dt
        )
        C_target_state = _spectral_species_diffuse(
            C_initial_state, diffusion_constant_vals[C], lap_op, dt
        )

        target_state = {
            A: A_target_state,
            B: B_target_state,
            C: C_target_state,
        }

        computed_state = _spectral_diffuse(
            lap_op, initial_state, diffusion_constant_vals, dt
        )

        self.assertEqual(computed_state.keys(), target_state.keys())
        self.assertTrue(dict_allclose(computed_state, target_state))
