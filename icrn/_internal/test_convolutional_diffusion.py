import unittest

import jax
from jax import numpy as jnp

from ._convolutional_diffusion import _conv_diffuse, _conv_species_diffuse

D_DEFAULT = jnp.array(0.1)


def _peaked(spatial_shape, value=10.0):
    center = tuple(s // 2 for s in spatial_shape)
    return jnp.zeros(spatial_shape).at[center].set(value)


def _gaussian_1d(x, x0, sigma):
    return jnp.exp(-((x - x0) ** 2) / (2.0 * sigma**2))


def _gaussian_1d_analytical(x, x0, sigma_0, D, t):
    sigma_t_sq = sigma_0**2 + 2.0 * D * t
    return jnp.sqrt(sigma_0**2 / sigma_t_sq) * jnp.exp(
        -((x - x0) ** 2) / (2.0 * sigma_t_sq)
    )


def _evolve(conc, D, dt, dspaces, n_steps, bc="neumann"):
    def step(c, _):
        return _conv_species_diffuse(c, D, dt, dspaces, bc), None

    out, _ = jax.lax.scan(step, conc, None, length=n_steps)
    return out


class _ConvSpeciesDiffuseMixin:
    DSPACES = ()
    SPATIAL_SHAPE = ()
    DT = 0.001
    SEED = 0

    def test_output_shape_preserved(self):
        conc = jnp.ones(self.SPATIAL_SHAPE)
        out = _conv_species_diffuse(conc, D_DEFAULT, self.DT, self.DSPACES)
        self.assertEqual(out.shape, conc.shape)

    def test_output_dtype_preserved(self):
        conc = jnp.ones(self.SPATIAL_SHAPE, dtype=jnp.float32)
        out = _conv_species_diffuse(conc, D_DEFAULT, self.DT, self.DSPACES)
        self.assertEqual(out.dtype, conc.dtype)

    def test_zero_diffusion_is_identity(self):
        conc = jax.random.uniform(
            jax.random.PRNGKey(self.SEED), self.SPATIAL_SHAPE
        )
        out = _conv_species_diffuse(conc, jnp.array(0.0), self.DT, self.DSPACES)
        self.assertTrue(jnp.allclose(out, conc, atol=1e-7))

    def test_uniform_field_unchanged_under_neumann(self):
        conc = jnp.ones(self.SPATIAL_SHAPE)
        out = _conv_species_diffuse(
            conc, D_DEFAULT, self.DT, self.DSPACES, "neumann"
        )
        self.assertTrue(jnp.allclose(out, conc, atol=1e-6))

    def test_diffusion_smooths_peak(self):
        conc = _peaked(self.SPATIAL_SHAPE, value=10.0)
        out = _conv_species_diffuse(conc, D_DEFAULT, self.DT, self.DSPACES)

        center = tuple(s // 2 for s in self.SPATIAL_SHAPE)
        self.assertLess(float(out[center]), 10.0)
        for axis in range(len(self.SPATIAL_SHAPE)):
            neighbor = list(center)
            neighbor[axis] += 1
            self.assertGreater(float(out[tuple(neighbor)]), 0.0)


class TestConvSpeciesDiffuse1D(_ConvSpeciesDiffuseMixin, unittest.TestCase):
    DSPACES = (1.0,)
    SPATIAL_SHAPE = (32,)
    SEED = 1


class TestConvSpeciesDiffuse2D(_ConvSpeciesDiffuseMixin, unittest.TestCase):
    DSPACES = (1.0, 1.0)
    SPATIAL_SHAPE = (16, 16)
    SEED = 2


class TestConvSpeciesDiffuse3D(_ConvSpeciesDiffuseMixin, unittest.TestCase):
    DSPACES = (1.0, 1.0, 1.0)
    SPATIAL_SHAPE = (8, 8, 8)
    SEED = 3


class TestBoundaryConditions(unittest.TestCase):
    DSPACES = (1.0, 1.0)
    SPATIAL_SHAPE = (16, 16)
    DT = 0.001

    def test_neumann_conserves_total_mass(self):
        conc = jax.random.uniform(jax.random.PRNGKey(10), self.SPATIAL_SHAPE)
        out = _conv_species_diffuse(
            conc, D_DEFAULT, self.DT, self.DSPACES, "neumann"
        )
        self.assertTrue(jnp.allclose(jnp.sum(out), jnp.sum(conc), atol=1e-4))

    def test_dirichlet_leaks_mass_at_boundary(self):
        conc = jnp.ones(self.SPATIAL_SHAPE)
        out = _conv_species_diffuse(
            conc, D_DEFAULT, self.DT, self.DSPACES, "dirichlet"
        )
        self.assertLess(float(jnp.sum(out)), float(jnp.sum(conc)))

    def test_periodic_wraps_around(self):
        H, W = self.SPATIAL_SHAPE
        conc = jnp.zeros(self.SPATIAL_SHAPE).at[0, W // 2].set(1.0)
        out = _conv_species_diffuse(
            conc, D_DEFAULT, self.DT, self.DSPACES, "periodic"
        )
        self.assertGreater(float(out[-1, W // 2]), 0.0)

    def test_neumann_does_not_wrap_around(self):
        H, W = self.SPATIAL_SHAPE
        conc = jnp.zeros(self.SPATIAL_SHAPE).at[0, W // 2].set(1.0)
        out = _conv_species_diffuse(
            conc, D_DEFAULT, self.DT, self.DSPACES, "neumann"
        )
        self.assertLess(float(jnp.max(jnp.abs(out[-1, :]))), 1e-7)

    def test_dirichlet_does_not_wrap_around(self):
        H, W = self.SPATIAL_SHAPE
        conc = jnp.zeros(self.SPATIAL_SHAPE).at[0, W // 2].set(1.0)
        out = _conv_species_diffuse(
            conc, D_DEFAULT, self.DT, self.DSPACES, "dirichlet"
        )
        self.assertLess(float(jnp.max(jnp.abs(out[-1, :]))), 1e-7)

    def test_unknown_boundary_condition_raises(self):
        conc = jnp.ones(self.SPATIAL_SHAPE)
        with self.assertRaises(ValueError):
            _conv_species_diffuse(
                conc, D_DEFAULT, self.DT, self.DSPACES, "no-such-bc"
            )


class TestDspacesScaling(unittest.TestCase):
    SPATIAL_SHAPE = (16, 16)
    DT = 0.001

    def test_smaller_dspace_means_stronger_diffusion(self):
        conc = _peaked(self.SPATIAL_SHAPE, value=10.0)
        center = tuple(s // 2 for s in self.SPATIAL_SHAPE)

        out_coarse = _conv_species_diffuse(conc, D_DEFAULT, self.DT, (1.0, 1.0))
        out_fine = _conv_species_diffuse(conc, D_DEFAULT, self.DT, (0.5, 0.5))

        drop_coarse = 10.0 - float(out_coarse[center])
        drop_fine = 10.0 - float(out_fine[center])
        self.assertGreater(drop_fine, drop_coarse)


class TestNumericalAccuracy(unittest.TestCase):
    L = 10.0
    SIGMA_0 = 1.0
    D = jnp.array(0.1)
    BC = "neumann"

    def _setup_grid(self, N):
        dx = self.L / N
        x = jnp.arange(N) * dx
        return x, dx

    def _run(self, N, dt, T):
        x, dx = self._setup_grid(N)
        x0 = self.L / 2
        c0 = _gaussian_1d(x, x0, self.SIGMA_0)
        truth = _gaussian_1d_analytical(x, x0, self.SIGMA_0, float(self.D), T)

        n_steps = int(round(T / dt))
        out = _evolve(c0, self.D, dt, (dx,), n_steps, self.BC)
        return float(jnp.max(jnp.abs(out - truth)))

    def test_temporal_error_decreases_when_dt_is_smaller(self):
        N = 200
        T = 0.5
        dt_coarse = 0.01
        dt_fine = 0.005

        err_coarse = self._run(N, dt_coarse, T)
        err_fine = self._run(N, dt_fine, T)

        self.assertLess(err_fine, 0.5 * err_coarse)
        self.assertLess(err_fine, 1e-4)

    def test_spatial_error_decreases_when_dspace_is_smaller(self):
        T = 0.1
        dt = 0.001

        err_coarse = self._run(N=50, dt=dt, T=T)
        err_fine = self._run(N=100, dt=dt, T=T)

        self.assertLess(err_fine, 0.5 * err_coarse)
        self.assertLess(err_fine, 1e-4)


class TestJaxCompatibility(unittest.TestCase):
    DSPACES = (1.0, 1.0)
    SPATIAL_SHAPE = (16, 16)
    DT = 0.001

    def test_jit_compatible(self):
        conc = jnp.ones(self.SPATIAL_SHAPE)
        eager = _conv_species_diffuse(conc, D_DEFAULT, self.DT, self.DSPACES)
        jitted = jax.jit(
            _conv_species_diffuse,
            static_argnames=("boundary_condition",),
        )(conc, D_DEFAULT, self.DT, self.DSPACES)
        self.assertTrue(jnp.allclose(eager, jitted, atol=1e-6))

    def test_vmap_over_initial_conditions(self):
        c0_batch = jnp.stack(
            [
                jnp.ones(self.SPATIAL_SHAPE),
                _peaked(self.SPATIAL_SHAPE, value=5.0),
                jax.random.uniform(jax.random.PRNGKey(20), self.SPATIAL_SHAPE),
            ]
        )

        def per_sample(c):
            return _conv_species_diffuse(c, D_DEFAULT, self.DT, self.DSPACES)

        batched = jax.vmap(per_sample)(c0_batch)
        explicit = jnp.stack([per_sample(c) for c in c0_batch])
        self.assertTrue(jnp.allclose(batched, explicit, atol=1e-6))

    def test_grad_through_kernel_returns_finite(self):
        conc = _peaked(self.SPATIAL_SHAPE, value=1.0)

        def loss(D):
            return jnp.sum(
                _conv_species_diffuse(conc, D, self.DT, self.DSPACES) ** 2
            )

        g = jax.grad(loss)(D_DEFAULT)
        self.assertTrue(jnp.isfinite(g))
        self.assertGreater(float(jnp.abs(g)), 1e-8)


class TestConvDiffuseMultiSpecies(unittest.TestCase):
    DSPACES = (1.0, 1.0)
    SPATIAL_SHAPE = (16, 16)
    DT = 0.001

    def test_multispecies_runs(self):
        concs = {
            "A": jnp.ones(self.SPATIAL_SHAPE),
            "B": jnp.full(self.SPATIAL_SHAPE, 2.0),
        }
        diffs = {
            "A": jnp.array(0.1),
            "B": jnp.array(0.05),
        }
        out = _conv_diffuse(concs, diffs, self.DT, self.DSPACES)

        self.assertEqual(set(out.keys()), set(concs.keys()))
        for sp in concs:
            self.assertEqual(out[sp].shape, self.SPATIAL_SHAPE)

    def test_multispecies_independence(self):
        concs = {
            "A": _peaked(self.SPATIAL_SHAPE, value=5.0),
            "B": _peaked(self.SPATIAL_SHAPE, value=5.0),
        }
        diffs = {
            "A": jnp.array(0.0),
            "B": jnp.array(0.1),
        }
        out = _conv_diffuse(concs, diffs, self.DT, self.DSPACES)
        self.assertTrue(jnp.allclose(out["A"], concs["A"], atol=1e-7))
        self.assertFalse(jnp.allclose(out["B"], concs["B"], atol=1e-7))

    def test_multispecies_works_in_3d(self):
        spatial_shape = (8, 8, 8)
        dspaces = (1.0, 1.0, 1.0)
        concs = {
            "A": _peaked(spatial_shape, value=5.0),
            "B": _peaked(spatial_shape, value=5.0),
        }
        diffs = {"A": jnp.array(0.1), "B": jnp.array(0.05)}
        out = _conv_diffuse(concs, diffs, self.DT, dspaces)

        for sp in concs:
            self.assertEqual(out[sp].shape, spatial_shape)


if __name__ == "__main__":
    unittest.main()
