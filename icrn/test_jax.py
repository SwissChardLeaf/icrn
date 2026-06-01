"""Tests for jax.jit / jax.grad / jax.vmap composing with the public solver API.

These exercise the headline differentiability / batching guarantees of icrn.
We use a single first-order decay reaction (A --k--> 0) as the base system
because it has a closed-form solution A(t) = A0 * exp(-k*t), keeping the
tests tight on accuracy without depending on reference data files.

Each transformation (jit/grad/vmap/...) lives in a single class with one
method per solver (``solve_well_mixed`` and ``solve_reaction_diffusion``).
"""

import unittest

import jax
from jax import numpy as jnp

from .reactions import MassActionReaction
from .solver import solve_reaction_diffusion, solve_well_mixed
from .symbols import many_rate_constants, many_species


def _decay_setup():
    """Smallest non-trivial reaction system: A --k--> 0."""
    A = many_species("A")
    k = many_rate_constants("k")
    rxns = [MassActionReaction(A, 0, k)]
    times = jnp.array([0.0, 1.0])
    dt = 0.001
    return A, k, rxns, times, dt


def _decay_rd_setup():
    """Smallest non-trivial reaction-diffusion system: A --k--> 0 + diffusion
    of A on a small 2D grid."""
    A = many_species("A")
    k = many_rate_constants("k")
    rxns = [MassActionReaction(A, 0, k)]
    times = jnp.array([0.0, 1.0])
    dt = 0.01
    spatial_dims = (8, 8)
    dspaces = (1.0, 1.0)
    return A, k, rxns, times, dt, spatial_dims, dspaces


class TestJit(unittest.TestCase):
    def test_solve_well_mixed_under_jit(self):
        """jit-compiled well-mixed solve produces identical results to eager
        solve."""
        A, k, rxns, times, dt = _decay_setup()

        def solve_fn(conc, rate):
            return solve_well_mixed(rxns, conc, rate, times, dt)

        c0 = {A: jnp.array(1.0)}
        rates = {k: jnp.array(0.5)}

        eager = solve_fn(c0, rates)
        jitted = jax.jit(solve_fn)(c0, rates)

        self.assertTrue(jnp.allclose(eager[A], jitted[A], atol=1e-6))

    def test_solve_reaction_diffusion_under_jit(self):
        """jit-compiled reaction-diffusion solve produces identical results
        to eager solve."""
        A, k, rxns, times, dt, spatial_dims, dspaces = _decay_rd_setup()

        def solve_fn(conc, rate, diff):
            return solve_reaction_diffusion(
                rxns, conc, rate, diff, times, dt, spatial_dims, dspaces
            )

        c0 = {A: jnp.full(spatial_dims, 1.0)}
        rates = {k: jnp.array(0.5)}
        diffs = {A: jnp.array(0.1)}

        eager = solve_fn(c0, rates, diffs)
        jitted = jax.jit(solve_fn)(c0, rates, diffs)

        self.assertTrue(jnp.allclose(eager[A], jitted[A], atol=1e-6))


class TestGrad(unittest.TestCase):
    def test_solve_well_mixed_grad_returns_finite(self):
        """jax.grad through solve_well_mixed yields a finite, nonzero
        gradient."""
        A, k, rxns, times, dt = _decay_setup()

        def loss(rate):
            out = solve_well_mixed(
                rxns, {A: jnp.array(1.0)}, {k: rate}, times, dt
            )
            return jnp.sum(out[A] ** 2)

        g = jax.grad(loss)(jnp.array(1.0))

        self.assertTrue(jnp.isfinite(g))
        self.assertGreater(float(jnp.abs(g)), 1e-6)

    def test_solve_well_mixed_grad_matches_finite_difference(self):
        """Auto-diff gradient through solve_well_mixed agrees with central
        FD."""
        A, k, rxns, times, dt = _decay_setup()

        def loss(rate):
            out = solve_well_mixed(
                rxns, {A: jnp.array(1.0)}, {k: rate}, times, dt
            )
            return jnp.sum(out[A][-1] ** 2)

        k0 = jnp.array(1.0)
        eps = 1e-2

        g_auto = jax.grad(loss)(k0)
        g_fd = (loss(k0 + eps) - loss(k0 - eps)) / (2 * eps)

        self.assertTrue(jnp.allclose(g_auto, g_fd, rtol=1e-2, atol=1e-3))

    def test_solve_reaction_diffusion_grad_returns_finite(self):
        """jax.grad through solve_reaction_diffusion yields a finite, nonzero
        gradient."""
        A, k, rxns, times, dt, spatial_dims, dspaces = _decay_rd_setup()
        c0 = {A: jnp.full(spatial_dims, 1.0)}
        diffs = {A: jnp.array(0.1)}

        def loss(rate):
            out = solve_reaction_diffusion(
                rxns, c0, {k: rate}, diffs, times, dt, spatial_dims, dspaces
            )
            return jnp.sum(out[A][-1] ** 2)

        g = jax.grad(loss)(jnp.array(1.0))

        self.assertTrue(jnp.isfinite(g))
        self.assertGreater(float(jnp.abs(g)), 1e-6)

    def test_solve_reaction_diffusion_grad_matches_finite_difference(self):
        """Auto-diff gradient through solve_reaction_diffusion agrees with
        central FD."""
        A, k, rxns, times, dt, spatial_dims, dspaces = _decay_rd_setup()
        c0 = {A: jnp.full(spatial_dims, 1.0)}
        diffs = {A: jnp.array(0.1)}

        def loss(rate):
            out = solve_reaction_diffusion(
                rxns, c0, {k: rate}, diffs, times, dt, spatial_dims, dspaces
            )
            return jnp.sum(out[A][-1] ** 2)

        k0 = jnp.array(1.0)
        eps = 1e-2

        g_auto = jax.grad(loss)(k0)
        g_fd = (loss(k0 + eps) - loss(k0 - eps)) / (2 * eps)

        self.assertTrue(jnp.allclose(g_auto, g_fd, rtol=1e-2, atol=1e-3))


class TestVmap(unittest.TestCase):
    def test_solve_well_mixed_vmap_over_initial_conditions(self):
        """vmap over a batch of initial conditions matches an explicit Python
        loop (well-mixed)."""
        A, k, rxns, times, dt = _decay_setup()
        rates = {k: jnp.array(1.0)}

        def per_sample(c0):
            return solve_well_mixed(rxns, {A: c0}, rates, times, dt)[A][-1]

        c0_batch = jnp.array([1.0, 2.0, 3.0])

        batched = jax.vmap(per_sample)(c0_batch)
        explicit = jnp.array([per_sample(c) for c in c0_batch])

        self.assertTrue(jnp.allclose(batched, explicit, atol=1e-6))

    def test_solve_well_mixed_vmap_over_rate_constants(self):
        """vmap over a batch of rate constants matches an explicit Python
        loop (well-mixed)."""
        A, k, rxns, times, dt = _decay_setup()
        c0 = {A: jnp.array(1.0)}

        def per_sample(rate):
            return solve_well_mixed(rxns, c0, {k: rate}, times, dt)[A][-1]

        rate_batch = jnp.array([0.5, 1.0, 2.0])

        batched = jax.vmap(per_sample)(rate_batch)
        explicit = jnp.array([per_sample(r) for r in rate_batch])

        self.assertTrue(jnp.allclose(batched, explicit, atol=1e-6))

    def test_solve_reaction_diffusion_vmap_over_initial_fields(self):
        """vmap over a batch of initial fields matches an explicit Python
        loop (RD)."""
        A, k, rxns, times, dt, spatial_dims, dspaces = _decay_rd_setup()
        rates = {k: jnp.array(1.0)}
        diffs = {A: jnp.array(0.1)}

        def per_sample(c0_field):
            return solve_reaction_diffusion(
                rxns,
                {A: c0_field},
                rates,
                diffs,
                times,
                dt,
                spatial_dims,
                dspaces,
            )[A][-1]

        c0_batch = jnp.stack(
            [
                jnp.full(spatial_dims, 1.0),
                jnp.full(spatial_dims, 2.0),
                jnp.full(spatial_dims, 0.5),
            ]
        )

        batched = jax.vmap(per_sample)(c0_batch)
        explicit = jnp.stack([per_sample(c) for c in c0_batch])

        self.assertTrue(jnp.allclose(batched, explicit, atol=1e-6))

    def test_solve_reaction_diffusion_vmap_over_diffusion_constants(self):
        """vmap over a batch of diffusion constants matches an explicit
        Python loop (RD)."""
        A, k, rxns, times, dt, spatial_dims, dspaces = _decay_rd_setup()
        c0 = {A: jnp.full(spatial_dims, 1.0)}
        rates = {k: jnp.array(1.0)}

        def per_sample(D):
            return solve_reaction_diffusion(
                rxns,
                c0,
                rates,
                {A: D},
                times,
                dt,
                spatial_dims,
                dspaces,
            )[A][-1]

        D_batch = jnp.array([0.05, 0.1, 0.2])

        batched = jax.vmap(per_sample)(D_batch)
        explicit = jnp.stack([per_sample(d) for d in D_batch])

        self.assertTrue(jnp.allclose(batched, explicit, atol=1e-6))


class TestCompositions(unittest.TestCase):
    def test_solve_well_mixed_jit_of_grad(self):
        """jit(grad(...)) — the canonical training-step pattern (well-mixed)."""
        A, k, rxns, times, dt = _decay_setup()

        def loss(rate):
            out = solve_well_mixed(
                rxns, {A: jnp.array(1.0)}, {k: rate}, times, dt
            )
            return jnp.sum(out[A] ** 2)

        g_fn = jax.jit(jax.grad(loss))
        g = g_fn(jnp.array(1.0))

        self.assertTrue(jnp.isfinite(g))
        self.assertGreater(float(jnp.abs(g)), 1e-6)

    def test_solve_well_mixed_vmap_of_grad(self):
        """vmap(grad(...)) — per-sample gradients across a batch of rate
        constants (well-mixed)."""
        A, k, rxns, times, dt = _decay_setup()

        def loss(rate):
            out = solve_well_mixed(
                rxns, {A: jnp.array(1.0)}, {k: rate}, times, dt
            )
            return jnp.sum(out[A] ** 2)

        rate_batch = jnp.array([0.5, 1.0, 2.0])
        g_batch = jax.vmap(jax.grad(loss))(rate_batch)

        self.assertEqual(g_batch.shape, rate_batch.shape)
        self.assertTrue(jnp.all(jnp.isfinite(g_batch)))

    def test_solve_reaction_diffusion_jit_of_grad(self):
        """jit(grad(...)) — the canonical training-step pattern (RD)."""
        A, k, rxns, times, dt, spatial_dims, dspaces = _decay_rd_setup()
        c0 = {A: jnp.full(spatial_dims, 1.0)}
        diffs = {A: jnp.array(0.1)}

        def loss(rate):
            out = solve_reaction_diffusion(
                rxns, c0, {k: rate}, diffs, times, dt, spatial_dims, dspaces
            )
            return jnp.sum(out[A][-1] ** 2)

        g_fn = jax.jit(jax.grad(loss))
        g = g_fn(jnp.array(1.0))

        self.assertTrue(jnp.isfinite(g))
        self.assertGreater(float(jnp.abs(g)), 1e-6)

    def test_solve_reaction_diffusion_vmap_of_grad(self):
        """vmap(grad(...)) — per-sample gradients across a batch of rate
        constants (RD)."""
        A, k, rxns, times, dt, spatial_dims, dspaces = _decay_rd_setup()
        c0 = {A: jnp.full(spatial_dims, 1.0)}
        diffs = {A: jnp.array(0.1)}

        def loss(rate):
            out = solve_reaction_diffusion(
                rxns, c0, {k: rate}, diffs, times, dt, spatial_dims, dspaces
            )
            return jnp.sum(out[A][-1] ** 2)

        rate_batch = jnp.array([0.5, 1.0, 2.0])
        g_batch = jax.vmap(jax.grad(loss))(rate_batch)

        self.assertEqual(g_batch.shape, rate_batch.shape)
        self.assertTrue(jnp.all(jnp.isfinite(g_batch)))


class Test64Bit(unittest.TestCase):
    """Verify icrn preserves float64 precision when JAX's x64 mode is enabled.

    JAX defaults to float32 and silently downcasts even explicitly typed
    float64 inputs unless ``jax_enable_x64`` is set. We toggle it on per-test
    (and restore the original setting in ``tearDown``) so other tests in the
    suite — which assume the float32 default — aren't affected.
    """

    def setUp(self):
        self._original_x64 = jax.config.x64_enabled
        jax.config.update("jax_enable_x64", True)

    def tearDown(self):
        jax.config.update("jax_enable_x64", self._original_x64)

    def test_solve_well_mixed_preserves_float64(self):
        """solve_well_mixed preserves float64 across every leaf of its
        output."""
        A, k, rxns, times, dt = _decay_setup()

        out = solve_well_mixed(
            rxns,
            {A: jnp.array(1.0)},
            {k: jnp.array(1.0)},
            times,
            dt,
        )

        for leaf in jax.tree_util.tree_leaves(out):
            self.assertEqual(leaf.dtype, jnp.float64)

    def test_solve_reaction_diffusion_preserves_float64(self):
        """solve_reaction_diffusion preserves float64 across every leaf of
        its output."""
        A, k, rxns, times, dt, spatial_dims, dspaces = _decay_rd_setup()

        out = solve_reaction_diffusion(
            rxns,
            {A: jnp.full(spatial_dims, 1.0)},
            {k: jnp.array(1.0)},
            {A: jnp.array(0.1)},
            times,
            dt,
            spatial_dims,
            dspaces,
        )

        for leaf in jax.tree_util.tree_leaves(out):
            self.assertEqual(leaf.dtype, jnp.float64)


if __name__ == "__main__":
    unittest.main()
