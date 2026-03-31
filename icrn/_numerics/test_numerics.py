import unittest

import os
import numpy as np
import jax.numpy as jnp
from ._spectral_diffusion import (
    _compute_lap_op,
    _spectral_diffuse,
    _spectral_species_diffuse,
    _test,
)

# from ..representation.symbols import many_species
import jax


class SpectralDiffusionTests(unittest.TestCase):
    def test_compute_lap_op(self):
        computed_lap_op1 = _compute_lap_op(4, 3, 1, 1)

        target_lap_op1 = jnp.array(
            [
                [-0.0, -4.3864913, -4.3864913],
                [-2.4674013, -6.8538923, -6.8538923],
                [-9.869605, -14.256096, -14.256096],
                [-2.4674013, -6.8538923, -6.8538923],
            ]
        )

        self.assertTrue(jnp.allclose(computed_lap_op1, target_lap_op1))

        computed_lap_op2 = _compute_lap_op(2, 3, 0.5, 2.0)

        target_lap_op2 = jnp.array(
            [[-0.0, -1.0966228, -1.0966228], [-39.47842, -40.575043, -40.575043]]
        )

        self.assertTrue(jnp.allclose(computed_lap_op2, target_lap_op2))

        computed_lap_op3 = _compute_lap_op(1, 4, 1, 1)
        target_lap_op3 = jnp.array([[-0.0, -2.4674013, -9.869605, -2.4674013]])
        self.assertTrue(jnp.allclose(computed_lap_op3, target_lap_op3))

    def test_spectral_species_diffuse(self):
        lap_op = _compute_lap_op(5, 5, 1, 1)

        initial_state1 = jnp.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        target_state1 = jnp.array(
            [
                [0.03760443, 0.03843227, 0.03896552, 0.03843227, 0.03760443],
                [0.03843226, 0.0406484, 0.04309084, 0.0406484, 0.03843226],
                [0.03896552, 0.04309085, 0.05130513, 0.04309085, 0.03896552],
                [0.03843226, 0.0406484, 0.04309084, 0.0406484, 0.03843226],
                [0.03760443, 0.03843226, 0.03896552, 0.03843226, 0.03760443],
            ]
        )

        computed_state1 = _spectral_species_diffuse(
            initial_state1, kd=jnp.array(10.0), lap_op=lap_op, dt=2.0
        )
        self.assertTrue(jnp.allclose(computed_state1, target_state1))

        initial_state2 = jnp.repeat(
            initial_state1[..., jnp.newaxis], repeats=3, axis=-1
        )

        target_state2 = jnp.stack(
            [
                _spectral_species_diffuse(
                    initial_state1, kd=jnp.array(1.0), lap_op=lap_op, dt=2.0
                ),
                _spectral_species_diffuse(
                    initial_state1, kd=jnp.array(2.0), lap_op=lap_op, dt=2.0
                ),
                _spectral_species_diffuse(
                    initial_state1, kd=jnp.array(10.0), lap_op=lap_op, dt=2.0
                ),
            ],
            axis=-1,
        )

        computed_state2 = _spectral_species_diffuse(
            initial_state2, kd=jnp.array([1.0, 2.0, 10.0]), lap_op=lap_op, dt=2.0
        )
        self.assertTrue(jnp.allclose(computed_state2, target_state2))
        self.assertTrue(jnp.allclose(computed_state2[..., 2], target_state1))

    # @unittest.skip("")
    def test_spectral_diffuse(self):
        lap_op = _compute_lap_op(5, 5, 1, 1)

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

        initial_state_dict = {
            1: initial_state,
            2: jnp.tile(initial_state[..., jnp.newaxis], (1, 1, 3)),
            3: jnp.tile(initial_state[..., jnp.newaxis, jnp.newaxis], (1, 1, 2, 3)),
        }

        target_state_dict = {
            1: target_state,
            2: jnp.stack(
                [
                    _spectral_species_diffuse(
                        initial_state, kd=jnp.array(1.0), lap_op=lap_op, dt=2.0
                    ),
                    _spectral_species_diffuse(
                        initial_state, kd=jnp.array(2.0), lap_op=lap_op, dt=2.0
                    ),
                    _spectral_species_diffuse(
                        initial_state, kd=jnp.array(10.0), lap_op=lap_op, dt=2.0
                    ),
                ],
                axis=-1,
            ),
            3: jnp.tile(target_state[..., jnp.newaxis, jnp.newaxis], (1, 1, 2, 3)),
        }

        kd_dict = {
            1: jnp.array(10.0),
            2: jnp.array([1.0, 2.0, 10.0]),
            3: jnp.array([[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]),
        }

        computed_state_dict = _spectral_diffuse(
            lap_op, initial_state_dict, kd_dict, dt=2.0
        )

        dicts_all_close = jax.tree_util.tree_map(
            lambda x, y: jnp.allclose(x, y), computed_state_dict, target_state_dict
        )

        self.assertFalse(False in dicts_all_close.values())

        # self.assertTrue(sjdict_allclose(computed_state, target_state_sjdict))

    def test_test(self):
        a = {
            1: jnp.arange(10, dtype="float"),
            2: jnp.arange(12, dtype="float").reshape((3, 4)),
        }

        b = {1: jnp.ones((10,)), 2: jnp.ones((12,)).reshape((3, 4))}

        res = _test(a, b)

        self.assertTrue(jnp.allclose(res[1], jnp.sum(jnp.arange(10) + 1, axis=0)))
        self.assertTrue(
            jnp.allclose(res[2], jnp.sum(jnp.arange(12).reshape((3, 4)) + 1, axis=0))
        )


class ConvolutionalDiffusionTests(unittest.TestCase):
    pass


class BuildForwardStepTests(unittest.TestCase):
    pass


class ScanBySegmentTests(unittest.TestCase):
    pass


class IntegratorTests(unittest.TestCase):
    pass


class DiffraxTests(unittest.TestCase):
    pass
