import unittest

import jax.numpy as jnp

import icrn._testing.configure  # noqa: F401  # CPU JAX before import

from .operator import (
    AbstractOperator,
    FastReactionsOperator,
    ReactionsOperator,
    SpectralDiffusionOperator,
)
from .reactions import FastReaction, MassActionReaction
from .symbols import (
    many_index_symbols,
    many_rate_constants,
    many_species,
)
from .utils.dict_utils import dict_allclose

rxn = MassActionReaction


class TestExtendingAbstractOperator(unittest.TestCase):
    def test_subtracting(self):
        class TestOperator(AbstractOperator):
            def update_state(self, solver_state, non_state, dt):
                solver_state["A"] -= non_state["a"] * dt
                solver_state["B"] -= non_state["b"] * dt
                return solver_state

            def get_mode(self):
                return None

            def get_is_stochastic(self):
                return False

        test_op = TestOperator()
        self.assertEqual(test_op.get_mode(), None)
        self.assertEqual(test_op.get_is_stochastic(), False)

        state = {"A": 1.0, "B": 2.0}
        non_state = {"a": 0.1, "b": 0.2}
        dt = 1.0
        self.assertEqual(
            test_op.update_state(state, non_state, dt), {"A": 0.9, "B": 1.8}
        )


class TestReactionsOperator(unittest.TestCase):
    # def setUp(self):
    #     A, B, C = many_species("A, B, C")
    #     alpha, beta = many_rate_constants("alpha, beta")
    #     i, j = many_index_symbols("i, j")

    #     self.rxns1 = [
    #         rxn(A + B, C, alpha),
    #         rxn(C, A + B, beta),
    #     ]

    #     self.rxns2 = [
    #         rxn(A[i] + B[j], C[i, j], alpha[i, j]),
    #         rxn(C[i, j], A[i] + B[j], beta[i, j]),
    #     ]

    #     self.rxns3 = [
    #         rxn(2*A + 3 * B, 2*A + 2*B, alpha),
    #         rxn(A + B, A + C, beta),
    #     ]
    def test_exponential_decay(self):
        A = many_species("A")
        k = many_rate_constants("k")

        rxns = [
            rxn(A, 0, k),
        ]

        op1 = ReactionsOperator(None, rxns, reaction_solver="Euler")

        self.assertEqual(op1.get_mode(), None)
        self.assertEqual(op1.get_is_stochastic(), False)

        state = {A: jnp.array(1.0)}
        non_state = {k: jnp.array(1.0)}
        dt = jnp.log(2)
        new_state = op1.update_state(state, non_state, dt)

        self.assertEqual(new_state, {A: 1 - jnp.log(2)})

        op2 = ReactionsOperator(None, rxns, reaction_solver="RK4")
        dt = jnp.log(2)
        new_state = op2.update_state(state, non_state, dt)
        self.assertTrue(
            dict_allclose(new_state, {A: jnp.array(0.5011933468419623)})
        )

        # op3 = ReactionsOperator("strict", rxns, reaction_solver="Euler")

        # self.assertEqual(op3.get_mode(), "strict")

        # state = {A: jnp.array(1.0)}
        # non_state = {k: jnp.array(1.0)}
        # dt = 1.1

        # new_state = op3.update_state(state, non_state, dt)
        # print(new_state)

        # with self.assertRaises(ValueError):
        #     op3.update_state(state, non_state, dt)

    def test_reversible_dimerization(self):
        A, B, C = many_species("A, B, C")
        alpha, beta = many_rate_constants("alpha, beta")

        rxns = [
            rxn(A + B, C, alpha),
            rxn(C, A + B, beta),
        ]

        op1 = ReactionsOperator(None, rxns, reaction_solver="Euler")

        state = {
            A: jnp.array(1.0),
            B: jnp.array(2.0),
            C: jnp.array(3.0),
        }

        non_state = {
            alpha: jnp.array(1.1),
            beta: jnp.array(2.2),
        }

        dt = 1.0

        new_state = op1.update_state(state, non_state, dt)

        self.assertTrue(
            dict_allclose(
                new_state,
                {
                    A: jnp.array(1.0 + dt * (2.2 * 3.0 - 1.1 * 1.0 * 2.0)),
                    B: jnp.array(2.0 + dt * (2.2 * 3.0 - 1.1 * 1.0 * 2.0)),
                    C: jnp.array(3.0 + dt * (1.1 * 1.0 * 2.0 - 2.2 * 3.0)),
                },
            )
        )

        ReactionsOperator(None, rxns, reaction_solver="RK4")

    def test_exponential_decay_spatial(self):

        A = many_species("A")
        k = many_rate_constants("k")

        rxns = [
            rxn(A, 0, k),
        ]

        op = ReactionsOperator(
            None,
            rxns,
            reaction_solver="Euler",
            spatial_axes=1,
            spatial_rate_constants=True,
        )

        state = {A: jnp.array([1.0, 2.0, 3.0])}
        non_state = ({k: jnp.array([1.0, 0.5, 1 / 3])}, None)
        dt = 1.0
        new_state = op.update_state(state, non_state, dt)
        self.assertTrue(
            dict_allclose(new_state, {A: jnp.array([0.0, 1.0, 2.0])})
        )

    def test_gray_scott_spatial(self):
        U, V = many_species("U, V")
        F, k = many_rate_constants("F, k")

        rxns = [
            MassActionReaction(U + 2 * V, 3 * V, 1),
            MassActionReaction(V, 0, F + k),
            MassActionReaction(0, U, F),
            MassActionReaction(U, 0, F),
        ]

        op = ReactionsOperator(
            None,
            rxns,
            reaction_solver="Euler",
            spatial_axes=2,
            spatial_rate_constants=True,
        )

        state = {
            U: jnp.array(
                [
                    [0.4, 0.5],
                    [0.6, 0.7],
                ]
            ),
            V: jnp.array(
                [
                    [0.1, 0.2],
                    [0.3, 0.4],
                ]
            ),
        }
        non_state = (
            {
                F: jnp.array(
                    [
                        [0.08, 0.03],
                        [0.04, 0.07],
                    ]
                ),
                k: jnp.array(
                    [
                        [0.04, 0.07],
                        [0.08, 0.03],
                    ]
                ),
            },
            None,
        )

        dt = 1.0
        new_state = op.update_state(state, non_state, dt)
        self.assertTrue(
            dict_allclose(
                new_state,
                {
                    U: jnp.array([[0.444, 0.495], [0.56200004, 0.60899997]]),
                    V: jnp.array(
                        [
                            [0.092, 0.2],
                            [0.31800002, 0.472],
                        ]
                    ),
                },
            )
        )


class TestFastReactionsOperator(unittest.TestCase):
    def test_annihilation(self):
        A, B = many_species("A, B")
        i = many_index_symbols("i")

        fast_rxns = [
            FastReaction(A[i] + B[i], 0),
        ]

        # print(fast_rxns[0].products)

        op = FastReactionsOperator(fast_rxns)

        state = {A: jnp.array([1.0, 2.0, 1.5]), B: jnp.array([2.0, 0.0, 1.5])}

        new_state = op.update_state(state, None, 1.0)
        target_state = {
            A: jnp.array([0.0, 2.0, 0.0]),
            B: jnp.array([1.0, 0.0, 0.0]),
        }

        # print(new_state)
        self.assertTrue(dict_allclose(new_state, target_state))

    def test_annihilation_spatial(self):
        A, B = many_species("A, B")
        i = many_index_symbols("i")

        fast_rxns = [
            FastReaction(A[i] + B[i], 0),
        ]

        op = FastReactionsOperator(fast_rxns, spatial_axes=1)

        state = {
            A: jnp.array([[1.0, 2.0, 1.5], [2.0, 3.0, 4.5]]),
            B: jnp.array([[2.0, 1.0, 1.5], [1.0, 2.0, 1.5]]),
        }
        non_state = (None, None)
        dt = 1.0
        new_state = op.update_state(state, non_state, dt)
        target_state = {
            A: jnp.array([[0.0, 1.0, 0.0], [1.0, 1.0, 3.0]]),
            B: jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        }
        self.assertTrue(dict_allclose(new_state, target_state))


class TestSpectralDiffusionOperator(unittest.TestCase):
    def test_scalar_species_diffusion(self):
        A = many_species("A")

        op = SpectralDiffusionOperator(None, (5, 5), (1, 1), dt_scale=2.0)

        initial_state = {
            A: jnp.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        }

        target_state = {
            A: jnp.array(
                [
                    [
                        0.03760443,
                        0.03843227,
                        0.03896552,
                        0.03843227,
                        0.03760443,
                    ],
                    [0.03843226, 0.0406484, 0.04309084, 0.0406484, 0.03843226],
                    [
                        0.03896552,
                        0.04309085,
                        0.05130513,
                        0.04309085,
                        0.03896552,
                    ],
                    [0.03843226, 0.0406484, 0.04309084, 0.0406484, 0.03843226],
                    [
                        0.03760443,
                        0.03843226,
                        0.03896552,
                        0.03843226,
                        0.03760443,
                    ],
                ]
            )
        }

        non_state = (None, {A: jnp.array(10.0)})

        computed_state = op.update_state(initial_state, non_state, 1.0)
        self.assertTrue(dict_allclose(computed_state, target_state))


class TestConvolutionalDiffusionOperator(unittest.TestCase):
    pass
