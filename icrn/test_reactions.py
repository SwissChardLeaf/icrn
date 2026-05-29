import unittest
from dataclasses import FrozenInstanceError

import jax.numpy as jnp

from .reactions import (
    FastReaction,
    MassActionReaction,
    _matching_shapes,
    fast_rxns_to_update_f,
    rxns_to_dynamics_f,
)
from .symbols import (
    Complex,
    RateConstant,
    TensorLiteral,
    many_index_symbols,
    many_rate_constants,
    many_species,
)
from .utils.dict_utils import dict_allclose


class TestReactionHelpers(unittest.TestCase):
    def test_matching_shapes(self):
        A, B, C = many_species("A, B, C")
        i, j = many_index_symbols("i, j")
        idx_l, m = many_index_symbols("l, m", 3)
        n, o = many_index_symbols("n, o", 7)

        _matching_shapes(set([A[i], B[j], A[j], B[i]]))
        _matching_shapes(set([A[idx_l], B[j], A[m], B[n]]))
        _matching_shapes(set([A[i, n], B[j], A[idx_l, j], B[o]]))
        _matching_shapes(set([A[i, j, m], A[j, idx_l, i], A[n, i, j]]))

        with self.assertRaises(ValueError):
            _matching_shapes(set([A[idx_l], A[n]]))

        with self.assertRaises(ValueError):
            _matching_shapes(set([A[i], A[j, i]]))

        with self.assertRaises(ValueError):
            _matching_shapes(set([A[i], A]))

        with self.assertRaises(ValueError):
            _matching_shapes(set([A[i, j, m], A[idx_l, m, o]]))


class TestRxnsToDynamicsF(unittest.TestCase):
    def test_dimerization(self):
        A, B, C = many_species("A, B, C")
        alpha = RateConstant("alpha")
        beta = RateConstant("beta")

        rxns = [
            MassActionReaction(A + B, C, alpha),
            MassActionReaction(C, A + B, beta),
        ]

        state = {A: jnp.array(1.0), B: jnp.array(2.0), C: jnp.array(3.0)}
        non_state = {alpha: jnp.array(1.1), beta: jnp.array(2.2)}

        dynamics_f = rxns_to_dynamics_f(rxns)
        computed_dynamics = dynamics_f(state, non_state)
        target_dynamics = {
            A: non_state[beta] * state[C]
            - non_state[alpha] * state[A] * state[B],
            B: non_state[beta] * state[C]
            - non_state[alpha] * state[A] * state[B],
            C: non_state[alpha] * state[A] * state[B]
            - non_state[beta] * state[C],
        }
        self.assertTrue(dict_allclose(computed_dynamics, target_dynamics))

    def test_exponential_decay(self):
        A = many_species("A")
        k = many_rate_constants("k")

        rxns = [
            MassActionReaction(A, 0, k),
        ]

        state = {A: jnp.array(1.0)}
        non_state = {k: jnp.array(1.0)}
        dynamics_f = rxns_to_dynamics_f(rxns)
        computed_dynamics = dynamics_f(state, non_state)
        target_dynamics = {A: -non_state[k] * state[A]}
        self.assertTrue(dict_allclose(computed_dynamics, target_dynamics))


class TestExtendAbstractReaction(unittest.TestCase):
    pass
    # def setUp(self):
    #     class TestReaction(AbstractReaction):
    #         A, B = many_species("A, B")
    #         alpha = RateConstant("alpha")

    #         def __init__(self, reactants, products, aux):
    #             super().__init__(reactants, products, aux)

    #         def flux(self):
    #             def flux_fn(state, rate_constant_data):
    # return {self.reactants: -self.aux, self.products: self.aux}

    #             return flux_fn

    #     self.TestReaction = TestReaction(A, B, alpha)

    # def test_flux(self):
    #     A, B = many_species("A, B")
    #     alpha = RateConstant("alpha")
    #     self.assertEqual(self.TestReaction.flux(), {A: -alpha, B: alpha})


class TestMassActionReaction(unittest.TestCase):
    def test_init(self):
        A, B, C = many_species("A, B, C")
        alpha = RateConstant("alpha")

        rxn = MassActionReaction(A + B, C, alpha)
        self.assertEqual(rxn.reactants, A + B)
        self.assertEqual(rxn.products, C)
        self.assertEqual(rxn.aux, alpha)

        rxn = MassActionReaction(A + B, C, 1.0)
        self.assertEqual(rxn.reactants, A + B)
        self.assertEqual(rxn.products, C)
        self.assertEqual(rxn.aux, TensorLiteral(1.0))

        rxn = MassActionReaction(Complex({}), C, TensorLiteral(1.0))
        self.assertEqual(rxn.reactants, Complex({}))
        self.assertEqual(rxn.products, C)
        self.assertEqual(rxn.aux, TensorLiteral(1.0))

        rxn = MassActionReaction(A + B, Complex({}), alpha)
        self.assertEqual(rxn.reactants, A + B)
        self.assertEqual(rxn.products, Complex({}))
        self.assertEqual(rxn.aux, alpha)

    def test_frozen(self):
        A, B, C = many_species("A, B, C")
        alpha = RateConstant("alpha")
        beta = RateConstant("beta")

        rxn = MassActionReaction(A + B, C, alpha)

        with self.assertRaises(FrozenInstanceError):
            rxn.reactants = A
        with self.assertRaises(FrozenInstanceError):
            rxn.products = B
        with self.assertRaises(FrozenInstanceError):
            rxn.aux = beta
        with self.assertRaises(FrozenInstanceError):
            rxn.rate_expr = beta

    def test_init_validation(self):
        A, B, C = many_species("A, B, C")
        alpha = RateConstant("alpha")
        RateConstant("beta")

        with self.assertRaises(TypeError):
            MassActionReaction(alpha, C, alpha)

        with self.assertRaises(ValueError):
            MassActionReaction(B, C, A)

        with self.assertRaises(ValueError):
            MassActionReaction(B, C, None)

    def test_flux(self):
        A, B, C = many_species("A, B, C")
        alpha = RateConstant("alpha")
        rxn = MassActionReaction(A + B, C, alpha)

        state = {
            A: jnp.array(2.0),
            B: jnp.array(3.0),
            C: jnp.array(4.0),
        }
        non_state = {
            alpha: jnp.array(1.1),
        }

        output = rxn.flux()(state, non_state)

        target_flux = {
            A: -jnp.array(1.1) * state[A] * state[B],
            B: -jnp.array(1.1) * state[A] * state[B],
            C: jnp.array(1.1) * state[A] * state[B],
        }
        self.assertTrue(dict_allclose(output, target_flux))


class TestFastReactionsToUpdateF(unittest.TestCase):
    def test_annihilation(self):
        A, B = many_species("A, B")
        i = many_index_symbols("i")

        fast_rxn = FastReaction(A[i] + B[i], 0)

        state = {A: jnp.array([1.0, 2.0, 1.5]), B: jnp.array([2.0, 0.0, 1.5])}

        output = fast_rxns_to_update_f([fast_rxn])(state)
        target_output = {
            A: jnp.array([-1.0, -2.0, -1.5]),
            B: jnp.array([-2.0, 0.0, -1.5]),
        }
        print(output)
        self.assertTrue(dict_allclose(output, target_output))
