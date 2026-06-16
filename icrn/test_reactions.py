import unittest
from dataclasses import FrozenInstanceError

import jax.numpy as jnp

from .reactions import (
    AbstractReaction,
    FastReaction,
    MassActionReaction,
    _matching_shapes,
    fast_rxns_to_update_f,
    rxns_to_dynamics_f,
    rxns_to_mpe_dynamics_f,
    rxns_to_pd_dynamics_f,
)
from .solver import solve_well_mixed
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
    class MichaelisMentenReaction(AbstractReaction):
        """Single-substrate enzyme reaction S -> P with Michaelis-Menten
        kinetics.

        The enzyme E appears in the rate law but is not consumed.
        """

        def flux(self):
            sub, prod, (enz, k_cat_expr, km_expr) = (
                self.reactants,
                self.products,
                self.aux,
            )

            def f(state, non_state):
                data = non_state | state
                s = sub.eval(data)
                e = enz.eval(data)
                kcat = k_cat_expr.eval(data)
                km = km_expr.eval(data)

                rate = kcat * e * s / (km + s)
                return {sub: -rate, prod: rate}

            return f

        def flux_pd(self):
            sub, prod, (enz, k_cat_expr, km_expr) = (
                self.reactants,
                self.products,
                self.aux,
            )

            def f(state, non_state):
                data = non_state | state
                s = sub.eval(data)
                e = enz.eval(data)
                kcat = k_cat_expr.eval(data)
                km = km_expr.eval(data)

                rate = kcat * e * s / (km + s)

                production = {sub: 0.0, prod: rate}
                destruction = {sub: rate, prod: 0.0}

                return production, destruction

            return f

        def flux_pairs(self, split="uniform"):
            sub, prod, (enz, k_cat_expr, km_expr) = (
                self.reactants,
                self.products,
                self.aux,
            )

            def f(state, non_state):
                data = non_state | state
                s = sub.eval(data)
                e = enz.eval(data)
                kcat = k_cat_expr.eval(data)
                km = km_expr.eval(data)

                rate = kcat * e * s / (km + s)

                destruction = {sub: rate}
                pairs = {(prod, sub): rate}
                explicit = {}

                return destruction, pairs, explicit

            return f

    def setUp(self):
        self.S, self.P, self.E = many_species("S, P, E")
        self.k_cat, self.K_m = many_rate_constants("k_cat, K_m")
        self.mm_rxn = self.MichaelisMentenReaction(
            self.S, self.P, (self.E, self.k_cat, self.K_m)
        )
        self.state = {
            self.S: jnp.array(1.0),
            self.P: jnp.array(0.0),
            self.E: jnp.array(0.1),
        }
        self.non_state = {
            self.k_cat: jnp.array(2.0),
            self.K_m: jnp.array(0.3),
        }

    def test_flux(self):
        output = self.mm_rxn.flux()(self.state, self.non_state)
        # v = k_cat * [E] * [S] / (K_m + [S]) = 2 * 0.1 * 1 / (0.3 + 1)
        # = 0.2/1.3
        rate = jnp.array(0.2 / 1.3)
        target = {self.S: -rate, self.P: rate}
        self.assertTrue(dict_allclose(output, target))

    def test_flux_pd_matches_net(self):
        net = rxns_to_dynamics_f([self.mm_rxn])(self.state, self.non_state)
        production, destruction = rxns_to_pd_dynamics_f([self.mm_rxn])(
            self.state, self.non_state
        )

        for s in self.state:
            self.assertTrue(
                jnp.allclose(production[s] - destruction[s], net[s])
            )

    def test_flux_pairs_matches_net(self):
        net = rxns_to_dynamics_f([self.mm_rxn])(self.state, self.non_state)
        destruction, pairs, explicit = rxns_to_mpe_dynamics_f([self.mm_rxn])(
            self.state, self.non_state
        )

        mpe_net = {s: explicit.get(s, 0.0) for s in self.state}
        for s, v in destruction.items():
            mpe_net[s] -= v
        for (prod, _react), v in pairs.items():
            mpe_net[prod] += v

        for s in self.state:
            self.assertTrue(jnp.allclose(mpe_net[s], net[s]))

    def test_solve_well_mixed(self):
        times = jnp.linspace(0.0, 5.0, 101)
        traj = solve_well_mixed(
            [self.mm_rxn],
            conc_vals=self.state,
            rate_constant_vals=self.non_state,
            times=times,
            dt=0.01,
            reaction_solver="RK4",
        )

        self.assertTrue(jnp.allclose(traj[self.E], self.state[self.E]))
        self.assertTrue(
            jnp.allclose(traj[self.S] + traj[self.P], self.state[self.S])
        )
        self.assertTrue(jnp.allclose(traj[self.S][-1], jnp.array(0.33136249)))
        self.assertTrue(jnp.allclose(traj[self.P][-1], jnp.array(0.66863739)))

    def test_mixed_with_mass_action_enzyme_decay(self):
        times = jnp.linspace(0.0, 5.0, 101)
        ma_rxn = MassActionReaction(self.E, 0, 0.5)

        traj = solve_well_mixed(
            [self.mm_rxn, ma_rxn],
            conc_vals=self.state,
            rate_constant_vals=self.non_state,
            times=times,
            dt=0.01,
            reaction_solver="RK4",
        )

        self.assertTrue(
            jnp.allclose(traj[self.S] + traj[self.P], self.state[self.S])
        )
        self.assertTrue(jnp.allclose(traj[self.E][-1], jnp.array(0.00820849)))
        self.assertTrue(jnp.allclose(traj[self.S][-1], jnp.array(0.72804976)))
        self.assertTrue(jnp.allclose(traj[self.P][-1], jnp.array(0.27195022)))


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


class TestRxnsToPDDynamicsF(unittest.TestCase):
    def test_pd_values(self):
        A, B = many_species("A, B")
        k = RateConstant("k")
        rxns = [MassActionReaction(A, B, k)]

        state = {A: jnp.array(2.0), B: jnp.array(0.0)}
        non_state = {k: jnp.array(3.0)}

        production, destruction = rxns_to_pd_dynamics_f(rxns)(state, non_state)

        self.assertTrue(jnp.allclose(production[A], 0.0))
        self.assertTrue(jnp.allclose(destruction[A], 6.0))
        self.assertTrue(jnp.allclose(production[B], 6.0))
        self.assertTrue(jnp.allclose(destruction[B], 0.0))

    def test_pd_matches_net_dynamics(self):
        A, B, C = many_species("A, B, C")
        k1, k2 = many_rate_constants("k1, k2")
        rxns = [
            MassActionReaction(A, B, k1),
            MassActionReaction(B, C, k2),
        ]

        state = {A: jnp.array(2.0), B: jnp.array(3.0), C: jnp.array(1.0)}
        non_state = {k1: jnp.array(0.5), k2: jnp.array(0.7)}

        net = rxns_to_dynamics_f(rxns)(state, non_state)
        production, destruction = rxns_to_pd_dynamics_f(rxns)(state, non_state)

        for s in state:
            self.assertTrue(
                jnp.allclose(production[s] - destruction[s], net[s])
            )


class TestFastReactionsToUpdateF(unittest.TestCase):
    def test_annihilation(self):
        A, B = many_species("A, B")
        i = many_index_symbols("i")

        fast_rxn = FastReaction(A[i] + B[i], 0)

        state = {A: jnp.array([1.0, 2.0, 1.5]), B: jnp.array([2.0, 0.0, 1.5])}

        output = fast_rxns_to_update_f([fast_rxn])(state)
        target_output = {
            A: jnp.array([0.0, 2.0, 0.0]),
            B: jnp.array([1.0, 0.0, 0.0]),
        }
        self.assertTrue(dict_allclose(output, target_output))
