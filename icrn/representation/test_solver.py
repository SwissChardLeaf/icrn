import unittest
from ..representation.symbols import many_species, many_rate_constants, many_index_symbols
from ..representation.reactions import MassActionReaction
from ..utils.dict_utils import dict_allclose
from ..representation.solver import solve_well_mixed

from jax import numpy as jnp

class TestSolveWellMixed(unittest.TestCase):
    def test_exponential_decay(self):
        A = many_species("A")
        k = many_rate_constants("k")

        rxns = [
            MassActionReaction(A, 0, k),
        ]

        conc_vals = {A: jnp.array(1.0)}
        rate_constant_vals = {k: jnp.array(1.0)}
        times = jnp.array([0, jnp.log(2), 1.0])
        dt = 0.005
        key = None
        checkpoint_length = None
        reaction_solver = "RK4"
        mode = None

        result = solve_well_mixed(rxns, conc_vals, rate_constant_vals, times, dt, key, checkpoint_length, reaction_solver, mode)
        target = {A: jnp.array([1.0, 0.5, jnp.exp(-1.0)])}

        self.assertTrue(dict_allclose(result, target))

    def test_two_decay(self):
        A = many_species("A")
        k = many_rate_constants("k")
        i = many_index_symbols("i")

        rxns = [
            MassActionReaction(A[i], 0, k[i]),
        ]

        conc_vals = {A: jnp.array([1.0, 2.0])}
        rate_constant_vals = {k: jnp.array([1.0, 2.0])}
        times = jnp.array([0, jnp.log(2), 1.0])
        dt = 0.01
        key = None
        checkpoint_length = None
        reaction_solver = "RK4"
        mode = None

        result = solve_well_mixed(rxns, conc_vals, rate_constant_vals, times, dt, key, checkpoint_length, reaction_solver, mode)
        target = {
            A: jnp.array([
                [1.0, 2.0], # initial conditions
                [0.50000554, 0.50315726], # half life
                [0.3667276 , 0.27067068]  # e^-1, 2e^-2
            ])
        }

        self.assertTrue(dict_allclose(result, target))

    def test_steady_state(self):
        A, B = many_species("A, B")
        k1, k2 = many_rate_constants("k1, k2")
        i = many_index_symbols("i")

        rxns = [
            MassActionReaction(A[i], 0, 1.),
            MassActionReaction(B[i], 0, 1.),
            MassActionReaction(0, A[i], k1[i]),
            MassActionReaction(0, B[i], k2[i]),
        ]
        
        conc_vals = {
            A: jnp.array([1.0, 1.0, 1.0]),
            B: jnp.array([1.0, 1.0, 1.0])
        }

        rate_constant_vals = {
            k1: jnp.array([1.0, 2.0, 3.0]), 
            k2: jnp.array([1.5, 3.5, 4.5])
        }
        
        times = jnp.array([0, 5, 10.])
        dt = 0.01
        key = None
        checkpoint_length = None
        reaction_solver = "RK4"
        mode = None

        result = solve_well_mixed(rxns, conc_vals, rate_constant_vals, times, dt, key, checkpoint_length, reaction_solver, mode)
        target = {
            A: jnp.array([1.0, 2.0, 3.0]),
            B: jnp.array([1.5, 3.0, 4.5])
        }

        print(result)
        # self.assertTrue(dict_allclose(result, target))



class TestSolveReactionsDiffusion(unittest.TestCase):
    def setUp(self):
        U, V = many_species("U, V")
        F, k = many_rate_constants("F, k")

        self.gs_rxns = [
            MassActionReaction(U + 2 * V, 3 * V, 1),
            MassActionReaction(V, 0, F + k),
            MassActionReaction(0, U, F),
            MassActionReaction(U, 0, F),
        ]
    def test_gray_scott(self):
        U, V = many_species("U, V")
        F, k = many_rate_constants("F, k")

        rxns = [
            MassActionReaction(U + 2 * V, 3 * V, 1),
            MassActionReaction(V, 0, F + k),
            MassActionReaction(0, U, F),
            MassActionReaction(U, 0, F),
        ]

        conc_vals = {U: jnp.array([1.0]), V: jnp.array([2.0])}