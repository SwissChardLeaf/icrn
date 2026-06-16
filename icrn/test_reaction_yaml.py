import tempfile
import unittest
from pathlib import Path

import jax.numpy as jnp

from .reactions import AbstractReaction, FastReaction, MassActionReaction
from .symbols import many_index_symbols, many_rate_constants, many_species
from .utils.dict_utils import dict_allclose


class TestReactionYaml(unittest.TestCase):
    def test_mass_action_roundtrip(self):
        A, B, C = many_species("A, B, C")
        k_f, k_r = many_rate_constants("k_f, k_r")
        rxns = [
            MassActionReaction(A + B, C, k_f),
            MassActionReaction(C, A + B, k_r),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "network.yaml"
            AbstractReaction.save_yaml(path, rxns)
            loaded, _symbols = AbstractReaction.load_yaml(path)

        self.assertEqual(len(loaded), 2)
        self.assertIsInstance(loaded[0], MassActionReaction)
        self.assertEqual(repr(loaded[0].reactants), repr(rxns[0].reactants))
        self.assertEqual(repr(loaded[0].products), repr(rxns[0].products))
        self.assertEqual(repr(loaded[0].aux), repr(rxns[0].aux))

    def test_indexed_mass_action_roundtrip(self):
        M, D = many_species("M, D")
        K1, K2 = many_rate_constants("K1, K2")
        i, j = many_index_symbols("i, j", 5)
        rxns = [
            MassActionReaction(M[i] + M[j], D[i, j], K1[i, j]),
            MassActionReaction(D[i, j], M[i] + M[j], K2[i, j]),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "network.yaml"
            AbstractReaction.save_yaml(path, rxns)
            loaded, _symbols = AbstractReaction.load_yaml(path)

        state = {M: jnp.ones(5), D: jnp.zeros((5, 5))}
        non_state = {K1: jnp.ones((5, 5)) * 0.1, K2: jnp.ones((5, 5)) * 0.05}

        for orig, new in zip(rxns, loaded):
            self.assertTrue(
                dict_allclose(
                    orig.flux()(state, non_state),
                    new.flux()(state, non_state),
                )
            )

    def test_custom_reaction_registration(self):
        class MichaelisMentenReaction(AbstractReaction):
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
                    return {sub: 0.0, prod: rate}, {sub: rate, prod: 0.0}

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
                    return {sub: rate}, {(prod, sub): rate}, {}

                return f

            def to_dict(self):
                enz, k_cat_expr, km_expr = self.aux
                return {
                    "type": "michaelis_menten",
                    "substrate": self.reactants.label,
                    "product": self.products.label,
                    "enzyme": enz.label,
                    "k_cat": k_cat_expr.label,
                    "K_m": km_expr.label,
                }

            @classmethod
            def from_dict(cls, entry, symbols):
                return cls(
                    symbols.species[entry["substrate"]],
                    symbols.species[entry["product"]],
                    (
                        symbols.species[entry["enzyme"]],
                        symbols.rate_constants[entry["k_cat"]],
                        symbols.rate_constants[entry["K_m"]],
                    ),
                )

            def collect_yaml_symbols(self, table):
                enz, k_cat_expr, km_expr = self.aux
                table.species[enz.label] = enz
                table.rate_constants[k_cat_expr.label] = k_cat_expr[()]
                table.rate_constants[km_expr.label] = km_expr[()]

        AbstractReaction.register_yaml_type(
            "michaelis_menten", MichaelisMentenReaction
        )

        S, P, E = many_species("S, P, E")
        k_cat, K_m = many_rate_constants("k_cat, K_m")
        rxn = MichaelisMentenReaction(S, P, (E, k_cat, K_m))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "network.yaml"
            AbstractReaction.save_yaml(path, [rxn])
            loaded, _symbols = AbstractReaction.load_yaml(path)

        self.assertEqual(len(loaded), 1)
        self.assertIsInstance(loaded[0], MichaelisMentenReaction)

    def test_fast_reaction_roundtrip(self):
        A, B = many_species("A, B")
        i = many_index_symbols("i")
        rxns = [FastReaction(A[i] + B[i], 0)]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "network.yaml"
            AbstractReaction.save_yaml(path, rxns)
            loaded, _symbols = AbstractReaction.load_yaml(path)

        self.assertEqual(len(loaded), 1)
        self.assertIsInstance(loaded[0], FastReaction)
        self.assertEqual(repr(loaded[0].reactants), repr(rxns[0].reactants))
        self.assertEqual(repr(loaded[0].products), repr(rxns[0].products))

    def test_mixed_reaction_types(self):
        A, B, C = many_species("A, B, C")
        k = many_rate_constants("k")
        i = many_index_symbols("i")
        rxns = [
            MassActionReaction(A, B, k),
            FastReaction(A[i] + B[i], 0),
            MassActionReaction(B, C, k),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "network.yaml"
            AbstractReaction.save_yaml(path, rxns)
            loaded, _symbols = AbstractReaction.load_yaml(path)

        self.assertEqual(len(loaded), 3)
        self.assertIsInstance(loaded[0], MassActionReaction)
        self.assertIsInstance(loaded[1], FastReaction)
        self.assertIsInstance(loaded[2], MassActionReaction)
