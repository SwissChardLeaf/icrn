import unittest
from collections import Counter

from .mass_action import (
    _get_tensors,
    _get_base_einsum_str,
    _get_diff_dict,
    _get_diff_and_einsum_strs,
    _setup_einsums,
    mass_action_flux_f,
    _get_all_index_symbols,
    _product_index_symbols,
)
from .symbols import (
    Complex,
    IndexSymbol,
    many_index_symbols,
    many_species,
    many_rate_constants,
    RateConstant,
    TensorLiteral,
)

from .reactions import MassActionReaction
from jax import numpy as jnp
import jax.random as jax_random


class TestMassActionFluxF(unittest.TestCase):
    def setUp(self):
        A, B, C, D = many_species("A, B, C, D")
        alpha, beta = many_rate_constants("alpha, beta")
        i, j = many_index_symbols("i, j")
        k = many_index_symbols("k", 3)
        l, m = many_index_symbols("l, m", 7)

        self.good_rxns = [
            MassActionReaction(A + B, C, alpha),
            MassActionReaction(A[i] + B[j], C[i, j], alpha[i]),
            MassActionReaction(0, C, alpha),
            MassActionReaction(A[i], 0, 1.0),
            MassActionReaction(A[i, l] + B[j], A[j, m] + B[l], 1.0),
            MassActionReaction(A[i], 0, 0),
            MassActionReaction(A[i] + A[j], B[i,j], 1.0),
        ]

        self.bad_rxns = [
            MassActionReaction(A[i], A[j], 2.0),
            MassActionReaction(A[i], B[j], alpha[k]),
            MassActionReaction(0, A[i], alpha[k]),
        ]

    def test_get_diff_dict(self):
        A, B, C, D = many_species("A, B, C, D")
        i, j, k = many_index_symbols("i, j, k")
        l, m = many_index_symbols("l, m", 7)

        def _get_diff_dict_from_rxn(rxn: MassActionReaction):
            return _get_diff_dict(rxn.reactants, rxn.products)

        self.assertEqual(
            _get_diff_dict_from_rxn(self.good_rxns[0]), {A: -1, B: -1, C: 1}
        )
        self.assertEqual(
            _get_diff_dict_from_rxn(self.good_rxns[1]), {A[i]: -1, B[j]: -1, C[i, j]: 1}
        )
        self.assertEqual(_get_diff_dict_from_rxn(self.good_rxns[2]), {C: 1})
        self.assertEqual(_get_diff_dict_from_rxn(self.good_rxns[3]), {A[i]: -1})
        self.assertEqual(
            _get_diff_dict_from_rxn(self.good_rxns[4]),
            {A[i, l]: -1, B[j]: -1, A[j, m]: 1, B[l]: 1},
        )
        self.assertEqual(_get_diff_dict_from_rxn(self.good_rxns[5]), {A[i]: -1})

        self.assertEqual(_get_diff_dict_from_rxn(self.bad_rxns[0]), {A[i]: -1, A[j]: 1})
        self.assertEqual(_get_diff_dict_from_rxn(self.bad_rxns[1]), {A[i]: -1, B[j]: 1})
        self.assertEqual(_get_diff_dict_from_rxn(self.bad_rxns[2]), {A[i]: 1})

    def test_get_all_index_symbols(self):
        i, j = many_index_symbols("i, j")
        k = many_index_symbols("k", 3)
        l, m = many_index_symbols("l, m", 7)

        def _get_all_index_symbols_from_rxn(rxn: MassActionReaction, target_set):
            try:
                self.assertEqual(
                    _get_all_index_symbols(rxn.reactants, rxn.products, rxn.rate_expr),
                    target_set,
                )
            except Exception as e:
                raise AssertionError(
                    f"Error getting all index symbols for {rxn}"
                ) from e

        _get_all_index_symbols_from_rxn(self.good_rxns[0], set())
        _get_all_index_symbols_from_rxn(self.good_rxns[1], {i, j})
        _get_all_index_symbols_from_rxn(self.good_rxns[2], set())
        _get_all_index_symbols_from_rxn(self.good_rxns[3], {i})
        _get_all_index_symbols_from_rxn(self.good_rxns[4], {i, j, m, l})
        _get_all_index_symbols_from_rxn(self.good_rxns[5], {i})

        _get_all_index_symbols_from_rxn(self.bad_rxns[0], {i, j})
        _get_all_index_symbols_from_rxn(self.bad_rxns[1], {i, j, k})
        _get_all_index_symbols_from_rxn(self.bad_rxns[2], {i, k})

    def test_product_index_symbols(self):
        i, j = many_index_symbols("i, j")
        k = many_index_symbols("k", 3)
        l, m = many_index_symbols("l, m", 7)

        def _product_index_symbols_from_rxn(rxn: MassActionReaction, target_set):
            self.assertEqual(
                _product_index_symbols(rxn.reactants, rxn.products, rxn.rate_expr),
                target_set,
            )

        _product_index_symbols_from_rxn(self.good_rxns[0], set())
        _product_index_symbols_from_rxn(self.good_rxns[1], set())
        _product_index_symbols_from_rxn(self.good_rxns[2], set())
        _product_index_symbols_from_rxn(self.good_rxns[3], set())
        _product_index_symbols_from_rxn(self.good_rxns[4], {m})
        _product_index_symbols_from_rxn(self.good_rxns[5], set())

        _product_index_symbols_from_rxn(self.bad_rxns[0], {j})
        _product_index_symbols_from_rxn(self.bad_rxns[1], {j})
        _product_index_symbols_from_rxn(self.bad_rxns[2], {i})

    def test_base_einsum_str(self):
        def _get_base_einsum_str_from_rxn(rxn: MassActionReaction, target_str):
            try:
                product_index_symbols = _product_index_symbols(
                    rxn.reactants, rxn.products, rxn.rate_expr
                )
                self.assertEqual(
                    _get_base_einsum_str(
                        rxn.reactants, rxn.rate_expr, product_index_symbols
                    ),
                    target_str,
                )
            except Exception as e:
                raise AssertionError(f"Error getting base einsum str for {rxn}") from e

        _get_base_einsum_str_from_rxn(self.good_rxns[0], ",,->")
        _get_base_einsum_str_from_rxn(self.good_rxns[1], "i,i,j->")
        _get_base_einsum_str_from_rxn(self.good_rxns[2], "->")
        _get_base_einsum_str_from_rxn(self.good_rxns[3], ",i->")
        _get_base_einsum_str_from_rxn(self.good_rxns[4], ",il,j,m->")
        _get_base_einsum_str_from_rxn(self.good_rxns[5], ",i->")

        _get_base_einsum_str_from_rxn(self.bad_rxns[0], ",i,j->")
        _get_base_einsum_str_from_rxn(self.bad_rxns[1], "k,i,j->")
        _get_base_einsum_str_from_rxn(self.bad_rxns[2], "k,i->")

    def test_setup_einsums(self):
        A, B, C, D = many_species("A, B, C, D")

        def _setup_einsums_from_rxn(rxn: MassActionReaction, target_dict):
            try:
                output = _setup_einsums(rxn.reactants, rxn.products, rxn.rate_expr)

                self.assertEqual(output.keys(), target_dict.keys())

                for s, val in target_dict.items():
                    self.assertEqual(Counter(output[s]), Counter(val))

            except Exception as e:
                raise AssertionError(f"Error setting up einsums for {rxn}") from e

        _setup_einsums_from_rxn(
            self.good_rxns[0],
            {
                A: [(-1, ",,->")],
                B: [(-1, ",,->")],
                C: [(1, ",,->")],
            },
        )
        _setup_einsums_from_rxn(
            self.good_rxns[1],
            {
                A: [(-1, "i,i,j->i")],
                B: [(-1, "i,i,j->j")],
                C: [(1, "i,i,j->ij")],
            },
        )
        _setup_einsums_from_rxn(
            self.good_rxns[2],
            {
                C: [(1, "->")],
            },
        )
        _setup_einsums_from_rxn(
            self.good_rxns[3],
            {
                A: [(-1, ",i->i")],
            },
        )
        _setup_einsums_from_rxn(
            self.good_rxns[4],
            {
                A: [(-1, ",il,j,m->il"), (1, ",il,j,m->jm")],
                B: [(-1, ",il,j,m->j"), (1, ",il,j,m->l")],
            },
        )
        _setup_einsums_from_rxn(
            self.good_rxns[5],
            {
                A: [(-1, ",i->i")],
            },
        )
        _setup_einsums_from_rxn(
            self.bad_rxns[0], {A: [(-1, ",i,j->i"), (1, ",i,j->j")]}
        )
        _setup_einsums_from_rxn(
            self.bad_rxns[1],
            {
                A: [(-1, "k,i,j->i")],
                B: [(1, "k,i,j->j")],
            },
        )
        _setup_einsums_from_rxn(
            self.bad_rxns[2],
            {
                A: [(1, "k,i->i")],
            },
        )

    def test_get_tensors(self):
        A, B, C, D = many_species("A, B, C, D")
        alpha, beta = many_rate_constants("alpha, beta")

        def _get_tensors_from_rxn(
            rxn: MassActionReaction, state, non_state, target_tensors
        ):
            try:
                product_index_symbols = _product_index_symbols(
                    rxn.reactants, rxn.products, rxn.rate_expr
                )
                output = _get_tensors(
                    rxn.reactants, rxn.rate_expr, product_index_symbols
                )

                output_tensor = output(state, non_state)

                if len(output_tensor) != len(target_tensors):
                    raise AssertionError(
                        f"Error getting tensors for {rxn}, output tensor has length {len(output_tensor)}, but target tensors have length {len(target_tensors)}"
                    )

                for i in range(len(output_tensor)):
                    self.assertTrue(
                        jnp.all(jnp.equal(output_tensor[i], target_tensors[i]))
                    )
            except ValueError as e:
                raise
            except Exception as e:
                raise AssertionError(f"Error getting tensors for {rxn}") from e

        _get_tensors_from_rxn(
            self.good_rxns[0],
            {A: jnp.array(1.0), B: jnp.array(2.0)},
            {
                alpha: jnp.array(7.0),
            },
            (jnp.array(7.0), jnp.array(1.0), jnp.array(2.0)),
        )

        _get_tensors_from_rxn(
            self.good_rxns[1],
            {A: 1.0 * jnp.ones((10,)), B: 2.0 * jnp.ones((10,))},
            {
                alpha: 7.0 * jnp.ones((10,)),
            },
            (7.0 * jnp.ones((10,)), 1.0 * jnp.ones((10,)), 2.0 * jnp.ones((10,))),
        )

        _get_tensors_from_rxn(
            self.good_rxns[2],
            {},
            {
                alpha: jnp.array(7.0),
            },
            (jnp.array(7.0),),
        )

        _get_tensors_from_rxn(
            self.good_rxns[3],
            {
                A: 1.0 * jnp.ones((10,)),
            },
            {},
            (jnp.array(1.0), 1.0 * jnp.ones((10,))),
        )

        _get_tensors_from_rxn(
            self.good_rxns[4],
            {
                A: 1.0 * jnp.ones((10, 7)),
                B: 2.0 * jnp.ones((7,)),
            },
            {},
            (
                jnp.array(1.0),
                1.0 * jnp.ones((10, 7)),
                2.0 * jnp.ones((7,)),
                jnp.ones((7,)),
            ),
        )

        _get_tensors_from_rxn(
            self.good_rxns[5],
            {
                A: 1.0 * jnp.ones((10,)),
            },
            {},
            (jnp.array(0.0), 1.0 * jnp.ones((10,))),
        )

        with self.assertRaises(ValueError):
            _get_tensors_from_rxn(
                self.bad_rxns[0],
                {
                    A: 1.0 * jnp.ones((10,)),
                },
                {},
                (jnp.array(2.0), 1.0 * jnp.ones((10,))),
            )

        with self.assertRaises(ValueError):
            _get_tensors_from_rxn(
                self.bad_rxns[1],
                {
                    A: 1.0 * jnp.ones((10,)),
                },
                {
                    alpha: 7.0 * jnp.ones((10,)),
                },
                (7.0 * jnp.ones((10,)), 1.0 * jnp.ones((10,))),
            )

        with self.assertRaises(ValueError):
            _get_tensors_from_rxn(
                self.bad_rxns[2],
                {},
                {
                    alpha: 7.0 * jnp.ones((10,)),
                },
                (7.0 * jnp.ones((10,)),),
            )

    @unittest.skip("Skipping mass action flux f test for now")
    def test_mass_action_flux_f(self):
        A, B, C, D = many_species("A, B, C, D")
        alpha, beta = many_rate_constants("alpha, beta")

        def _test_mass_action_flux_f_from_rxn(
            rxn: MassActionReaction, state, non_state, target_flux
        ):
            try:
                output = mass_action_flux_f(rxn.reactants, rxn.products, rxn.rate_expr)
                output_flux = output(state, non_state)

                self.assertEqual(output_flux.keys(), target_flux.keys())

                for s, val in target_flux.items():
                    self.assertTrue(jnp.all(jnp.allclose(output_flux[s], val)))

            except ValueError as e:
                raise
            except Exception as e:
                raise AssertionError(
                    f"Error testing mass action flux f for {rxn}"
                ) from e

        _test_mass_action_flux_f_from_rxn(
            self.good_rxns[0],
            state={A: jnp.array(2.0), B: jnp.array(3.0)},
            non_state={alpha: jnp.array(7.0)},
            target_flux={A: -jnp.array(42.0), B: -jnp.array(42.0), C: jnp.array(42.0)},
        )

        A_data = jax_random.uniform(jax_random.key(0), (5,))
        B_data = jax_random.uniform(jax_random.key(0), (6,))
        alpha_data = jax_random.uniform(jax_random.key(0), (5,))

        _test_mass_action_flux_f_from_rxn(
            self.good_rxns[1],
            state={
                A: A_data,
                B: B_data,
            },
            non_state={
                alpha: alpha_data,
            },
            target_flux={
                A: -jnp.einsum("i,i,j->i", alpha_data, A_data, B_data),
                B: -jnp.einsum("i,i,j->j", alpha_data, A_data, B_data),
                C: jnp.einsum("i,i,j->ij", alpha_data, A_data, B_data),
            },
        )

        alpha_data = jax_random.uniform(jax_random.key(0), ())

        _test_mass_action_flux_f_from_rxn(
            self.good_rxns[2],
            state={},
            non_state={
                alpha: alpha_data,
            },
            target_flux={C: alpha_data},
        )

        A_data = jax_random.uniform(jax_random.key(0), (5,))

        _test_mass_action_flux_f_from_rxn(
            self.good_rxns[3],
            state={
                A: A_data,
            },
            non_state={},
            target_flux={A: -A_data},
        )

        A_data = jax_random.uniform(jax_random.key(0), (7, 7))
        B_data = jax_random.uniform(jax_random.key(0), (7,))

        _test_mass_action_flux_f_from_rxn(
            self.good_rxns[4],
            state={
                A: A_data,
                B: B_data,
            },
            non_state={},
            target_flux={
                A: -jnp.einsum(
                    ",il,j,m->il", jnp.array(1.0), A_data, B_data, jnp.ones((7,))
                )
                + jnp.einsum(
                    ",il,j,m->jm", jnp.array(1.0), A_data, B_data, jnp.ones((7,))
                ),
                B: -jnp.einsum(
                    ",il,j,m->j", jnp.array(1.0), A_data, B_data, jnp.ones((7,))
                )
                + jnp.einsum(
                    ",il,j,m->l", jnp.array(1.0), A_data, B_data, jnp.ones((7,))
                ),
            },
        )
