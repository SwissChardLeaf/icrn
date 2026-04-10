import unittest
from ._fast_flux import (
    _fast_flux_f, _standard_reactants_and_indexing, _build_base_einsum_str, _einsum_prep, _get_reactant_unit
)
from collections import Counter
from ..representation.reactions import FastReaction
from ..representation.symbols import Species, Complex, many_species, many_index_symbols

import jax.numpy as jnp

class TestFastFluxF(unittest.TestCase):
    def setUp(self):
        A, B, C, D, E = many_species("A, B, C, D, E")
        i, j = many_index_symbols("i, j")

        self.rxns = [
            FastReaction(A + B, C),
            FastReaction(2*A + 2*B, A + 3*B),
            FastReaction(A + B, 0),
            FastReaction(A[i] + B[i], C[i]),
            FastReaction(2*A[i] + 3*B[i], C[i] + 2*D[i]),
            FastReaction(A, 0),
            FastReaction(A[i,j] + 2*B[i,j], 3*C[i,j]),
            FastReaction(A[i,j] + B[i,j], C[j,i] + D[j] + E),
            FastReaction(A[j,i], A[i,j]),
            FastReaction(A[i,j] + 3*B[i,j], A[j,i] + 2*B[i,j]),
            FastReaction(A + B + C, D),
        ]

    def test_standard_reactants_and_indexing(self):
        A, B, C, D, E = many_species("A, B, C, D, E")
        i, j = many_index_symbols("i, j")

        def _test_standard_reactants_and_indexing_from_rxn(rxn: FastReaction, target_standard_reactants, target_standard_indexing):
            try:
                standard_reactants, standard_indexing = _standard_reactants_and_indexing(rxn.reactants)
                self.assertEqual(standard_reactants, target_standard_reactants)
                self.assertEqual(standard_indexing, target_standard_indexing)
            except Exception as e:
                raise AssertionError(f"Error testing standard reactants and indexing for {rxn}") from e
        
        _test_standard_reactants_and_indexing_from_rxn(self.rxns[0], (A, B), ())
        _test_standard_reactants_and_indexing_from_rxn(self.rxns[1], (A, B), ())
        _test_standard_reactants_and_indexing_from_rxn(self.rxns[2], (A, B), ())
        _test_standard_reactants_and_indexing_from_rxn(self.rxns[3], (A[i], B[i]), (i,))
        _test_standard_reactants_and_indexing_from_rxn(self.rxns[4], (A[i], B[i]), (i,))
        _test_standard_reactants_and_indexing_from_rxn(self.rxns[5], (A,), ())
        _test_standard_reactants_and_indexing_from_rxn(self.rxns[6], (A[i,j], B[i,j]), (i, j))
        _test_standard_reactants_and_indexing_from_rxn(self.rxns[7], (A[i,j], B[i,j]), (i, j))
        _test_standard_reactants_and_indexing_from_rxn(self.rxns[8], (A[j,i],), (j, i))
        _test_standard_reactants_and_indexing_from_rxn(self.rxns[9], (A[i,j], B[i,j]), (i, j))
        _test_standard_reactants_and_indexing_from_rxn(self.rxns[10], (A, B, C), ())


    def test_build_base_einsum_str(self):
        def _test_base_einsum_str_from_rxn(rxn: FastReaction, target_str):
            try:
                standard_reactants, standard_indexing = _standard_reactants_and_indexing(rxn.reactants)
                self.assertEqual(_build_base_einsum_str(standard_indexing), target_str)
            except Exception as e:
                raise AssertionError(f"Error testing base einsum str for {rxn}") from e

        _test_base_einsum_str_from_rxn(self.rxns[0], "->")
        _test_base_einsum_str_from_rxn(self.rxns[1], "->")
        _test_base_einsum_str_from_rxn(self.rxns[2], "->")
        _test_base_einsum_str_from_rxn(self.rxns[3], "i->")
        _test_base_einsum_str_from_rxn(self.rxns[4], "i->")
        _test_base_einsum_str_from_rxn(self.rxns[5], "->")
        _test_base_einsum_str_from_rxn(self.rxns[6], "ij->")
        _test_base_einsum_str_from_rxn(self.rxns[7], "ij->")
        _test_base_einsum_str_from_rxn(self.rxns[8], "ji->")
        _test_base_einsum_str_from_rxn(self.rxns[9], "ij->")
        _test_base_einsum_str_from_rxn(self.rxns[10], "->")

    def test_einsum_prep(self):
        A, B, C, D, E = many_species("A, B, C, D, E")

        def _test_einsum_prep_from_rxn(rxn: FastReaction, target_dict):
            try:
                standard_reactants, standard_indexing = _standard_reactants_and_indexing(rxn.reactants)
                output = _einsum_prep(rxn.reactants, rxn.products, standard_indexing)
                self.assertEqual(output.keys(), target_dict.keys())
                for s, val in target_dict.items():
                    self.assertEqual(Counter(output[s]), Counter(val))
            except Exception as e:
                raise AssertionError(f"Error testing einsum prep for {rxn}") from e

        _test_einsum_prep_from_rxn(
            self.rxns[0], 
            {
                A: [(-1, "->")], 
                B: [(-1, "->")], 
                C: [(1, "->")]
            }
        )

        _test_einsum_prep_from_rxn(
            self.rxns[1], 
            {
                A: [(-1, "->")], 
                B: [(1, "->")]
            }
        )

        _test_einsum_prep_from_rxn(
            self.rxns[2], 
            {
                A: [(-1, "->")], 
                B: [(-1, "->")]
            }
        )

        _test_einsum_prep_from_rxn(
            self.rxns[3], 
            {
                A: [(-1, "i->i")], 
                B: [(-1, "i->i")], 
                C: [(1, "i->i")]
            }
        )

        _test_einsum_prep_from_rxn(
            self.rxns[4], 
            {
                A: [(-2, "i->i")], 
                B: [(-3, "i->i")], 
                C: [(1, "i->i")],
                D: [(2, "i->i")]
            }
        )

        _test_einsum_prep_from_rxn(
            self.rxns[5], 
            {
                A: [(-1, "->")],
            }
        )

        _test_einsum_prep_from_rxn(
            self.rxns[6], 
            {
                A: [(-1, "ij->ij")], 
                B: [(-2, "ij->ij")], 
                C: [(3, "ij->ij")]
            }
        )

        _test_einsum_prep_from_rxn(
            self.rxns[7], 
            {
                A: [(-1, "ij->ij")], 
                B: [(-1, "ij->ij")], 
                C: [(1, "ij->ji")],
                D: [(1, "ij->j")],
                E: [(1, "ij->")]
            }
        )

        _test_einsum_prep_from_rxn(
            self.rxns[8], 
            {
                A: [(-1, "ji->ji"), (1, "ji->ij")]
            }
        )

        _test_einsum_prep_from_rxn(
            self.rxns[9], 
            {
                A: [(-1, "ij->ij"), (1, "ij->ji")],
                B: [(-1, "ij->ij")]
            }
        )
        
        
    def test_get_reactant_unit_from_rxn(self):
        A, B, C, D, E = many_species("A, B, C, D, E")

        def _test_get_reactant_unit_from_rxn(rxn: FastReaction, state_data, target_unit):
            try:
                standard_reactants, standard_indexing = _standard_reactants_and_indexing(rxn.reactants)
                output = _get_reactant_unit(standard_reactants, rxn.reactants, state_data)
                self.assertTrue(jnp.all(jnp.allclose(output, target_unit)))
            except Exception as e:
                raise AssertionError(f"Error testing get reactant unit for {rxn}") from e

        _test_get_reactant_unit_from_rxn(
            self.rxns[0],
            {
                A: jnp.array(2),
                B: jnp.array(3)
            },
            jnp.array(2)
        )

        _test_get_reactant_unit_from_rxn(
            self.rxns[1],
            {
                A: jnp.array(2),
                B: jnp.array(3)
            },
            jnp.array(1)
        )

        _test_get_reactant_unit_from_rxn(
            self.rxns[2],
            {
                A: jnp.array(2),
                B: jnp.array(3)
            },
            jnp.array(2)
        )

        _test_get_reactant_unit_from_rxn(
            self.rxns[3],
            {
                A: jnp.array([0, 1, 2, 3, 4]),
                B: jnp.array([5, 4, 3, 2, 1])
            },
            jnp.array([0, 1, 2, 2, 1])
        )

        _test_get_reactant_unit_from_rxn(
            self.rxns[4],
            {
                A: jnp.array([0, 1, 2, 3, 4]),
                B: jnp.array([5, 4, 3, 2, 1])
            },
            jnp.array([0, 1/2, 1, 2/3, 1/3])
        )

        _test_get_reactant_unit_from_rxn(
            self.rxns[5],
            {
                A: jnp.array(2)
            },
            jnp.array(2)
        )

        _test_get_reactant_unit_from_rxn(
            self.rxns[6],
            {
                A: jnp.array([
                    [0, 1], 
                    [2, 3], 
                    [4, 5]
                ]),
                B: jnp.array([
                    [6, 5], 
                    [4, 3], 
                    [2, 1]
                ])
            },
            jnp.array([
                [0, 1], 
                [2, 3/2], 
                [1, 1/2]
            ]),
        )

        _test_get_reactant_unit_from_rxn(
            self.rxns[7],
            {
                A: jnp.array([
                    [0, 1], 
                    [2, 3], 
                    [4, 5]
                ]),
                B: jnp.array([
                    [6, 5], 
                    [4, 3], 
                    [2, 1]
                ])
            },
            jnp.array([
                [0, 1], 
                [2, 3], 
                [2, 1]
            ]),
        )

        _test_get_reactant_unit_from_rxn(
            self.rxns[8],
            {
                A: jnp.array([
                    [0, 1], 
                    [2, 3], 
                    [4, 5]
                ])
            },
            jnp.array([
                [0, 1], 
                [2, 3], 
                [4, 5]
            ]),
        )

        _test_get_reactant_unit_from_rxn(
            self.rxns[9],
            {
                A: jnp.array([
                    [0, 1], 
                    [2, 3], 
                    [4, 5]
                ]),
                B: jnp.array([
                    [6, 5], 
                    [4, 3], 
                    [2, 1]
                ])
            },
            jnp.array([
                [0, 1], 
                [4/3, 1], 
                [2/3, 1/3]
            ]),
        )

        _test_get_reactant_unit_from_rxn(
            self.rxns[10],
            {
                A: jnp.array(2),
                B: jnp.array(3),
                C: jnp.array(4)
            },
            jnp.array(2)
        )
        

    def test_fast_flux_f(self):
        A, B, C, D, E = many_species("A, B, C, D, E")

        def test_fast_flux_f_from_rxn(rxn: FastReaction, state_data, target_flux: dict):
            try:
                flux_f = rxn.flux()
                output = flux_f(state_data)

                self.assertEqual(output.keys(), target_flux.keys())
                for s, val in target_flux.items():
                    self.assertTrue(jnp.all(jnp.allclose(output[s], val)))
            except Exception as e:
                raise AssertionError(f"Error testing fast flux f for {rxn}") from e

        test_fast_flux_f_from_rxn(
            self.rxns[0],
            {
                A: jnp.array(2), 
                B: jnp.array(3)
            },
            {
                A: - jnp.array(2),
                B: - jnp.array(2),
                C: jnp.array(2)
            }
        )

        test_fast_flux_f_from_rxn(
            self.rxns[1],
            {
                A: jnp.array(10),
                B: jnp.array(7),
            },
            {
                A: - jnp.array(3.5),
                B: jnp.array(3.5),
            }
        )

        test_fast_flux_f_from_rxn(
            self.rxns[2],
            {
                A: jnp.array(2), 
                B: jnp.array(3)
            },
            {
                A: - jnp.array(2),
                B: - jnp.array(2)
            }
        )

        test_fast_flux_f_from_rxn(
            self.rxns[3],
            {
                A: jnp.array([0, 1, 2, 3, 4]), 
                B: jnp.array([5, 4, 3, 2, 1])
            },
            {
                A: - jnp.array([0, 1, 2, 2, 1]),
                B: - jnp.array([0, 1, 2, 2, 1]),
                C: jnp.array([0, 1, 2, 2, 1])
            }
        )

        test_fast_flux_f_from_rxn(
            self.rxns[4],
            {
                A: jnp.array([0, 1, 2, 3, 4]), 
                B: jnp.array([5, 4, 3, 2, 1])
            },
            {
                A: - jnp.array([0, 1, 2, 4/3, 2/3]),
                B: - jnp.array([0, 3/2, 3, 2, 1]),
                C: jnp.array([0, 1/2, 1, 2/3, 1/3]),
                D: jnp.array([0, 1, 2, 4/3, 2/3])
            }
        )
        
        test_fast_flux_f_from_rxn(
            self.rxns[5],
            {
                A: jnp.array(1)
            },
            {
                A: - jnp.array(1),
            }
        )

        test_fast_flux_f_from_rxn(
            self.rxns[6],
            {
                A: jnp.array([
                    [0, 1],
                    [2, 3],
                    [4, 5]
                ]),
                B: jnp.array([
                    [6, 5],
                    [5, 3],
                    [2, 1]
                ])
            },
            {
                A: - jnp.array([
                    [0, 1],
                    [2, 3/2],
                    [1, 1/2]
                ]),
                B: - jnp.array([
                    [0, 2],
                    [4, 3],
                    [2, 1]
                ]),
                C: jnp.array([
                    [0, 3],
                    [6, 9/2],
                    [3, 3/2]
                ])
            }
        )

        test_fast_flux_f_from_rxn(
            self.rxns[7],
            {
                A: jnp.array([
                    [0, 1],
                    [2, 3],
                    [4, 5]
                ]),
                B: jnp.array([
                    [6, 5],
                    [4, 3],
                    [2, 1]
                ])
            },
            {
                A: - jnp.array([
                    [0, 1],
                    [2, 3],
                    [2, 1]
                ]),
                B: - jnp.array([
                    [0, 1],
                    [2, 3],
                    [2, 1]
                ]),
                C: jnp.array([
                    [0, 2, 2],
                    [1, 3, 1]
                ]),
                D: jnp.array([4, 5]),
                E: jnp.array(9)
            }
        )

        test_fast_flux_f_from_rxn(
            self.rxns[8],
            {
                A: jnp.array([
                    [0, 1],
                    [2, 3],
                ])
            },
            {
                A: jnp.array([
                    [0, 2],
                    [1, 3],
                ]) - jnp.array([
                    [0, 1],
                    [2, 3],
                ]),
            }
        )

        test_fast_flux_f_from_rxn(
            self.rxns[9],
            {
                A: jnp.array([
                    [0, 1],
                    [2, 3],
                ]),
                B: jnp.array([
                    [3, 2],
                    [1, 0],
                ])
            },
            {
                A: jnp.array([
                    [0, 1/3],
                    [2/3, 0]
                ]) - jnp.array([
                    [0, 2/3],
                    [1/3, 0],
                ]),
                B: -jnp.array([
                    [0, 2/3],
                    [1/3, 0],
                ])
            }
        )

        test_fast_flux_f_from_rxn(
            self.rxns[10],
            {
                A: jnp.array(2),
                B: jnp.array(3),
                C: jnp.array(4)
            },
            {
                A: - jnp.array(2),
                B: - jnp.array(2),
                C: - jnp.array(2),
                D: jnp.array(2)
            }
        )
