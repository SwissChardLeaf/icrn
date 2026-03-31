import unittest
from dataclasses import FrozenInstanceError
from .reactions import AbstractReaction, MassActionReaction, _matching_shapes
from .symbols import (
    Complex,
    many_index_symbols,
    many_species,
    RateConstant,
    TensorLiteral,
)

class TestReactionHelpers(unittest.TestCase):
    def test_matching_shapes(self):
        A, B, C = many_species("A, B, C")
        i, j = many_index_symbols("i, j")
        l, m = many_index_symbols("l, m", 3)
        n, o = many_index_symbols("n, o", 7)

        _matching_shapes(set([A[i], B[j], A[j], B[i]]))
        _matching_shapes(set([A[l], B[j], A[m], B[n]]))
        _matching_shapes(set([A[i, n], B[j], A[l,j], B[o]]))
        _matching_shapes(set([A[i, j, m], A[j, l, o], A[n, i, j]]))

        with self.assertRaises(ValueError):
            _matching_shapes(set([A[l], A[n]]))

        with self.assertRaises(ValueError):
            _matching_shapes(set([A[i], A[j, i]]))

        with self.assertRaises(ValueError):
            _matching_shapes(set([A[i], A]))

        with self.assertRaises(ValueError):
            _matching_shapes(set([A[i, j, m], A[l, m, o]]))

class TestExtendAbstractReaction(unittest.TestCase):
    def setUp(self):
        class TestReaction(AbstractReaction):
            A, B = many_species("A, B")
            alpha = RateConstant("alpha")

            def __init__(self, reactants, products, aux):
                super().__init__(reactants, products, aux)

            def flux(self):
                def flux_fn(state, rate_constant_data):
                    return {self.reactants: -self.aux, self.products: self.aux}
                return flux_fn

        self.TestReaction = TestReaction(A, B, alpha)

    def test_flux(self):
        A, B = many_species("A, B")
        alpha = RateConstant("alpha")
        self.assertEqual(self.TestReaction.flux(), {A: -alpha, B: alpha})

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
        beta = RateConstant("beta")

        with self.assertRaises(ValueError):
            MassActionReaction(alpha, C, alpha)

        with self.assertRaises(ValueError):
            MassActionReaction(B, C, A)

        with self.assertRaises(ValueError):
            MassActionReaction(B, C, None)
        

class MassActionReactionTests(unittest.TestCase):
    def test_flux_prep(self):
        pass

    def test_species_to_str(self):
        pass

    def test_all_index_symbols(self):
        pass

    def test_reaction_flux(self):
        pass


class FastReaction(unittest.TestCase):
    def test_reaction_flux(self):
        pass
