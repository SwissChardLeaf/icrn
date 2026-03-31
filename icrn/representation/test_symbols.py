import unittest
from dataclasses import FrozenInstanceError
from .symbols import (
    Complex,
    IndexSymbol,
    RateConstant,
    Species,
    TensorExpression,
    TensorFunction,
    TensorLiteral,
    many_index_symbols,
    many_species,
    many_rate_constants,
)

import jax.numpy as jnp


class TestIndexSymbol(unittest.TestCase):
    def test_init(self):
        i = IndexSymbol("i")
        j = IndexSymbol("j", 3)

        self.assertTrue(i.label == "i")
        self.assertTrue(i.aux == 0)
        self.assertTrue(i.index_set == 0)

        self.assertTrue(j.label == "j")
        self.assertTrue(j.aux == 3)
        self.assertTrue(j.index_set == 3)

    def test_init_validation(self):
        with self.assertRaises(ValueError):
            IndexSymbol(1)
        with self.assertRaises(ValueError):
            IndexSymbol("i", "1")
        with self.assertRaises(ValueError):
            IndexSymbol("i", 1.0)
        with self.assertRaises(ValueError):
            IndexSymbol("i", -1)

    def test_many_index_symbols(self):
        i, j, k = many_index_symbols("i, j, k", 5)
        self.assertEqual(i, IndexSymbol("i", 5))
        self.assertEqual(j, IndexSymbol("j", 5))
        self.assertEqual(k, IndexSymbol("k", 5))

        l, m, n = many_index_symbols("l, m, n")
        self.assertEqual(l, IndexSymbol("l"))
        self.assertEqual(m, IndexSymbol("m"))
        self.assertEqual(n, IndexSymbol("n"))

        o = many_index_symbols("o")
        self.assertEqual(o, IndexSymbol("o"))

        with self.assertRaises(ValueError):
            many_index_symbols("i, j, k, l", -1)
        with self.assertRaises(ValueError):
            many_index_symbols("i, j, k, l", "1")
        with self.assertRaises(ValueError):
            many_index_symbols("i, j, k, l", 1.0)
        with self.assertRaises(ValueError):
            many_index_symbols(1, True)

    def test_ordering(self):
        i1 = IndexSymbol("i", 10)
        i2 = IndexSymbol("i", 9)
        i3 = IndexSymbol("i", 10)
        i4 = IndexSymbol("i")
        j = IndexSymbol("j", 10)
        k = IndexSymbol("k", 10)

        self.assertTrue(i1 < j)
        self.assertTrue(i1 <= j)
        self.assertTrue(i2 < i1)
        self.assertTrue(i1 > i4)

        self.assertTrue(k > j)
        self.assertTrue(k >= j)

        self.assertNotEqual(id(i1), id(i3))
        self.assertTrue(i1 == i3)
        self.assertFalse(i1 == i2)

        with self.assertRaises(TypeError):
            i1 == "i"

        with self.assertRaises(TypeError):
            i1 < 2

    def test_hash(self):
        i1 = IndexSymbol("i", 10)
        i2 = IndexSymbol("i", 10)
        i3 = IndexSymbol("i", 9)
        i4 = IndexSymbol("i")

        j1 = IndexSymbol("j")
        j2 = IndexSymbol("j", 10)

        self.assertNotEqual(id(i1), id(i2))
        self.assertEqual(hash(i1), hash(i2))
        self.assertNotEqual(hash(i1), hash(i3))
        self.assertNotEqual(hash(i1), hash(i4))

        self.assertNotEqual(hash(i4), hash(j1))
        self.assertNotEqual(hash(i1), hash(j2))

    def test_str(self):
        i = IndexSymbol("i", 10)
        self.assertEqual(str(i), "i")

        j = IndexSymbol("j")
        self.assertEqual(str(j), "j")

    def test_repr(self):
        i = IndexSymbol("i", 10)
        self.assertEqual(repr(i), "i:0...9")

        j = IndexSymbol("j")
        self.assertEqual(repr(j), "j:0")

    def test_frozen(self):
        i = IndexSymbol("i", 10)

        with self.assertRaises(FrozenInstanceError):
            i.label = "j"

        with self.assertRaises(FrozenInstanceError):
            i.aux = 11

        with self.assertRaises(FrozenInstanceError):
            i.index_set = 11


class TestSpecies(unittest.TestCase):
    def test_init(self):
        i = IndexSymbol("i")
        j = IndexSymbol("j")

        A = Species("A")
        self.assertEqual(A.label, "A")
        self.assertEqual(A.index_symbols, ())

        B = Species("B", (i,))
        self.assertEqual(B.label, "B")
        self.assertEqual(B.index_symbols, (i,))

        C = Species("C", (i, j))
        self.assertEqual(C.label, "C")
        self.assertEqual(C.index_symbols, (i, j))

        with self.assertRaises(ValueError):
            A = Species("A", (i, 2))

    def test_many_species(self):
        A, B, C = many_species("A, B, C")
        self.assertEqual(A, Species("A"))
        self.assertEqual(B, Species("B"))
        self.assertEqual(C, Species("C"))

        AB = many_species("A B")
        self.assertEqual(AB, Species("A B"))

        D = many_species("D")
        self.assertEqual(D, Species("D"))

        with self.assertRaises(ValueError):
            many_species(1)
        with self.assertRaises(ValueError):
            many_species(True)

    def test_frozen(self):
        A = Species("A")

        with self.assertRaises(FrozenInstanceError):
            A.label = "B"

    def test_ordering(self):
        A = Species("A")
        B = Species("B")
        C = Species("C")

        self.assertTrue(A < B)
        self.assertFalse(B > C)

    def test_str(self):
        i = IndexSymbol("i")
        j = IndexSymbol("j")

        A = Species("A")
        B = Species("B", (i, j))
        C = Species("C", (i,))

        self.assertEqual(str(A), "A")
        self.assertEqual(str(B), "B[i,j]")
        self.assertEqual(str(C), "C[i]")


    def test_repr(self):
        i = IndexSymbol("i", 5)
        j = IndexSymbol("j", 5)
        k = IndexSymbol("k")

        A = Species("A")
        B = Species("B", (i,))
        C = Species("C", (i, j))
        D = Species("D", (i, k))

        self.assertEqual(repr(A), "A")
        self.assertEqual(repr(B), "B[(i:0...4,)]")
        self.assertEqual(repr(C), "C[(i:0...4, j:0...4)]")
        self.assertEqual(repr(D), "D[(i:0...4, k:0)]")

    def test_add(self):
        A = Species("A")
        B = Species("B")
        C = Species("C")

        AB = A + B
        double_A = A + A
        CAB = C + A + B

        self.assertIsInstance(AB, Complex)
        self.assertIsInstance(double_A, Complex)
        self.assertIsInstance(CAB, Complex)

        self.assertEqual(AB.count_dict, {A: 1, B: 1})
        self.assertEqual(double_A.count_dict, {A: 2})
        self.assertEqual(CAB.count_dict, {A: 1, B: 1, C: 1})

    def test_mul(self):
        A = Species("A")

        double_A = 2 * A
        triple_A = A * 3

        self.assertIsInstance(double_A, Complex)
        self.assertIsInstance(triple_A, Complex)

        self.assertEqual(double_A.count_dict, {A: 2})
        self.assertEqual(triple_A.count_dict, {A: 3})

    def test_eval(self):
        i = IndexSymbol("i", 5)
        j = IndexSymbol("j", 5)

        A = Species("A")
        B = Species("B", (i, j))

        tensor_data = {
            A: jnp.arange(25).reshape((5, 5)),
            B: jnp.arange(25).reshape((5, 5))
        }

        self.assertTrue(
            jnp.all(jnp.equal(A.eval(tensor_data), jnp.arange(25).reshape((5, 5))))
        )

        self.assertTrue(
            jnp.all(jnp.equal(B.eval(tensor_data), jnp.arange(25).reshape((5, 5))))
        )

    def test_getitem(self):
        A = Species("A")
        i = IndexSymbol("i", 5)

        self.assertEqual(A.index_symbols, ())

        with self.assertRaises(ValueError):
            A[1, 2, 3]

        A_i = A[i]
        self.assertNotEqual(id(A), id(A_i))
        self.assertEqual(A_i.index_symbols, (i,))

    def test_hash(self):
        A = Species("A")
        i = IndexSymbol("i", 5)

        A_i1 = A[i]
        A_i2 = Species("A", (i,))

        self.assertNotEqual(id(A_i1), id(A_i2))
        self.assertEqual(hash(A_i1), hash(A_i2))
        self.assertEqual(A_i1, A_i2)


class TestComplex(unittest.TestCase):
    def setUp(self):
        A = Species("A")
        B = Species("B")

        self.comp = Complex({A: 1, B: 2})

        A, B, C = many_species("A, B, C")
        i, j, k = many_index_symbols("i, j, k", 5)

        self.complex1 = Complex({A[i]: 1, A[j]: 2, B: 3, C[i, j]: 2, C[j, j]: 1})
        self.complex2 = Complex({A[k]: 1, B: 2, C[j, i]: 2, C[k, j]: 3})

    def test_frozen(self):
        A = Species("A")
        D = Species("D")

        with self.assertRaises(FrozenInstanceError):
            self.complex1.count_dict = {A: 1, D: 2}

    def test_equal(self):
        A, B, C = many_species("A, B, C")
        i, j, k = many_index_symbols("i, j, k", 5)

        new_complex = Complex({A[k]: 1, B: 2, C[j, i]: 2, C[k, j]: 3})

        self.assertNotEqual(id(new_complex), id(self.complex2))
        self.assertEqual(new_complex, self.complex2)

    def test_add_species(self):
        A = Species("A")
        B = Species("B")
        C = Species("C")
        D = Species("D")

        add_c_complex = self.comp.add_species(C)

        self.assertEqual(add_c_complex, Complex({A: 1, B: 2, C: 1}))

        add_d_complex = add_c_complex.add_species(D, 3)

        self.assertEqual(add_d_complex, Complex({A: 1, B: 2, C: 1, D: 3}))

        A, B, C = many_species("A, B, C")
        i, j = many_index_symbols("i, j", 5)

        new_complex = Complex({})

        add_a_complex = new_complex.add_species(A[i])
        self.assertEqual(add_a_complex, Complex({A[i]: 1}))

        add_aj_complex = add_a_complex.add_species(A[j], 2)
        self.assertEqual(add_aj_complex, Complex({A[i]: 1, A[j]: 2}))

        add_b_complex = add_aj_complex.add_species(B, 3)
        self.assertEqual(add_b_complex, Complex({A[i]: 1, A[j]: 2, B: 3}))

        add_cij_complex = add_b_complex.add_species(C[i, j], 2)
        self.assertEqual(add_cij_complex, Complex({A[i]: 1, A[j]: 2, B: 3, C[i, j]: 2}))

        add_cjj_complex = add_cij_complex.add_species(C[j, j], 1)
        self.assertEqual(add_cjj_complex, self.complex1)

    def test_add_species_validation(self):
        A = Species("A")
        B = Species("B")
        empty = Complex({})
        nonempty = Complex({A: 1})

        with self.assertRaises(ValueError):
            empty.add_species(RateConstant("k"))

        with self.assertRaises(ValueError):
            empty.add_species("not a species")

        with self.assertRaises(ValueError):
            empty.add_species(Complex({B: 1}))

        with self.assertRaises(ValueError):
            empty.add_species(A, 1.0)

        with self.assertRaises(ValueError):
            empty.add_species(A, "1")

        with self.assertRaises(ValueError):
            empty.add_species(A, -1)

        with self.assertRaises(ValueError):
            empty.add_species(A, 0)

        with self.assertRaises(ValueError):
            nonempty.add_species(A, 0)

    def test_count_dict_validation(self):
        A, B = many_species("A, B")

        Complex({})
        Complex({A: 1, B: 2})

        with self.assertRaises(ValueError):
            Complex({A: 0})

        with self.assertRaises(ValueError):
            Complex({A: -1})

        with self.assertRaises(ValueError):
            Complex({A: 1.0})

        with self.assertRaises(ValueError):
            Complex({A: 1, B: True})

        with self.assertRaises(ValueError):
            Complex({"A": 1})

        with self.assertRaises(ValueError):
            Complex({RateConstant("k"): 1})

    def test_construction_operators(self):
        A, B, C = many_species("A, B, C")
        i, j = many_index_symbols("i, j", 5)

        new_complex = C[i, j] + A[i] + A[j] * 2 + 3 * B + C[i, j] + 1 * C[j, j]
        self.assertEqual(new_complex, self.complex1)

    def test_add(self):
        A, B, C = many_species("A, B, C")
        i, j, k = many_index_symbols("i, j, k", 5)

        add_complex = self.complex1 + self.complex2

        target_complex = Complex(
            {
                A[i]: 1,
                A[j]: 2,
                A[k]: 1,
                B: 5,
                C[i, j]: 2,
                C[j, i]: 2,
                C[j, j]: 1,
                C[k, j]: 3,
            }
        )

        self.assertEqual(add_complex, target_complex)

    def test_str(self):
        self.assertEqual(str(self.complex1), "A[i] + 2*A[j] + 3*B + 2*C[i,j] + C[j,j]")
        self.assertEqual(str(self.complex2), "A[k] + 2*B + 2*C[j,i] + 3*C[k,j]")


class TestRateConstant(unittest.TestCase):
    def test_init(self):
        i = IndexSymbol("i")
        j = IndexSymbol("j")

        A = RateConstant("A")
        self.assertEqual(A.label, "A")
        self.assertEqual(A.index_symbols, ())

        B = RateConstant("B", (i,))
        self.assertEqual(B.label, "B")
        self.assertEqual(B.index_symbols, (i,))

        C = RateConstant("C", (i, j))
        self.assertEqual(C.label, "C")
        self.assertEqual(C.index_symbols, (i, j))

        with self.assertRaises(ValueError):
            RateConstant("D", (1, 2))
        with self.assertRaises(ValueError):
            RateConstant("E", 1)
        with self.assertRaises(ValueError):
            RateConstant("A", (i, 2))

    def test_many_rate_constants(self):
        A, B, C = many_rate_constants("A, B, C")
        self.assertEqual(A, RateConstant("A"))
        self.assertEqual(B, RateConstant("B"))
        self.assertEqual(C, RateConstant("C"))

        AB = many_rate_constants("A B")
        self.assertEqual(AB, RateConstant("A B"))

        D = many_rate_constants("D")
        self.assertEqual(D, RateConstant("D"))

        with self.assertRaises(ValueError):
            many_rate_constants(1)

    def test_frozen(self):
        A = RateConstant("A")

        with self.assertRaises(FrozenInstanceError):
            A.label = "B"

    def test_ordering(self):
        A = RateConstant("A")
        B = RateConstant("B")
        C = RateConstant("C")

        self.assertTrue(A < B)
        self.assertFalse(B > C)

    def test_str(self):
        i = IndexSymbol("i")
        j = IndexSymbol("j")

        A = RateConstant("A")
        B = RateConstant("B", (i, j))
        C = RateConstant("C", (i,))

        self.assertEqual(str(A), "A")
        self.assertEqual(str(B), "B[i,j]")
        self.assertEqual(str(C), "C[i]")

    def test_repr(self):
        i = IndexSymbol("i", 5)
        j = IndexSymbol("j", 5)
        k = IndexSymbol("k")

        A = RateConstant("A")
        B = RateConstant("B", (i,))
        C = RateConstant("C", (i, j))
        D = RateConstant("D", (i, k))

        self.assertEqual(repr(A), "A")
        self.assertEqual(repr(B), "B[(i:0...4,)]")
        self.assertEqual(repr(C), "C[(i:0...4, j:0...4)]")
        self.assertEqual(repr(D), "D[(i:0...4, k:0)]")

    def test_add(self):
        A = RateConstant("A")
        B = RateConstant("B")
        C = RateConstant("C")

        AB = A + B
        double_A = A + A
        CAB = C + A + B

        self.assertIsInstance(AB, TensorFunction)
        self.assertIsInstance(double_A, TensorFunction)
        self.assertIsInstance(CAB, TensorFunction)

        self.assertEqual(AB, TensorFunction(jnp.add, (A, B)))
        self.assertEqual(double_A, TensorFunction(jnp.add, (A, A)))
        self.assertEqual(
            CAB,
            TensorFunction(jnp.add, (TensorFunction(jnp.add, (C, A)), B)),
        )

        A_plus_2 = A + 2
        self.assertEqual(A_plus_2, TensorFunction(jnp.add, (A, TensorLiteral(2))))
        A_plus_half = A + 0.5
        self.assertEqual(A_plus_half, TensorFunction(jnp.add, (A, TensorLiteral(0.5))))
        two_plus_A = 2 + A
        self.assertEqual(two_plus_A, TensorFunction(jnp.add, (TensorLiteral(2), A)))
        half_plus_A = 0.5 + A
        self.assertEqual(half_plus_A, TensorFunction(jnp.add, (TensorLiteral(0.5), A)))

        data = {A: jnp.array(3.0)}
        self.assertTrue(jnp.equal(A_plus_2.eval(data), 5.0))
        self.assertTrue(jnp.equal(A_plus_half.eval(data), 3.5))
        self.assertTrue(jnp.equal(two_plus_A.eval(data), 5.0))
        self.assertTrue(jnp.equal(half_plus_A.eval(data), 3.5))

    def test_mul(self):
        A = RateConstant("A")
        B = RateConstant("B")
        C = RateConstant("C")

        double_A = 2 * A
        triple_A = A * 3
        scaled_A = 2.5 * A
        A_times_float = A * 3.5

        AB = 2 * A * B
        ABC = A * B * C
        ABf = 2.5 * A * B

        self.assertIsInstance(double_A, TensorFunction)
        self.assertIsInstance(triple_A, TensorFunction)
        self.assertIsInstance(scaled_A, TensorFunction)
        self.assertIsInstance(A_times_float, TensorFunction)
        self.assertIsInstance(AB, TensorFunction)
        self.assertIsInstance(ABC, TensorFunction)
        self.assertIsInstance(ABf, TensorFunction)

        self.assertEqual(double_A, TensorFunction(jnp.multiply, (TensorLiteral(2), A)))
        self.assertEqual(triple_A, TensorFunction(jnp.multiply, (A, TensorLiteral(3))))
        self.assertEqual(scaled_A, TensorFunction(jnp.multiply, (TensorLiteral(2.5), A)))
        self.assertEqual(A_times_float, TensorFunction(jnp.multiply, (A, TensorLiteral(3.5))))
        self.assertEqual(
            AB,
            TensorFunction(
                jnp.multiply, (TensorFunction(jnp.multiply, (TensorLiteral(2), A)), B)
            ),
        )
        self.assertEqual(
            ABC,
            TensorFunction(
                jnp.multiply, (TensorFunction(jnp.multiply, (A, B)), C)
            ),
        )
        self.assertEqual(
            ABf,
            TensorFunction(
                jnp.multiply, (TensorFunction(jnp.multiply, (TensorLiteral(2.5), A)), B)
            ),
        )

        data = {A: jnp.array(4.0), B: jnp.array(2.0)}
        self.assertTrue(jnp.equal(double_A.eval(data), 8.0))
        self.assertTrue(jnp.equal(scaled_A.eval(data), 10.0))
        self.assertTrue(jnp.equal(AB.eval(data), 16.0))
        self.assertTrue(jnp.equal(ABf.eval(data), 20.0))

    def test_sub(self):
        A = RateConstant("A")
        B = RateConstant("B")

        AB_sub = A - B
        self.assertIsInstance(AB_sub, TensorFunction)
        self.assertEqual(AB_sub, TensorFunction(jnp.subtract, (A, B)))

        A_minus_2 = A - 2
        self.assertEqual(A_minus_2, TensorFunction(jnp.subtract, (A, TensorLiteral(2))))
        A_minus_two = A - 2.0
        self.assertEqual(A_minus_two, TensorFunction(jnp.subtract, (A, TensorLiteral(2.0))))

        two_minus_A = 2 - A
        self.assertEqual(two_minus_A, TensorFunction(jnp.subtract, (TensorLiteral(2), A)))
        two_point_zero_minus_A = 2.0 - A
        self.assertEqual(two_point_zero_minus_A, TensorFunction(jnp.subtract, (TensorLiteral(2.0), A)))

        data = {A: jnp.array(6.0), B: jnp.array(3.0)}
        self.assertTrue(jnp.equal(AB_sub.eval(data), 3.0))
        self.assertTrue(jnp.equal(A_minus_2.eval(data), 4.0))
        self.assertTrue(jnp.equal(A_minus_two.eval(data), 4.0))
        self.assertTrue(jnp.equal(two_minus_A.eval(data), -4.0))
        self.assertTrue(jnp.equal(two_point_zero_minus_A.eval(data), -4.0))

    def test_div(self):
        A = RateConstant("A")
        B = RateConstant("B")

        AB_div = A / B
        self.assertIsInstance(AB_div, TensorFunction)
        self.assertEqual(AB_div, TensorFunction(jnp.true_divide, (A, B)))

        A_over_2 = A / 2
        self.assertEqual(A_over_2, TensorFunction(jnp.true_divide, (A, TensorLiteral(2))))
        A_over_two = A / 2.0
        self.assertEqual(A_over_two, TensorFunction(jnp.true_divide, (A, TensorLiteral(2.0))))

        two_over_A = 2 / A
        self.assertEqual(two_over_A, TensorFunction(jnp.true_divide, (TensorLiteral(2), A)))
        two_point_zero_over_A = 2.0 / A
        self.assertEqual(two_point_zero_over_A, TensorFunction(jnp.true_divide, (TensorLiteral(2.0), A)))

        data = {A: jnp.array(6.0), B: jnp.array(3.0)}
        self.assertTrue(jnp.equal(AB_div.eval(data), 2.0))
        self.assertTrue(jnp.equal(A_over_2.eval(data), 3.0))
        self.assertTrue(jnp.equal(A_over_two.eval(data), 3.0))
        self.assertTrue(jnp.equal(two_over_A.eval(data), 1/3))
        self.assertTrue(jnp.equal(two_point_zero_over_A.eval(data),1/3))

    def test_eval(self):
        i = IndexSymbol("i", 5)
        j = IndexSymbol("j", 5)

        A = RateConstant("A")
        B = RateConstant("B", (i, j))

        tensor_data = {
            A: jnp.arange(25).reshape((5, 5)),
            B: jnp.arange(25).reshape((5, 5)),
        }

        self.assertTrue(
            jnp.all(jnp.equal(A.eval(tensor_data), jnp.arange(25).reshape((5, 5))))
        )

        self.assertTrue(
            jnp.all(jnp.equal(B.eval(tensor_data), jnp.arange(25).reshape((5, 5))))
        )

    def test_getitem(self):
        A = RateConstant("A")
        i = IndexSymbol("i", 5)

        self.assertEqual(A.index_symbols, ())

        with self.assertRaises(ValueError):
            A[1, 2, 3]

        with self.assertRaises(ValueError):
            A[1.5]

        A_i = A[i]
        self.assertNotEqual(id(A), id(A_i))
        self.assertEqual(A_i.index_symbols, (i,))

    def test_hash(self):
        A = RateConstant("A")
        i = IndexSymbol("i", 5)

        A_i1 = A[i]
        A_i2 = RateConstant("A", (i,))

        self.assertNotEqual(id(A_i1), id(A_i2))
        self.assertEqual(hash(A_i1), hash(A_i2))
        self.assertEqual(A_i1, A_i2)

class TestTensorLiteral(unittest.TestCase):
    def test_init(self):
        a = TensorLiteral(1)
        b = TensorLiteral(2.5)
        arr = jnp.array([1.0, 2.0])
        c = TensorLiteral(arr)

        self.assertEqual(a.numeric_value, 1)
        self.assertEqual(b.numeric_value, 2.5)
        self.assertTrue(jnp.all(jnp.equal(c.numeric_value, arr)))

    def test_index_symbols(self):
        self.assertEqual(TensorLiteral(0).index_symbols, ())
        self.assertEqual(TensorLiteral(jnp.array(1.0)).index_symbols, ())

    def test_eval(self):
        lit = TensorLiteral(7)
        arr_lit = TensorLiteral(jnp.array([1.0, 2.0]))
        data = {RateConstant("k"): jnp.array(99.0)}

        self.assertEqual(lit.eval(data), 7)
        self.assertEqual(lit.eval({}), 7)
        self.assertTrue(jnp.all(jnp.equal(arr_lit.eval(data), jnp.array([1.0, 2.0]))))

    def test_str(self):
        self.assertEqual(str(TensorLiteral(3)), "3")
        self.assertEqual(str(TensorLiteral(1.5)), "1.5")

    def test_repr(self):
        self.assertEqual(repr(TensorLiteral(-2)), "-2")
        x = jnp.array([1.0, 2.0])
        self.assertEqual(repr(TensorLiteral(x)), repr(x))

    def test_frozen(self):
        lit = TensorLiteral(1)
        with self.assertRaises(FrozenInstanceError):
            lit.numeric_value = 2

    def test_equality_and_hash(self):
        u = TensorLiteral(1)
        v = TensorLiteral(1)
        w = TensorLiteral(2)

        self.assertNotEqual(id(u), id(v))
        self.assertEqual(u, v)
        self.assertNotEqual(u, w)
        self.assertEqual(hash(u), hash(v))
        self.assertNotEqual(hash(u), hash(w))

    def test_add(self):
        A = RateConstant("A")
        two = TensorLiteral(2)
        i = IndexSymbol("i")

        A_plus_two = A + two
        self.assertIsInstance(A_plus_two, TensorFunction)
        self.assertEqual(A_plus_two, TensorFunction(jnp.add, (A, two)))
        self.assertEqual(A_plus_two.index_symbols, ())

        two_plus_A = two + A
        self.assertIsInstance(two_plus_A, TensorFunction)
        self.assertEqual(two_plus_A, TensorFunction(jnp.add, (two, A)))
        self.assertEqual(two_plus_A.index_symbols, ())

        with self.assertRaises(ValueError):
            A[i] + two

    def test_mul(self):
        A = RateConstant("A")
        two = TensorLiteral(2)
        i = IndexSymbol("i")

        A_times_two = A * two
        self.assertIsInstance(A_times_two, TensorFunction)
        self.assertEqual(A_times_two, TensorFunction(jnp.multiply, (A, two)))
        self.assertEqual(A_times_two.index_symbols, ())

        two_times_A = two * A
        self.assertIsInstance(two_times_A, TensorFunction)
        self.assertEqual(two_times_A, TensorFunction(jnp.multiply, (two, A)))
        self.assertEqual(two_times_A.index_symbols, ())

        with self.assertRaises(ValueError):
            A[i] * two

    def test_sub(self):
        A = RateConstant("A")
        two = TensorLiteral(2)
        i = IndexSymbol("i")

        A_minus_two = A - two
        self.assertIsInstance(A_minus_two, TensorFunction)
        self.assertEqual(A_minus_two, TensorFunction(jnp.subtract, (A, two)))
        self.assertEqual(A_minus_two.index_symbols, ())

        two_minus_A = two - A
        self.assertIsInstance(two_minus_A, TensorFunction)
        self.assertEqual(two_minus_A, TensorFunction(jnp.subtract, (two, A)))
        self.assertEqual(two_minus_A.index_symbols, ())

        with self.assertRaises(ValueError):
            A[i] - two

    def test_div(self):
        A = RateConstant("A")
        two = TensorLiteral(2)
        i = IndexSymbol("i")

        A_over_two = A / two
        self.assertIsInstance(A_over_two, TensorFunction)
        self.assertEqual(A_over_two, TensorFunction(jnp.true_divide, (A, two)))
        self.assertEqual(A_over_two.index_symbols, ())

        two_over_A = two / A
        self.assertIsInstance(two_over_A, TensorFunction)
        self.assertEqual(two_over_A, TensorFunction(jnp.true_divide, (two, A)))
        self.assertEqual(two_over_A.index_symbols, ())

        with self.assertRaises(ValueError):
            A[i] / two

    def test_neg(self):
        A = RateConstant("A")
        two = TensorLiteral(2)
        i = IndexSymbol("i")

        neg_A = -A
        self.assertIsInstance(neg_A, TensorFunction)
        self.assertEqual(neg_A, TensorFunction(jnp.negative, (A,)))
        self.assertEqual(neg_A.index_symbols, ())

        neg_two = -two
        self.assertIsInstance(neg_two, TensorFunction)
        self.assertEqual(neg_two, TensorFunction(jnp.negative, (two,)))
        self.assertEqual(neg_two.index_symbols, ())


class TestTensorFunction(unittest.TestCase):
    def test_init(self):
        A = RateConstant("A")
        B = RateConstant("B")

        tensor_function = TensorFunction(jnp.add, (A, B))
        self.assertEqual(tensor_function.fn, jnp.add)
        self.assertEqual(tensor_function.args, (A, B))

    def test_ufunc_nin(self):
        A = RateConstant("A")
        B = RateConstant("B")
        C = RateConstant("C")

        with self.assertRaises(ValueError):
            TensorFunction(jnp.add, (A,))

        with self.assertRaises(ValueError):
            TensorFunction(jnp.add, (A, B, C))

        with self.assertRaises(ValueError):
            TensorFunction(jnp.negative, (A, B))

    def test_init_validation(self):
        A = RateConstant("A")
        B = RateConstant("B")
        C = RateConstant("C")

        i = IndexSymbol("i")
        j = IndexSymbol("j")

        with self.assertRaises(ValueError):
            TensorFunction(jnp.add, (A[i], B[i,j]))

        with self.assertRaises(ValueError):
            TensorFunction(A, (A, B))

        with self.assertRaises(ValueError):
            TensorFunction(1.0, (A, B))


    def test_frozen(self):
        A = RateConstant("A")
        B = RateConstant("B")
        tensor_function = TensorFunction(jnp.add, (A, B))

        with self.assertRaises(FrozenInstanceError):
            tensor_function.fn = jnp.subtract

        with self.assertRaises(FrozenInstanceError):
            tensor_function.args = (A, B)

    def test_str(self):
        A = RateConstant("A")
        B = RateConstant("B")

        tensor_function = TensorFunction(jnp.add, (A, B))
        self.assertEqual(str(tensor_function), "add(A,B)")

    def test_repr(self):
        A = RateConstant("A")
        B = RateConstant("B")

        tensor_function = TensorFunction(jnp.add, (A, B))
        self.assertEqual(repr(tensor_function), "add(A,B)")

    def test_eval(self):
        A = RateConstant("A")
        B = RateConstant("B")
        C = RateConstant("C")

        data = {A: jnp.array(1.0), B: jnp.array(2.0), C: jnp.array(3.0)}

        tensor_function = A + B * C
        self.assertEqual(tensor_function.eval(data), 7.0)
        self.assertIsInstance(tensor_function.eval(data), jnp.ndarray)

        custom_f = lambda x, y, z: x + (y / z)
        tensor_function = TensorFunction(custom_f, (A, B, C))
        self.assertTrue(
            jnp.allclose(tensor_function.eval(data), 1.0 + 2.0 / 3.0)
        )