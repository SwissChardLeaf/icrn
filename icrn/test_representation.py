import unittest
from icrn import Species, RateConstant, many_species, many_rate_constants, \
                 many_index_symbols, IndexSymbol, IndexedRateConstant, \
                 IndexedSpecies, ConcreteSpecies, Complex, \
                 ConcreteRateConstant, BulkReaction, relu, \
                 NumericRateConstant, RateConstantFunction, FastReaction, \
                 ICRN
import jax.numpy as jnp
import jax

class IndexSymbolsTests(unittest.TestCase):
    def test_ordering(self):
        i = IndexSymbol("i", 10)
        j = IndexSymbol("j", 10)
        k = IndexSymbol("k", 10)

        self.assertTrue(i < j)
        self.assertFalse(j > k)

    def test_equality(self):
        i1 = IndexSymbol("i", 10)
        i2 = IndexSymbol("i", 10)
        i3 = IndexSymbol("i", 9)

        self.assertTrue(i1 == i2)
        self.assertFalse(i1 == i3)

class SpeciesTests(unittest.TestCase):
    def test_getitem_indexed_species(self):
        A = Species("A")
        B = Species("B")
        C = Species("C")

        i = IndexSymbol("i", 10)
        j = IndexSymbol("j", 10)

        A_i = IndexedSpecies(A, (i,))
        B_ij = IndexedSpecies(B, (i, j))
        C_jj = IndexedSpecies(C, (j, j))

        self.assertIsInstance(A_i, IndexedSpecies)
        self.assertEqual(A_i, A[i])
        self.assertEqual(B_ij, B[i,j])
        self.assertEqual(C_jj, C[j,j])
    
    def test_getitem_concrete_species(self):
        A = Species("A")
        B = Species("B")
        C = Species("C")

        A_2 = ConcreteSpecies(A, (2,))
        B_34 = ConcreteSpecies(B, (3, 4))
        C_55 = ConcreteSpecies(C, (5, 5))

        self.assertEqual(A_2, A[2])
        self.assertEqual(B_34, B[3,4])
        self.assertEqual(C_55, C[5,5])

    def test_ordering(self):
        A = Species("A")
        B = Species("B")
        C = Species("C")

        self.assertTrue(A < B)
        self.assertFalse(B > C)

    def test_str(self):
        A = Species("A")

        self.assertEqual(str(A), "A")

    def test_repr(self):
        A = Species("A")

        self.assertEqual(repr(A), "A")

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

        self.assertEqual(AB.count_dict, {A:1, B:1})
        self.assertEqual(double_A.count_dict, {A:2})
        self.assertEqual(CAB.count_dict, {A:1, B:1, C:1})

    def test_mul(self):
        A = Species("A")

        double_A = 2 * A
        triple_A = A * 3

        self.assertEqual(double_A.count_dict, {A:2})
        self.assertEqual(triple_A.count_dict, {A:3})

    def test_eval(self):
        A = Species("A")

        tensor_data = {
            A : jnp.arange(5)
        }

        self.assertTrue(jnp.all(jnp.equal(A.eval(tensor_data), jnp.arange(5))))

    def test_einsum_index_symbol_string(self):
        A = Species("A")

        self.assertEqual(A.einsum_index_symbol_string(False, False), "")
        self.assertEqual(A.einsum_index_symbol_string(True, False), "HW")
        self.assertEqual(A.einsum_index_symbol_string(True, True), "HW")

class IndexedSpeciesTests(unittest.TestCase):
    def setUp(self):
        A = Species("A")
        i = IndexSymbol("i", 10)
        j = IndexSymbol("j", 10)

        self.A_ij = IndexedSpecies(A, (i, j))
        self.A_ii = IndexedSpecies(A, (i, i))
    
    def test_eq(self):
        A2 = Species("A")
        i2 = IndexSymbol("i", 10)
        j2 = IndexSymbol("j", 10)

        A2_i2j2 = IndexedSpecies(A2, (i2, j2))

        self.assertEqual(self.A_ij, A2_i2j2)
    
    def test_index_symbols(self):
        i = IndexSymbol("i", 10)
        j = IndexSymbol("j", 10)

        self.assertEqual(self.A_ij.get_index_symbols_set(), {i, j}) 

    def test_index_symbols_replace_1(self):
        A = self.A_ij.base
        i, j = self.A_ij.index_symbols

        A_12 = A[1,2]

        self.assertIsInstance(A_12, ConcreteSpecies)
        self.assertEqual(self.A_ij.index_symbols_replace({i:1, j:2}), A_12)

    def test_index_symbols_replace_2(self):
        A = self.A_ij.base
        i, i = self.A_ii.index_symbols

        A_11 = A[1,1]

        self.assertIsInstance(A_11, ConcreteSpecies)
        self.assertEqual(self.A_ii.index_symbols_replace({i:1}), A_11)

    def test_str(self):
        self.assertEqual(str(self.A_ii), "A[i,i]")
        self.assertEqual(str(self.A_ij), "A[i,j]")

    def test_repr(self):
        self.assertEqual(repr(self.A_ii), "A[i,i]")
        self.assertEqual(repr(self.A_ij), "A[i,j]")

    def test_eval(self):
        A = self.A_ij.base

        tensor_data = {
            A : jnp.arange(5)
        }

        self.assertTrue(jnp.all(self.A_ij.eval(tensor_data) == jnp.arange(5)))

    def test_einsum_index_symbol_string(self):
        self.assertEqual(self.A_ij.einsum_index_symbol_string(False, False), "ij")
        self.assertEqual(self.A_ij.einsum_index_symbol_string(True, False), "HWij")
        self.assertEqual(self.A_ij.einsum_index_symbol_string(True, True), "HWij")

        self.assertEqual(self.A_ii.einsum_index_symbol_string(False, False), "ii")
        self.assertEqual(self.A_ii.einsum_index_symbol_string(True, False), "HWii")
        self.assertEqual(self.A_ii.einsum_index_symbol_string(True, True), "HWii")

class ConcreteSpeciesTests(unittest.TestCase):
    def setUp(self):
        A = Species("A")

        self.A_12 = ConcreteSpecies(A, (1,2))

    def test_str(self):
        self.assertEqual(str(self.A_12), "A[1,2]")

    def test_repr(self):
        self.assertEqual(repr(self.A_12), "A[1,2]")

    def test_eval(self):
        A = self.A_12.base

        tensor_data = {
            A : jnp.arange(10).reshape((2,5))
        }

        self.assertTrue(self.A_12.eval(tensor_data) == 7)

    def test_einsum_index_symbol_string(self):
        self.assertEqual(self.A_12.einsum_index_symbol_string(False, False), "")
        self.assertEqual(self.A_12.einsum_index_symbol_string(True, False), "")

class ComplexTests(unittest.TestCase):
    def setUp(self):
        A, B, C = many_species("A, B, C")
        i, j, k = many_index_symbols("i, j, k", 5)

        self.complex1 = Complex({A[i]:1, A[j]:2, B:3, C[i,j]:2, C[j,j]:1})
        self.complex2 = Complex({A[k]:1, B:2, C[j,i]:2, C[k,j]:3})

    def test_add_reactant(self):
        A, B, C = many_species("A, B, C")
        i, j = many_index_symbols("i, j", 5)

        new_complex = Complex({})

        new_complex.add_reactant(A[i])
        self.assertEqual(new_complex, Complex({A[i]:1}))

        new_complex.add_reactant(A[j], 2)
        self.assertEqual(new_complex, Complex({A[i]:1, A[j]:2})) 

        new_complex.add_reactant(B, 3)
        self.assertEqual(new_complex, Complex({A[i]:1, A[j]:2, B:3}))

        new_complex.add_reactant(C[i,j], 2)
        self.assertEqual(new_complex, Complex({A[i]:1, A[j]:2, B:3, C[i,j]:2}))

        new_complex.add_reactant(C[j,j], 1)
        self.assertEqual(new_complex, self.complex1) 

    def test_construction_operators(self):
        A, B, C = many_species("A, B, C")
        i, j = many_index_symbols("i, j", 5)

        new_complex = C[i,j] + A[i] + A[j] * 2 + 3 * B + C[i,j] + 1 * C[j,j]
        self.assertEqual(new_complex, self.complex1)

    def test_add(self):
        A, B, C = many_species("A, B, C")
        i, j, k = many_index_symbols("i, j, k", 5)

        add_complex = self.complex1 + self.complex2

        target_complex = Complex({
            A[i]   : 1, 
            A[j]   : 2,
            A[k]   : 1,
            B      : 5,
            C[i,j] : 2,
            C[j,i] : 2, 
            C[j,j] : 1,
            C[k,j] : 3
        })

        self.assertEqual(add_complex, target_complex) 

    def test_get_index_symbols_set(self):
        i, j, k = many_index_symbols("i, j, k", 5)

        self.assertEqual(self.complex1.get_index_symbols_set(), {i,j})
        self.assertEqual(self.complex2.get_index_symbols_set(), {i,j,k})

    def test_index_symbols_replace(self):
        A, B, C = many_species("A, B, C")
        i, j, k = many_index_symbols("i, j, k", 5)

        target1 = Complex({A[1]:1, A[2]:2, B:3, C[1,2]:2, C[2,2]:1})
        self.assertEqual(self.complex1.index_symbols_replace({i:1,j:2}), target1)

        target2 = Complex({A[3]:1, B:2, C[2,1]:2, C[3,2]:3})
        self.assertEqual(self.complex2.index_symbols_replace({i:1,j:2,k:3}), target2)

class RateConstantTests(unittest.TestCase):
    def test_getitem_indexed_rate_constant(self):
        alpha = RateConstant("alpha")
        beta = RateConstant("beta")
        gamma = RateConstant("gamma")

        i = IndexSymbol("i", 10)
        j = IndexSymbol("j", 10)

        alpha_i = IndexedRateConstant(alpha, (i,))
        beta_ij = IndexedRateConstant(beta, (i, j))
        gamma_jj = IndexedRateConstant(gamma, (j, j))

        self.assertEqual(alpha_i, alpha[i])
        self.assertEqual(beta_ij, beta[i,j])
        self.assertEqual(gamma_jj, gamma[j,j])

    def test_getitem_concrete_rate_constant(self):
        alpha = RateConstant("alpha")
        beta = RateConstant("beta")
        gamma = RateConstant("gamma")

        alpha_2 = ConcreteRateConstant(alpha, (2,))
        beta_34 = ConcreteRateConstant(beta, (3, 4))
        gamma_55 = IndexedRateConstant(gamma, (5, 5))

        self.assertEqual(alpha_2, alpha[2])
        self.assertEqual(beta_34, beta[3,4])
        self.assertEqual(gamma_55, gamma[5,5])

    def test_ordering(self):
        alpha = RateConstant("alpha")
        beta = RateConstant("beta")
        gamma = RateConstant("gamma")

        self.assertTrue(alpha < beta)
        self.assertFalse(beta > gamma)

    def test_str(self):
        alpha = RateConstant("alpha")

        self.assertEqual(str(alpha), "alpha")

    def test_repr(self):
        alpha = RateConstant("alpha")

        self.assertEqual(repr(alpha), "alpha")

    def test_eval(self):
        alpha = RateConstant("alpha")
        
        tensor_data = {
            alpha : jnp.arange(5)
        }

        self.assertTrue(jnp.all(alpha.eval(tensor_data) == jnp.arange(5)))

    def test_einsum_index_symbol_string(self):
        alpha = RateConstant("alpha")
        self.assertEqual(alpha.einsum_index_symbol_string(False, False), "")
        self.assertEqual(alpha.einsum_index_symbol_string(True, False), "")
        self.assertEqual(alpha.einsum_index_symbol_string(True, True), "HW")

class IndexedRateConstantTests(unittest.TestCase):
    def setUp(self):
        alpha = RateConstant("alpha")
        i = IndexSymbol("i", 10)
        j = IndexSymbol("j", 10)

        self.alpha_ij = IndexedRateConstant(alpha, (i, j))
        self.alpha_ii = IndexedRateConstant(alpha, (i, i))
    
    def test_eq(self):
        alpha2 = RateConstant("alpha")
        i2 = IndexSymbol("i", 10)
        j2 = IndexSymbol("j", 10)

        alpha2_i2j2 = IndexedRateConstant(alpha2, (i2, j2))

        self.assertEqual(self.alpha_ij, alpha2_i2j2)
    
    def test_index_symbols(self):
        i = IndexSymbol("i", 10)
        j = IndexSymbol("j", 10)

        self.assertEqual(self.alpha_ij.get_index_symbols_set(), {i, j}) 

    def test_index_symbols_replace_1(self):
        alpha = self.alpha_ij.base
        i, j = self.alpha_ij.index_symbols

        alpha_12 = alpha[1,2]

        self.assertIsInstance(alpha_12, ConcreteRateConstant)
        self.assertEqual(self.alpha_ij.index_symbols_replace({i:1, j:2}), alpha_12)

    def test_index_symbols_replace_2(self):
        alpha = self.alpha_ij.base
        i, i = self.alpha_ii.index_symbols

        alpha_11 = alpha[1,1]

        self.assertIsInstance(alpha_11, ConcreteRateConstant)
        self.assertEqual(self.alpha_ii.index_symbols_replace({i:1}), alpha_11)

    def test_str(self):
        self.assertEqual(str(self.alpha_ii), "alpha[i,i]")
        self.assertEqual(str(self.alpha_ij), "alpha[i,j]")

    def test_repr(self):
        self.assertEqual(repr(self.alpha_ii), "alpha[i,i]")
        self.assertEqual(repr(self.alpha_ij), "alpha[i,j]")

    def test_eval(self):
        alpha = RateConstant("alpha")

        tensor_data = {
            alpha : jnp.arange(5)
        }

        self.assertTrue(jnp.all(self.alpha_ij.eval(tensor_data) == jnp.arange(5)))

    def test_einsum_index_symbol_string(self):
        self.assertEqual(self.alpha_ij.einsum_index_symbol_string(False, False), "ij")
        self.assertEqual(self.alpha_ij.einsum_index_symbol_string(True, False), "ij")
        self.assertEqual(self.alpha_ij.einsum_index_symbol_string(True, True), "HWij")

        self.assertEqual(self.alpha_ii.einsum_index_symbol_string(False, False), "ii")
        self.assertEqual(self.alpha_ii.einsum_index_symbol_string(True, False), "ii")
        self.assertEqual(self.alpha_ii.einsum_index_symbol_string(True, True), "HWii")

class ConcreteRateConstantTests(unittest.TestCase):
    def setUp(self):
        alpha = RateConstant("alpha")

        self.alpha_12 = ConcreteRateConstant(alpha, (1,2))

    def test_str(self):
        self.assertEqual(str(self.alpha_12), "alpha[1,2]")

    def test_repr(self):
        self.assertEqual(repr(self.alpha_12), "alpha[1,2]")

    def test_eval(self):
        alpha = RateConstant("alpha")

        tensor_data = {
            alpha : jnp.arange(10).reshape((2,5))
        }

        self.assertTrue(self.alpha_12.eval(tensor_data) == 7)

    def test_einsum_index_symbol_string(self):
        self.assertEqual(self.alpha_12.einsum_index_symbol_string(False, False), "")
        self.assertEqual(self.alpha_12.einsum_index_symbol_string(True, False), "")
        self.assertEqual(self.alpha_12.einsum_index_symbol_string(True, True), "")

class NumericRateConstantTests(unittest.TestCase):
    def setUp(self):
        self.two = NumericRateConstant(2)
        self.pi = NumericRateConstant(3.14)
    
    def test_eval(self):
        A = Species("A")

        tensor_data = {A : 3}

        self.assertEqual(self.two.eval(tensor_data), 2)
        self.assertEqual(self.pi.eval(tensor_data), 3.14)

    def test_einsum_index_symbol_string(self):
        self.assertEqual(self.two.einsum_index_symbol_string(), "")
        self.assertEqual(self.pi.einsum_index_symbol_string(), "")

    def test_str(self):
        self.assertEqual(str(self.two), "2")
        self.assertEqual(str(self.pi), "3.14")

    def test_repr(self):
        self.assertEqual(repr(self.two), "2")
        self.assertEqual(repr(self.pi), "3.14")

class RateConstantFunctionTests(unittest.TestCase):
    def setUp(self):
        alpha = RateConstant("alpha")
        i, j = many_index_symbols("i, j", 5)

        self.alpha = alpha
        self.alpha_ij = alpha[i,j]
        self.alpha_ii = alpha[i,i]
        self.alpha_12 = alpha[1,2]

        def add_one(x): return x + 1
        self.add_one = add_one

        self.add_one_alpha = RateConstantFunction(add_one, self.alpha)
        self.add_one_alpha_ii = RateConstantFunction(add_one, self.alpha_ii)
        self.add_one_alpha_ij = RateConstantFunction(add_one, self.alpha_ij)
        self.add_one_alpha_12 = RateConstantFunction(add_one, self.alpha_12)

        self.relu_alpha = relu(self.alpha)
        self.relu_alpha_ii = relu(self.alpha_ii)
        self.relu_alpha_ij = relu(self.alpha_ij)
        self.relu_alpha_12 = relu(self.alpha_12)

        self.relu_neg_alpha = relu(-self.alpha)
        self.relu_neg_alpha_ii = relu(-self.alpha_ii)
        self.relu_neg_alpha_ij = relu(-self.alpha_ij)
        self.relu_neg_alpha_12 = relu(-self.alpha_12)

    def test_get_index_symbols_set(self):
        i, j = many_index_symbols("i, j", 5)

        self.assertEqual(self.add_one_alpha.get_index_symbols_set(), set())
        self.assertEqual(self.add_one_alpha_ii.get_index_symbols_set(), {i})
        self.assertEqual(self.add_one_alpha_ij.get_index_symbols_set(), {i, j})
        self.assertEqual(self.add_one_alpha_12.get_index_symbols_set(), set())

        self.assertEqual(self.relu_alpha.get_index_symbols_set(), set())
        self.assertEqual(self.relu_alpha_ii.get_index_symbols_set(), {i})
        self.assertEqual(self.relu_alpha_ij.get_index_symbols_set(), {i, j})
        self.assertEqual(self.relu_alpha_12.get_index_symbols_set(), set())
        
    def test_index_symbols_replace(self):
        alpha = RateConstant("alpha")
        i, j = many_index_symbols("i, j", 5)

        self.assertEqual(
            self.add_one_alpha.index_symbols_replace({i:1, j:2}),
            RateConstantFunction(self.add_one, alpha)
        )

        self.assertEqual(
            self.add_one_alpha_ii.index_symbols_replace({i:1, j:2}),
            RateConstantFunction(self.add_one, alpha[1,1])
        )

        self.assertEqual(
            self.add_one_alpha_ij.index_symbols_replace({i:1, j:2}),
            RateConstantFunction(self.add_one, alpha[1,2])
        )

        self.assertEqual(
            self.add_one_alpha_12.index_symbols_replace({i:1, j:2}),
            RateConstantFunction(self.add_one, alpha[1,2])
        )

        self.assertEqual(
            self.relu_alpha.index_symbols_replace({i:1, j:2}),
            RateConstantFunction(jax.nn.relu, alpha)
        )

        self.assertEqual(
            self.relu_alpha_ii.index_symbols_replace({i:1, j:2}),
            RateConstantFunction(jax.nn.relu, alpha[1,1])
        )

        self.assertEqual(
            self.relu_alpha_ij.index_symbols_replace({i:1, j:2}),
            RateConstantFunction(jax.nn.relu, alpha[1,2])
        )

        self.assertEqual(
            self.relu_alpha_12.index_symbols_replace({i:1, j:2}),
            RateConstantFunction(jax.nn.relu, alpha[1,2])
        )
        
        self.assertEqual(
            self.relu_neg_alpha.index_symbols_replace({i:1, j:2}),
            RateConstantFunction(jax.nn.relu, RateConstantFunction(jnp.negative, alpha))
        )

        self.assertEqual(
            self.relu_neg_alpha_ii.index_symbols_replace({i:1, j:2}),
            RateConstantFunction(jax.nn.relu, RateConstantFunction(jnp.negative, alpha[1,1]))
        )

        self.assertEqual(
            self.relu_neg_alpha_ij.index_symbols_replace({i:1, j:2}),
            RateConstantFunction(jax.nn.relu, RateConstantFunction(jnp.negative, alpha[1,2]))
        )

        self.assertEqual(
            self.relu_neg_alpha_12.index_symbols_replace({i:1, j:2}),
            RateConstantFunction(jax.nn.relu, RateConstantFunction(jnp.negative, alpha[1,2]))
        )

    def test_eval(self):
        alpha = RateConstant("alpha")

        tensor_data = {
            alpha : jnp.array([[1, -2, 5], [3, -4, -6]])
        }

        self.assertTrue(jnp.all(
            self.add_one_alpha.eval(tensor_data) == \
            jnp.array([[2, -1, 6], [4, -3, -5]])
        ))

        self.assertTrue(jnp.all(
            self.add_one_alpha_ii.eval(tensor_data) == \
            jnp.array([[2, -1, 6], [4, -3, -5]])
        ))

        self.assertTrue(jnp.all(
            self.add_one_alpha_ij.eval(tensor_data) == \
            jnp.array([[2, -1, 6], [4, -3, -5]])
        ))

        self.assertTrue(jnp.all(
            self.add_one_alpha_12.eval(tensor_data) == \
            jnp.array(-5)
        ))

        self.assertTrue(jnp.all(
            self.relu_alpha.eval(tensor_data) == \
            jnp.array([[1, 0, 5], [3, 0, 0]])
        ))

        self.assertTrue(jnp.all(
            self.relu_alpha_ii.eval(tensor_data) == \
            jnp.array([[1, 0, 5], [3, 0, 0]])
        ))

        self.assertTrue(jnp.all(
            self.relu_alpha_ij.eval(tensor_data) == \
            jnp.array([[1, 0, 5], [3, 0, 0]])
        ))

        self.assertTrue(jnp.all(
            self.relu_alpha_12.eval(tensor_data) == \
            jnp.array(0)
        ))

        self.assertTrue(jnp.all(
            self.relu_neg_alpha.eval(tensor_data) == \
            jnp.array([[0, 2, 0], [0, 4, 6]])
        ))

        self.assertTrue(jnp.all(
            self.relu_neg_alpha_ii.eval(tensor_data) == \
            jnp.array([[0, 2, 0], [0, 4, 6]])
        ))

        self.assertTrue(jnp.all(
            self.relu_neg_alpha_ij.eval(tensor_data) == \
            jnp.array([[0, 2, 0], [0, 4, 6]])
        ))

        self.assertTrue(jnp.all(
            self.relu_neg_alpha_12.eval(tensor_data) == \
            jnp.array(6)
        ))


    def test_einsum_index_symbol_string(self):
        self.assertEqual(self.add_one_alpha.einsum_index_symbol_string(False, False), "")
        self.assertEqual(self.add_one_alpha.einsum_index_symbol_string(True, False), "")
        self.assertEqual(self.add_one_alpha.einsum_index_symbol_string(True, True), "HW")

        self.assertEqual(self.add_one_alpha_ii.einsum_index_symbol_string(False, False), "ii")
        self.assertEqual(self.add_one_alpha_ii.einsum_index_symbol_string(True, False), "ii")
        self.assertEqual(self.add_one_alpha_ii.einsum_index_symbol_string(True, True), "HWii")

        self.assertEqual(self.add_one_alpha_ij.einsum_index_symbol_string(False, False), "ij")
        self.assertEqual(self.add_one_alpha_ij.einsum_index_symbol_string(True, False), "ij")
        self.assertEqual(self.add_one_alpha_ij.einsum_index_symbol_string(True, True), "HWij")

        self.assertEqual(self.add_one_alpha_12.einsum_index_symbol_string(False, False), "")
        self.assertEqual(self.add_one_alpha_12.einsum_index_symbol_string(True, False), "")
        self.assertEqual(self.add_one_alpha_12.einsum_index_symbol_string(True, True), "")

        self.assertEqual(self.relu_alpha.einsum_index_symbol_string(False, False), "")
        self.assertEqual(self.relu_alpha.einsum_index_symbol_string(True, False), "")
        self.assertEqual(self.relu_alpha.einsum_index_symbol_string(True, True), "HW")

        self.assertEqual(self.relu_alpha_ii.einsum_index_symbol_string(False, False), "ii")
        self.assertEqual(self.relu_alpha_ii.einsum_index_symbol_string(True, False), "ii")
        self.assertEqual(self.relu_alpha_ii.einsum_index_symbol_string(True, True), "HWii")

        self.assertEqual(self.relu_alpha_ij.einsum_index_symbol_string(False, False), "ij")
        self.assertEqual(self.relu_alpha_ij.einsum_index_symbol_string(True, False), "ij")
        self.assertEqual(self.relu_alpha_ij.einsum_index_symbol_string(True, True), "HWij")

        self.assertEqual(self.relu_alpha_12.einsum_index_symbol_string(False, False), "")
        self.assertEqual(self.relu_alpha_12.einsum_index_symbol_string(True, False), "")
        self.assertEqual(self.relu_alpha_12.einsum_index_symbol_string(True, True), "")

    def test_str(self):
        self.assertEqual(str(self.add_one_alpha), "add_one(alpha)")
        self.assertEqual(str(self.add_one_alpha_ii), "add_one(alpha[i,i])")
        self.assertEqual(str(self.add_one_alpha_ij), "add_one(alpha[i,j])")
        self.assertEqual(str(self.add_one_alpha_12), "add_one(alpha[1,2])")

        self.assertEqual(str(self.relu_alpha), "relu(alpha)")
        self.assertEqual(str(self.relu_alpha_ii), "relu(alpha[i,i])")
        self.assertEqual(str(self.relu_alpha_ij), "relu(alpha[i,j])")
        self.assertEqual(str(self.relu_alpha_12), "relu(alpha[1,2])")

    def test_repr(self):
        self.assertEqual(repr(self.add_one_alpha), "add_one(alpha)")
        self.assertEqual(repr(self.add_one_alpha_ii), "add_one(alpha[i,i])")
        self.assertEqual(repr(self.add_one_alpha_ij), "add_one(alpha[i,j])")
        self.assertEqual(repr(self.add_one_alpha_12), "add_one(alpha[1,2])")

        self.assertEqual(repr(self.relu_alpha), "relu(alpha)")
        self.assertEqual(repr(self.relu_alpha_ii), "relu(alpha[i,i])")
        self.assertEqual(repr(self.relu_alpha_ij), "relu(alpha[i,j])")
        self.assertEqual(repr(self.relu_alpha_12), "relu(alpha[1,2])")

class ManyObjectsTests(unittest.TestCase):
    def test_many_index_symbols(self):
        i1, j1, k1 = many_index_symbols("i, j, k", 10)

        i2 = IndexSymbol("i", 10)
        j2 = IndexSymbol("j", 10)
        k2 = IndexSymbol("k", 10)

        self.assertIsInstance(i1, IndexSymbol)
        self.assertIsInstance(j1, IndexSymbol)
        self.assertIsInstance(k1, IndexSymbol)

        self.assertEqual(i1, i2)
        self.assertEqual(j1, j2)
        self.assertEqual(k1, k2)
    
    def test_many_species(self):
        A1, B1, C1 = many_species("A, B, C")

        A2 = Species("A")
        B2 = Species("B")
        C2 = Species("C")

        self.assertIsInstance(A1, Species)
        self.assertIsInstance(B1, Species)
        self.assertIsInstance(C1, Species)

        self.assertEqual(A1, A2)
        self.assertEqual(B1, B2)
        self.assertEqual(C1, C2)

    def test_many_rate_constants(self):
        alpha1, beta1, gamma1 = many_rate_constants("alpha, beta, gamma")

        alpha2 = RateConstant("alpha")
        beta2 = RateConstant("beta")
        gamma2 = RateConstant("gamma")

        self.assertIsInstance(alpha1, RateConstant)
        self.assertIsInstance(beta1, RateConstant)
        self.assertIsInstance(gamma1, RateConstant)

        self.assertEqual(alpha1, alpha2)
        self.assertEqual(beta1, beta2)
        self.assertEqual(gamma1, gamma2)

class BulkReactionTests(unittest.TestCase):
    def setUp(self):
        A, B, C, D, E, F, G = many_species("A, B, C, D, E, F, G")
        alpha, beta, gamma = many_rate_constants("alpha, beta, gamma")
        i, j, k = many_index_symbols("i, j, k", 10)

        self.rxn1 = BulkReaction(A[i,j]+2*B[j,k], A[i,j] + C[i,k], alpha[i])
        self.rxn2 = BulkReaction(D + E, F, relu(gamma[i,j]))
        self.rxn3 = BulkReaction(D + B[i,j] + 3 * G[i], 2 * D + B[j, i] + G[k], 2.)
        self.rxn4 = BulkReaction(D + D, E + F, beta)
        self.rxn5 = BulkReaction(A[1,1] + B[2,3], 2 * C[1,2], alpha[4])

    def test_shapes(self):
        A, B, C, D, E, F, G = many_species("A, B, C, D, E, F, G")
        alpha, beta, gamma = many_rate_constants("alpha, beta, gamma")

        self.assertEqual(self.rxn1.shapes(),
                         {
                             A : (10, 10),
                             B : (10, 10),
                             C : (10, 10),
                             alpha : (10,)
                         }
        )
        self.assertEqual(self.rxn2.shapes(),
                         {
                             D : (),
                             E : (),
                             F : (),
                             gamma : (10, 10)
                         }
        )
        self.assertEqual(self.rxn3.shapes(),
                         {
                             D : (),
                             B : (10, 10),
                             G : (10,)
                         }
        )
        self.assertEqual(self.rxn4.shapes(),
                         {
                             D : (),
                             E : (),
                             F : (),
                             beta : ()
                         }
        )
        self.assertEqual(self.rxn5.shapes(),
                         dict()
        )


    def test_get_index_symbols_set(self):
        i, j, k = many_index_symbols("i, j, k", 10)

        self.assertEqual(self.rxn1.get_index_symbols_set(), {i, j, k})
        self.assertEqual(self.rxn2.get_index_symbols_set(), {i, j})
        self.assertEqual(self.rxn3.get_index_symbols_set(), {i, j, k})
        self.assertEqual(self.rxn4.get_index_symbols_set(), set())
        self.assertEqual(self.rxn5.get_index_symbols_set(), set())

    def test_index_symbols_replace(self):
        A, B, C, D, E, F, G = many_species("A, B, C, D, E, F, G")
        alpha, beta, gamma = many_rate_constants("alpha, beta, gamma")
        i, j, k = many_index_symbols("i, j, k", 10)

        values_dict = {i:1, j:2, k:3}

        rxn1_replaced = self.rxn1.index_symbols_replace(values_dict)
        rxn1_target = BulkReaction(A[1,2]+2*B[2,3], A[1,2] + C[1,3], alpha[1])
        self.assertEqual(rxn1_replaced.reactants, rxn1_target.reactants)
        self.assertEqual(rxn1_replaced.products, rxn1_target.products)
        self.assertEqual(rxn1_replaced.aux, rxn1_target.aux)
        self.assertEqual(rxn1_replaced.name, rxn1_target.name)
        self.assertEqual(rxn1_replaced.rule, rxn1_target.rule)

        rxn2_replaced = self.rxn2.index_symbols_replace(values_dict)
        rxn2_target = BulkReaction(D + E, F, relu(gamma[1,2]))
        self.assertEqual(rxn2_replaced.reactants, rxn2_target.reactants)
        self.assertEqual(rxn2_replaced.products, rxn2_target.products)
        self.assertEqual(rxn2_replaced.aux, rxn2_target.aux)
        self.assertEqual(rxn2_replaced.name, rxn2_target.name)
        self.assertEqual(rxn2_replaced.rule, rxn2_target.rule)

        rxn3_replaced = self.rxn3.index_symbols_replace(values_dict)
        rxn3_target = BulkReaction(D + B[1,2] + 3 * G[1], 2 * D + B[2, 1] + G[3], 2.)
        self.assertEqual(rxn3_replaced.reactants, rxn3_target.reactants)
        self.assertEqual(rxn3_replaced.products, rxn3_target.products)
        self.assertEqual(rxn3_replaced.aux, rxn3_target.aux)
        self.assertEqual(rxn3_replaced.name, rxn3_target.name)
        self.assertEqual(rxn3_replaced.rule, rxn3_target.rule)

        rxn4_replaced = self.rxn4.index_symbols_replace(values_dict)
        rxn4_target = BulkReaction(D + D, E + F, beta)
        self.assertEqual(rxn4_replaced.reactants, rxn4_target.reactants)
        self.assertEqual(rxn4_replaced.products, rxn4_target.products)
        self.assertEqual(rxn4_replaced.aux, rxn4_target.aux)
        self.assertEqual(rxn4_replaced.name, rxn4_target.name)
        self.assertEqual(rxn4_replaced.rule, rxn4_target.rule)

        rxn5_replaced = self.rxn5.index_symbols_replace(values_dict)
        rxn5_target = BulkReaction(A[1,1] + B[2,3], 2 * C[1,2], alpha[4])
        self.assertEqual(rxn5_replaced.reactants, rxn5_target.reactants)
        self.assertEqual(rxn5_replaced.products, rxn5_target.products)
        self.assertEqual(rxn5_replaced.aux, rxn5_target.aux)
        self.assertEqual(rxn5_replaced.name, rxn5_target.name)
        self.assertEqual(rxn5_replaced.rule, rxn5_target.rule)

    def test_flux(self):
        A, B, C, D, E, F, G = many_species("A, B, C, D, E, F, G")
        alpha, beta, gamma = many_rate_constants("alpha, beta, gamma")

        tensor_data = {
            A : jnp.arange(100).reshape((10,10)),
            B : 2 * jnp.arange(100).reshape((10,10)),
            C : 3 * jnp.arange(100).reshape((10,10)),
            D : jnp.array(10.1),
            E : jnp.array(11.2),
            F : jnp.array(12.3),
            G : 0.5 * jnp.arange(10),
            alpha : 1.1 * jnp.arange(10),
            beta : jnp.array(10.1),
            gamma : 0.01 * jnp.arange(-50, 50, 1).reshape((10,10))
        }

        dynamics_dict1 = self.rxn1.build_flux(None, False)(tensor_data)
        self.assertEqual(dynamics_dict1.keys(), {B, C})
        self.assertTrue(jnp.allclose(dynamics_dict1[B],
                                -2 * jnp.einsum("i,ij,jk->jk",
                                            tensor_data[alpha],
                                            tensor_data[A],
                                            jnp.power(tensor_data[B], 2)
                                )))
        
        self.assertTrue(jnp.allclose(dynamics_dict1[C],
                                jnp.einsum("i,ij,jk->ik",
                                           tensor_data[alpha],
                                           tensor_data[A],
                                           jnp.power(tensor_data[B], 2)
                                )))

        dynamics_dict2 = self.rxn2.build_flux(None, False)(tensor_data)
        self.assertEqual(dynamics_dict2.keys(), {D, E, F})
        self.assertTrue(jnp.allclose(dynamics_dict2[D],
                                -jnp.einsum("ij,,->",
                                            jax.nn.relu(tensor_data[gamma]),
                                            tensor_data[D],
                                            tensor_data[E]
                                )))
        
        self.assertTrue(jnp.allclose(dynamics_dict2[E],
                                -jnp.einsum("ij,,->",
                                            jax.nn.relu(tensor_data[gamma]),
                                            tensor_data[D],
                                            tensor_data[E]
                                )))
        self.assertTrue(jnp.allclose(dynamics_dict2[F],
                                jnp.einsum("ij,,->",
                                            jax.nn.relu(tensor_data[gamma]),
                                            tensor_data[D],
                                            tensor_data[E]
                                )))
        
        dynamics_dict3 = self.rxn3.build_flux(None, False)(tensor_data)
        self.assertEqual(dynamics_dict3.keys(), {B, D, G})
        self.assertTrue(jnp.allclose(dynamics_dict3[B],
                                jnp.sum(jnp.tile(jnp.einsum(",,ij,i->ji",
                                            2.,
                                            tensor_data[D],
                                            tensor_data[B],
                                            jnp.power(tensor_data[G], 3)
                                )[jnp.newaxis, ...], reps = (10, 1, 1)), axis = 0)
                                - jnp.sum(jnp.tile(jnp.einsum(",,ij,i->ij",
                                            2.,
                                            tensor_data[D],
                                            tensor_data[B],
                                            jnp.power(tensor_data[G], 3)
                                )[jnp.newaxis, ...], reps = (10, 1, 1)), axis = 0)
        ))
        self.assertTrue(jnp.allclose(dynamics_dict3[D],
                                jnp.sum(jnp.tile(jnp.einsum(",,ij,i->",
                                            2.,
                                            tensor_data[D],
                                            tensor_data[B],
                                            jnp.power(tensor_data[G], 3)
                                )[jnp.newaxis, ...], reps = (10, 1, 1)), axis = 0)
        ))
        self.assertTrue(jnp.allclose(dynamics_dict3[G],
                                jnp.repeat(jnp.einsum(",,ij,i->",
                                            2.,
                                            tensor_data[D],
                                            tensor_data[B],
                                            jnp.power(tensor_data[G], 3)
                                ), repeats=10)
                                - 3 * jnp.sum(jnp.tile(jnp.einsum(",,ij,i->i",
                                            2.,
                                            tensor_data[D],
                                            tensor_data[B],
                                            jnp.power(tensor_data[G], 3)
                                )[jnp.newaxis, ...], reps = (10, 1)), axis = 0)
        ))

        dynamics_dict4 = self.rxn4.build_flux(None, False)(tensor_data)
        self.assertEqual(dynamics_dict4.keys(), {D, E, F})
        self.assertTrue(jnp.allclose(dynamics_dict4[D],
                                     -2 * jnp.einsum(",->",
                                            tensor_data[beta],
                                            jnp.power(tensor_data[D], 2)
                                     )
        ))
        self.assertTrue(jnp.allclose(dynamics_dict4[E],
                                     jnp.einsum(",->",
                                            tensor_data[beta],
                                            jnp.power(tensor_data[D], 2)
                                     )
        ))
        self.assertTrue(jnp.allclose(dynamics_dict4[F],
                                     jnp.einsum(",->",
                                            tensor_data[beta],
                                            jnp.power(tensor_data[D], 2)
                                     )
        ))

        dynamics_dict5 = self.rxn5.build_flux(None, False)(tensor_data)
        self.assertEqual(dynamics_dict5.keys(), {A[1,1], B[2,3], C[1,2]})
        self.assertTrue(jnp.allclose(dynamics_dict5[A[1,1]],
                                     - jnp.einsum(",,->",
                                            tensor_data[A][1,1],
                                            tensor_data[B][2,3],
                                            tensor_data[alpha][4]
                                     )
        ))
        self.assertTrue(jnp.allclose(dynamics_dict5[B[2,3]],
                                     - jnp.einsum(",,->",
                                            tensor_data[A][1,1],
                                            tensor_data[B][2,3],
                                            tensor_data[alpha][4]
                                     )
        ))
        self.assertTrue(jnp.allclose(dynamics_dict5[C[1,2]],
                                     2 * jnp.einsum(",,->",
                                            tensor_data[A][1,1],
                                            tensor_data[B][2,3],
                                            tensor_data[alpha][4]
                                     )
        ))
    
    def test_enumerate(self):
        i, j, k = many_index_symbols("i, j, k", 10)
        
        enum_list1 = self.rxn1.enumerate()
        self.assertEqual(len(enum_list1), 1000)
        for i_v in range(10):
            for j_v in range(10):
                for k_v in range(10):
                    self.assertIn(self.rxn1.index_symbols_replace({i:i_v, j:j_v, k:k_v}), enum_list1)

        enum_list2 = self.rxn2.enumerate()
        self.assertEqual(len(enum_list2), 100)
        for i_v in range(10):
            for j_v in range(10):
                self.assertIn(self.rxn2.index_symbols_replace({i:i_v, j:j_v}), enum_list2)

        enum_list3 = self.rxn3.enumerate()
        self.assertEqual(len(enum_list3), 1000)
        for i_v in range(10):
            for j_v in range(10):
                for k_v in range(10):
                    self.assertIn(self.rxn3.index_symbols_replace({i:i_v, j:j_v, k:k_v}), enum_list3)

        enum_list4 = self.rxn4.enumerate()
        self.assertEqual(enum_list4, [self.rxn4.index_symbols_replace({})])

        enum_list5 = self.rxn5.enumerate()
        self.assertEqual(enum_list5, [self.rxn5.index_symbols_replace({})])

class FastReactionTests(unittest.TestCase):
    def setUp(self):
        A, B, C, D, E, F, G = many_species("A, B, C, D, E, F, G")
        i, j = many_index_symbols("i, j", 3)

        self.rxn1 = FastReaction(A[i] + 2*B[i] + C[i], A[i] + B[i] + 3*C[i])
        self.rxn2 = FastReaction(D[i,j] + E[i,j], 0)
        self.rxn3 = FastReaction(2*F, 3*G)
        self.rxn4 = FastReaction(3*A[1] + B[2], 2*C[0])

    def test_shapes(self):
        A, B, C, D, E, F, G = many_species("A, B, C, D, E, F, G")
        self.assertEqual(self.rxn1.shapes(),
                         {
                             A : (3,),
                             B : (3,),
                             C : (3,)
                         }
        )
        self.assertEqual(self.rxn2.shapes(),
                         {
                             D : (3, 3),
                             E : (3, 3),
                         }
        )
        self.assertEqual(self.rxn3.shapes(),
                         {
                             F : (),
                             G : ()
                         }
        )
        self.assertEqual(self.rxn4.shapes(),
                         dict()
        )

    def test_get_index_symbols_set(self):
        i, j = many_index_symbols("i, j", 3)

        self.assertEqual(self.rxn1.get_index_symbols_set(), {i})
        self.assertEqual(self.rxn2.get_index_symbols_set(), {i, j})
        self.assertEqual(self.rxn3.get_index_symbols_set(), set())
        self.assertEqual(self.rxn4.get_index_symbols_set(), set())

    def test_index_symbols_replace(self):
        A, B, C, D, E, F, G = many_species("A, B, C, D, E, F, G")
        i, j = many_index_symbols("i, j", 3)

        values_dict = {i:1, j:2}
        self.assertEqual(self.rxn1.index_symbols_replace(values_dict), 
                         FastReaction(A[1] + 2*B[1] + C[1], A[1] + B[1] + 3*C[1]))
        self.assertEqual(self.rxn2.index_symbols_replace(values_dict),
                         FastReaction(D[1,2] + E[1,2], 0))
        self.assertEqual(self.rxn3.index_symbols_replace(values_dict),
                         FastReaction(2*F, 3*G))
        self.assertEqual(self.rxn4.index_symbols_replace(values_dict),
                         FastReaction(3*A[1] + B[2], 2*C[0]))

    def test_flux(self):
        A, B, C, D, E, F, G = many_species("A, B, C, D, E, F, G")
        tensor_data = {
            A : jnp.array([5.2, 1.5, 2.1]),
            B : jnp.array([1.1, 23., 1.4]),
            C : jnp.array([19., 0.3, 0.4]),
            D : jnp.array([
                    [3.2, 0.1, 1.1],
                    [1.4, 3.1, 3.2],
                    [9.3, 4.3, 8.0]
                ]),
            E : jnp.array([
                    [2.2, 1.9, 1.3],
                    [2.2, 1.2, 2.4],
                    [12., 3.1, 2.6]
                ]),
            F : jnp.array(1.3),
            G : jnp.array(1.7)
        }

        flux1 = self.rxn1.flux(tensor_data)
        self.assertEqual(flux1.keys(), {B, C})
        self.assertTrue(jnp.all(jnp.allclose(
            flux1[B],
            jnp.array([-0.55, -0.3, -0.4])
        )))
        self.assertTrue(jnp.all(jnp.allclose(
            flux1[C],
            jnp.array([1.1, 0.6, 0.8])
        )))

        flux2 = self.rxn2.flux(tensor_data)
        self.assertEqual(flux2.keys(), {D, E})
        self.assertTrue(jnp.all(jnp.allclose(
            flux2[D],
            -jnp.array([
                    [2.2, 0.1, 1.1],
                    [1.4, 1.2, 2.4],
                    [9.3, 3.1, 2.6]
            ])
        )))
        self.assertTrue(jnp.all(jnp.allclose(
            flux2[E],
            -jnp.array([
                    [2.2, 0.1, 1.1],
                    [1.4, 1.2, 2.4],
                    [9.3, 3.1, 2.6]
            ])
        )))

        flux3 = self.rxn3.flux(tensor_data)
        self.assertEqual(flux3.keys(), {F, G})
        self.assertTrue(jnp.all(jnp.allclose(
            flux3[F],
            -jnp.array(1.3)
        )))
        self.assertTrue(jnp.all(jnp.allclose(
            flux3[G],
            jnp.array(3*1.3/2)
        )))

        flux4 = self.rxn4.flux(tensor_data)
        self.assertEqual(flux4.keys(), {A[1], B[2], C[0]})
        self.assertTrue(jnp.all(jnp.allclose(
            flux4[A[1]],
            -jnp.array(1.5)
        )))
        self.assertTrue(jnp.all(jnp.allclose(
            flux4[B[2]],
            -jnp.array(1.5/3)
        )))
        self.assertTrue(jnp.all(jnp.allclose(
            flux4[C[0]],
            jnp.array(2*1.5/3)
        )))


    def test_enumerate(self):
        i, j = many_index_symbols("i, j", 3)
        
        enum_list1 = self.rxn1.enumerate()
        self.assertEqual(len(enum_list1), 3)
        for i_v in range(3):
            self.assertIn(self.rxn1.index_symbols_replace({i:i_v}), enum_list1)

        enum_list2 = self.rxn2.enumerate()
        self.assertEqual(len(enum_list2), 9)
        for i_v in range(3):
            for j_v in range(3):
                self.assertIn(self.rxn2.index_symbols_replace({i:i_v, j:j_v}), enum_list2)

        enum_list4 = self.rxn4.enumerate()
        self.assertEqual(enum_list4, [self.rxn4.index_symbols_replace({})])

        enum_list4 = self.rxn4.enumerate()
        self.assertEqual(enum_list4, [self.rxn4.index_symbols_replace({})])

class ICRNTests(unittest.TestCase):
    def setUp(self):
        A, B, C, D, E, F, G = many_species("A, B, C, D, E, F, G")
        alpha, beta, gamma = many_rate_constants("alpha, beta, gamma")
        i, j, k = many_index_symbols("i, j, k", 10)

        self.rxns = [
            BulkReaction(A[i,j]+2*B[j,k], A[i,j] + C[i,k], alpha[i]),
            BulkReaction(D + E, F, relu(gamma[i,j])),
            BulkReaction(0, B[i,j], beta),
            BulkReaction(A[1,2], 2*B[2,3], 2.),
            FastReaction(D + 2*F, 3*G),
            FastReaction(A[i,j] + C[i,j], 0),
        ]
        self.icrn = ICRN(self.rxns)

    def test_shapes(self):
        A, B, C, D, E, F, G = many_species("A, B, C, D, E, F, G")
        alpha, beta, gamma = many_rate_constants("alpha, beta, gamma")

        self.assertEqual(self.icrn.shapes(),
                         {
                             A : (10, 10),
                             B : (10, 10),
                             C : (10, 10),
                             D : (),
                             E : (),
                             F : (),
                             G : (),
                             alpha : (10,),
                             beta : (),
                             gamma : (10, 10)
                         }
        )

    def test_dynamics(self):
        A, B, C, D, E, F, G = many_species("A, B, C, D, E, F, G")
        alpha, beta, gamma = many_rate_constants("alpha, beta, gamma")
        i, j, k = many_index_symbols("i, j, k", 10)

        tensor_data = {
            A : jnp.arange(100).reshape((10,10)),
            B : 2 * jnp.arange(100).reshape((10,10)),
            C : 3 * jnp.arange(100).reshape((10,10)),
            D : jnp.array(10.1),
            E : jnp.array(11.2),
            F : jnp.array(12.3),
            G : jnp.array(12.3),
            alpha : 1.1 * jnp.arange(10),
            beta : jnp.array(10.1),
            gamma : 0.01 * jnp.arange(-50, 50, 1).reshape((10,10))
        }

        fast_f, normal_f = self.icrn.dynamics(None, False)
        fast_dynamics = fast_f(tensor_data)
        normal_dynamics = normal_f(tensor_data)

        self.assertEqual(fast_dynamics.keys(), {A, C, D, F, G})
        self.assertTrue(jnp.allclose(
            fast_dynamics[A],
            -jnp.minimum(tensor_data[A], tensor_data[C])
        ))
        self.assertTrue(jnp.allclose(
            fast_dynamics[C],
            -jnp.minimum(tensor_data[A], tensor_data[C])
        ))
        self.assertTrue(jnp.allclose(
            fast_dynamics[D],
            -jnp.minimum(tensor_data[D], tensor_data[F] / 2)
        ))
        self.assertTrue(jnp.allclose(
            fast_dynamics[F],
            -2*jnp.minimum(tensor_data[D], tensor_data[F] / 2)
        ))
        self.assertTrue(jnp.allclose(
            fast_dynamics[G],
            3 * jnp.minimum(tensor_data[D], tensor_data[F] / 2)
        ))

        self.assertEqual(normal_dynamics.keys(), {A[1,2], B, B[2,3], C, D, E, F})
        self.assertTrue(jnp.allclose(
            normal_dynamics[A[1,2]],
            -2 * tensor_data[A][1,2]
        ))
        self.assertTrue(jnp.allclose(
            normal_dynamics[B],
            -2 * jnp.einsum("ij,jk,i->jk", tensor_data[A], jnp.power(tensor_data[B], 2), tensor_data[alpha]) \
            + jnp.einsum(",i,j->ij", tensor_data[beta], jnp.ones((10,)), jnp.ones((10,)))
        ))
        self.assertTrue(jnp.allclose(
            normal_dynamics[B[2,3]],
            2 * 2 * tensor_data[A][1,2]
        ))
        self.assertTrue(jnp.allclose(
            normal_dynamics[C],
            jnp.einsum("ij,jk,i->ik", tensor_data[A], jnp.power(tensor_data[B], 2), tensor_data[alpha])
        ))
        self.assertTrue(jnp.allclose(
            normal_dynamics[D],
            -jnp.einsum("ij,,->", jax.nn.relu(tensor_data[gamma]), tensor_data[D], tensor_data[E])
        ))
        self.assertTrue(jnp.allclose(
            normal_dynamics[E],
            -jnp.einsum("ij,,->", jax.nn.relu(tensor_data[gamma]), tensor_data[D], tensor_data[E])
        ))
        self.assertTrue(jnp.allclose(
            normal_dynamics[F],
            jnp.einsum("ij,,->", jax.nn.relu(tensor_data[gamma]), tensor_data[D], tensor_data[E])
        ))
        

    def test_enumerate(self):
        enum_icrn = self.icrn.enumerate()

        self.assertIsInstance(enum_icrn, ICRN)
        self.assertEqual(len(enum_icrn.reactions), 1000 + 100 + 100 + 1 + 1 + 100)

        for i_rxn in self.rxns:
            for rxn in i_rxn.enumerate():
                self.assertIn(rxn, enum_icrn.reactions)