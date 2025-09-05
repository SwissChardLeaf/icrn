import unittest
import os
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
from icrn import save_sjdict, load_sjdict, SJDict, Species, RateConstant, many_species, many_rate_constants, sjdict_allclose, sjdict_allequal

# SJDict as pytree
class PytreeSJDict(unittest.TestCase):
    def setUp(self):
        A, B, C = many_species("A, B, C")

        self.test_dict = SJDict({
            A : jnp.arange(5),
            B : jnp.arange(20).reshape((4,5)),
            C : jnp.ones(10)
        })

    def test_tree_flatten_unflatten_simple(self):
        value_flat, value_tree = tree_flatten(self.test_dict)
        new_dict = tree_unflatten(value_tree, value_flat)
        self.assertIsInstance(new_dict, SJDict)

        self.assertTrue(new_dict == self.test_dict)

    def test_SJDict_map(self):
        new_dict = tree_map(lambda x : 2 * x, self.test_dict)

        A, B, C = many_species("A, B, C")
        target_dict = SJDict({
            A : 2 * jnp.arange(5),
            B : 2 * jnp.arange(20).reshape((4,5)),
            C : 2 * jnp.ones(10)
        })

        self.assertTrue(new_dict == target_dict)

class SJDictWithDict(unittest.TestCase):
    def setUp(self):
        A, B, C = many_species("A, B, C")

        self.sjdict = SJDict({
            A : jnp.arange(5),
            B : jnp.arange(20).reshape((4,5)),
            C : jnp.ones(10)
        })

        self.d = {
            A : jnp.ones(5),
            B[1,2] : 3,
            B : jnp.arange(20).reshape((4,5))
        }

    def test_zeros(self):
        A, B, C = many_species("A, B, C")
        self.assertEqual(self.sjdict.zeros(), SJDict({
            A : jnp.zeros(5),
            B : jnp.zeros((4,5)),
            C : jnp.zeros(10)
        }))

    def test_add_with_dict(self):
        A, B, C = many_species("A, B, C")
        self.assertEqual(self.sjdict.add_with_dict(self.d), SJDict({
            A : jnp.arange(5) + jnp.ones(5),
            B : (2 * jnp.arange(20).reshape((4,5))).at[1,2].add(3),
            C : jnp.ones(10)
        }))

    def test_init_with_dict(self):
        A, B, C = many_species("A, B, C")
        self.assertEqual(self.sjdict.init_with_dict(self.d), SJDict({
            A : jnp.ones(5),
            B : jnp.arange(20).reshape((4,5)).at[1,2].add(3),
            C : jnp.zeros(10)
        }))

# overloaded operators
class SJDictOp(unittest.TestCase):
    def setUp(self):
        A, B = many_species("A, B")
        
        self.test_dict1 = SJDict({
        A : jnp.arange(5),
        B : jnp.arange(20).reshape((4,5))
        })

        self.test_dict2 = SJDict({
            A : jnp.ones(5),
            B : jnp.array([
                    [1, 1, 1, 0.4, 1],
                    [3.9, 3, 5.2, 9, 1],
                    [1, 3.2, 0, 2, 1],
                    [4., 3., 8, 7, 1],
                ])
        })
    
        self.test_dict3 = SJDict({
            A : 1,
            B : 2 * jnp.ones(5)
        })

    def test_bin_op_helper(self):
        pass

    def test_add(self):
        A, B = many_species("A, B")

        target_dict1 = SJDict({
            A : jnp.arange(5) + jnp.ones(5),
            B : jnp.array([
                    [1, 1, 1, 0.4, 1],
                    [3.9, 3, 5.2, 9, 1],
                    [1, 3.2, 0, 2, 1],
                    [4., 3., 8, 7, 1],
                ]) + jnp.arange(20).reshape((4,5))
        })

        self.assertTrue(self.test_dict1 + self.test_dict2 == target_dict1)
        self.assertTrue(self.test_dict2 + self.test_dict1 == target_dict1)

        target_dict2 = SJDict({
            A : jnp.arange(5) + 1,
            B : jnp.arange(20).reshape((4,5)) + 2 * jnp.ones(5)
        })

        self.assertTrue(self.test_dict1 + self.test_dict3 == target_dict2)
        self.assertTrue(self.test_dict3 + self.test_dict1 == target_dict2)

        target_dict3 = SJDict({
            A : 3 + jnp.arange(5),
            B : 3 + jnp.arange(20).reshape((4,5))
        })

        self.assertTrue(3 + self.test_dict1 == target_dict3)
        self.assertTrue(self.test_dict1 + 3 == target_dict3)

        target_dict4 = SJDict({
            A : jnp.arange(5) + jnp.ones(5),
            B : jnp.arange(20).reshape((4,5)) + jnp.ones(5)
        })

        self.assertTrue(self.test_dict1 + jnp.ones(5) == target_dict4)
        self.assertTrue(jnp.ones(5) + self.test_dict1 == target_dict4)

    def test_neg(self):
        A, B = many_species("A, B")

        target_dict1 = SJDict({
            A : -jnp.arange(5),
            B : -jnp.arange(20).reshape((4,5))
        })

        self.assertTrue(-self.test_dict1 == target_dict1)

        target_dict2 = SJDict({
            A : -jnp.ones(5),
            B : -jnp.array([
                    [1, 1, 1, 0.4, 1],
                    [3.9, 3, 5.2, 9, 1],
                    [1, 3.2, 0, 2, 1],
                    [4., 3., 8, 7, 1],
                ])
        })

        self.assertTrue(-self.test_dict2 == target_dict2)

        target_dict3 = SJDict({
            A : -1,
            B : -2 * jnp.ones(5)
        })

        self.assertTrue(-self.test_dict3 == target_dict3)

    def test_sub(self):
        A, B = many_species("A, B")

        target_dict1 = SJDict({
            A : jnp.arange(5) - jnp.ones(5),
            B : jnp.arange(20).reshape((4,5)) - jnp.array([
                    [1, 1, 1, 0.4, 1],
                    [3.9, 3, 5.2, 9, 1],
                    [1, 3.2, 0, 2, 1],
                    [4., 3., 8, 7, 1],
                ])
        })

        self.assertTrue(self.test_dict1 - self.test_dict2 == target_dict1)
        self.assertTrue(-(self.test_dict2 - self.test_dict1) == target_dict1)

        target_dict2 = SJDict({
            A : 1 - jnp.arange(5),
            B : 2 * jnp.ones(5) - jnp.arange(20).reshape((4,5))
        })

        self.assertTrue(self.test_dict3 - self.test_dict1 == target_dict2)
        self.assertTrue(-(self.test_dict1 - self.test_dict3) == target_dict2)

        target_dict3 = SJDict({
            A : jnp.arange(5) - 3,
            B : jnp.arange(20).reshape((4,5)) - 3
        })

        self.assertTrue(self.test_dict1 - 3 == target_dict3)
        self.assertTrue(-(3 - self.test_dict1) == target_dict3)

        target_dict4 = SJDict({
            A : jnp.arange(5) - jnp.ones(5),
            B : jnp.arange(20).reshape((4,5)) - jnp.ones(5)
        })

        self.assertTrue(self.test_dict1 - jnp.ones(5) == target_dict4)
        self.assertTrue(-(jnp.ones(5) - self.test_dict1) == target_dict4)

    def test_mul(self):
        A, B = many_species("A, B")

        target_dict1 = SJDict({
            A : jnp.arange(5) * jnp.ones(5),
            B : jnp.arange(20).reshape((4,5)) * jnp.array([
                    [1, 1, 1, 0.4, 1],
                    [3.9, 3, 5.2, 9, 1],
                    [1, 3.2, 0, 2, 1],
                    [4., 3., 8, 7, 1],
                ])
        })

        self.assertTrue(self.test_dict1 * self.test_dict2 == target_dict1)
        self.assertTrue(self.test_dict2 * self.test_dict1 == target_dict1)

        target_dict2 = SJDict({
            A : 1 * jnp.arange(5),
            B : 2 * jnp.ones(5) * jnp.arange(20).reshape((4,5))
        })

        self.assertTrue(self.test_dict3 * self.test_dict1 == target_dict2)
        self.assertTrue(self.test_dict1 * self.test_dict3 == target_dict2)

        target_dict3 = SJDict({
            A : jnp.arange(5) * 3,
            B : jnp.arange(20).reshape((4,5)) * 3
        })

        self.assertTrue(self.test_dict1 * 3 == target_dict3)
        self.assertTrue(3 * self.test_dict1 == target_dict3)

        target_dict4 = SJDict({
            A : jnp.arange(5) * jnp.ones(5),
            B : jnp.arange(20).reshape((4,5)) * jnp.ones(5)
        })

        self.assertTrue(self.test_dict1 * jnp.ones(5) == target_dict4)
        self.assertTrue(jnp.ones(5) * self.test_dict1 == target_dict4)

    def test_div(self):
        A, B = many_species("A, B")

        target_dict1 = SJDict({
            A : jnp.arange(5) / jnp.ones(5),
            B : jnp.arange(20).reshape((4,5)) / jnp.array([
                    [1, 1, 1, 0.4, 1],
                    [3.9, 3, 5.2, 9, 1],
                    [1, 3.2, 0, 2, 1],
                    [4., 3., 8, 7, 1],
                ])
        })

        self.assertTrue(self.test_dict1 / self.test_dict2 == target_dict1)
        self.assertTrue(sjdict_allclose(1 / (self.test_dict2 / self.test_dict1), target_dict1))

        target_dict2 = SJDict({
            A : 1 / jnp.arange(5),
            B : 2 * jnp.ones(5) / jnp.arange(20).reshape((4,5))
        })

        self.assertTrue(self.test_dict3 / self.test_dict1 == target_dict2)
        self.assertTrue(sjdict_allclose(1 / (self.test_dict1 / self.test_dict3), target_dict2))

        target_dict3 = SJDict({
            A : jnp.arange(5) / 3,
            B : jnp.arange(20).reshape((4,5)) / 3
        })

        self.assertTrue(self.test_dict1 / 3 == target_dict3)
        self.assertTrue(sjdict_allclose(1 / (3 / self.test_dict1), target_dict3))

        target_dict4 = SJDict({
            A : jnp.arange(5) / jnp.ones(5),
            B : jnp.arange(20).reshape((4,5)) / jnp.ones(5)
        })

        self.assertTrue(self.test_dict1 / jnp.ones(5) == target_dict4)
        self.assertTrue(sjdict_allclose(1 / (jnp.ones(5) / self.test_dict1), target_dict4))

    def test_pow(self):
        A, B = many_species("A, B")

        target_dict1 = SJDict({
            A : jnp.arange(5) ** jnp.ones(5),
            B : jnp.arange(20).reshape((4,5)) ** jnp.array([
                    [1, 1, 1, 0.4, 1],
                    [3.9, 3, 5.2, 9, 1],
                    [1, 3.2, 0, 2, 1],
                    [4., 3., 8, 7, 1],
                ])
        })

        self.assertTrue(self.test_dict1 ** self.test_dict2 == target_dict1)

        target_dict2 = SJDict({
            A : 1 ** jnp.arange(5),
            B : (2 * jnp.ones(5)) ** jnp.arange(20).reshape((4,5))
        })

        self.assertTrue(self.test_dict3 ** self.test_dict1 == target_dict2)

        target_dict3 = SJDict({
            A : jnp.arange(5) ** 3,
            B : jnp.arange(20).reshape((4,5)) ** 3
        })

        self.assertTrue(self.test_dict1 ** 3 == target_dict3)
                        
        target_dict4 = SJDict({
            A : jnp.ones(5) ** jnp.arange(5),
            B : jnp.ones(5) ** jnp.arange(20).reshape((4,5))
        })

        self.assertTrue(jnp.ones(5) ** self.test_dict1 == target_dict4)

class SJDictDictMethods(unittest.TestCase):
    def setUp(self):
        A, B = many_species("A, B")
        
        self.test_dict1 = SJDict({
            A : jnp.arange(5),
            B : jnp.arange(20).reshape((4,5))
        })
    
    def test_setitem(self):
        A, B = many_species("A, B")

        self.test_dict1[A] = jnp.ones(5)

        target_dict1 = SJDict({
            A : jnp.ones(5),
            B : jnp.arange(20).reshape((4,5))
        })

        self.assertEqual(self.test_dict1, target_dict1)

        self.test_dict1[A[2]] = 0
        target_dict2 = SJDict({
            A : jnp.array([1, 1, 0, 1, 1]),
            B : jnp.arange(20).reshape((4,5))
        })

        self.assertEqual(self.test_dict1, target_dict2)

    def test_getitem(self):
        A, B = many_species("A, B")

        self.assertTrue(jnp.all(self.test_dict1[A] == jnp.arange(5)))
        self.assertEqual(self.test_dict1[B[1,2]], 7)


    def test_add_with_dict(self):
        A, B = many_species("A, B")

        self.test_dict1.add_with_dict({
            A : jnp.ones(5),
            A[2] : -0.5,
            B : jnp.ones(20).reshape((4,5)),
            B[1,2] : 5,
            B[2,3] : -3
        })

        target_dict = SJDict({
            A : jnp.array([1, 2, 2.5, 4, 5]),
            B : jnp.array([
                [1, 2, 3, 4, 5],
                [6, 7, 13, 9, 10],
                [11, 12, 13, 11, 15],
                [16, 17, 18, 19, 20],
            ])
        })

        self.assertEqual(self.test_dict1, target_dict)

    def test_and(self):
        A, B = many_species("A, B")

        extend_dict = SJDict({
            A : jnp.ones(5),
            B : jnp.ones((4,5))
        })

        target_dict = SJDict({
            A : jnp.stack([jnp.arange(5), jnp.ones(5)]),
            B : jnp.stack([jnp.arange(20).reshape((4,5)), jnp.ones((4,5))])
        })
        self.assertEqual(self.test_dict1 & extend_dict, target_dict)


# load/save SJDict
class SJDictLoad(unittest.TestCase):
    def test_load_save_sjdict_species(self):
        A, B, C = many_species("A, B, C")

        test_dict = SJDict({
            A : jnp.arange(5),
            B : jnp.arange(20).reshape((4,5)),
            C : jnp.ones(10)
        })

        save_path = os.path.join("test", "test_load_save_sjdict")
        save_sjdict(test_dict, save_path)
        loaded_dict = load_sjdict(save_path, Species)

        self.assertTrue(test_dict == loaded_dict)