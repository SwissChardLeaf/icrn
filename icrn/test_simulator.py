import unittest
from icrn import many_species, many_index_symbols, many_rate_constants, MassActionReaction, ICRN, Experiment, SJDict
import jax.numpy as jnp

rxn = MassActionReaction

class ExperimentTests(unittest.TestCase):
    def setUp(self):
        A, B, C = many_species("A, B, C")
        alpha, beta = many_rate_constants("alpha, beta")
        i, j = many_index_symbols("i, j", 10)
    
        icrn = ICRN([
            rxn(A[i] + B[j], C[i,j], alpha[i,j]),
            rxn(C[i,j], A[j], beta[j])
        ])

        exp_params1 = {
            "dt" : 0.01,
            "batch" : False,
            "integration_method" : "euler",
            "spatial_dim" : None
        }

        self.exp1 = Experiment(icrn, exp_params1)

        exp_params2 = {
            "dt" : 0.01,
            "batch" : True,
            "integration_method" : "euler",
            "spatial_dim" : (100, 100),
            "spatial_rate_constant" : True,
            "dh" : 1,
            "dw" : 1
        }

        self.exp2 = Experiment(icrn, exp_params2)

    def test_simulate_segments(self):
        A, B, C = many_species("A, B, C")
        alpha, beta = many_rate_constants("alpha, beta")

        concs_data = SJDict({
            A : jnp.ones(10).astype('float32'),
            B : jnp.arange(10).astype('float32'),
            C : jnp.zeros((10,10)).astype('float32')
        })

        rate_constant_data = SJDict({
            alpha : jnp.tile(jnp.arange(10)[jnp.newaxis,...], reps=(10,1)).astype('float32'),
            beta : 2. * jnp.ones(10)
        })

        last_concs, concs_hist = self.exp1.simulate_segments(concs_data=concs_data,
                                                            rate_constant_data=rate_constant_data,
                                                            diff_data={},
                                                            segments=5,
                                                            scan_length=10)
        
        self.assertIsInstance(last_concs, SJDict)
        self.assertIsInstance(concs_hist, SJDict)

        self.assertEqual(last_concs.dict.keys(), {A, B, C})
        self.assertEqual(concs_hist.dict.keys(), {A, B, C})

        self.assertEqual(concs_hist[A].shape, (6,10))
        self.assertEqual(concs_hist[B].shape, (6,10))
        self.assertEqual(concs_hist[C].shape, (6,10,10))

        self.assertTrue(jnp.all(concs_data[A] == concs_hist[A][0]))
        self.assertTrue(jnp.all(concs_data[B] == concs_hist[B][0]))
        self.assertTrue(jnp.all(concs_data[C] == concs_hist[C][0]))

        self.assertTrue(jnp.all(last_concs[A] == concs_hist[A][-1]))
        self.assertTrue(jnp.all(last_concs[B] == concs_hist[B][-1]))
        self.assertTrue(jnp.all(last_concs[C] == concs_hist[C][-1]))

        concs_data = SJDict({
            A : jnp.ones((3,100,100,10)).astype('float32'),
            B : jnp.tile(jnp.arange(10)[jnp.newaxis, jnp.newaxis, jnp.newaxis, ...], reps=(3,100,100,1)).astype('float32'),
            C : jnp.zeros((3,100,100,10,10)).astype('float32')
        })

        rate_constant_data = SJDict({
            alpha : jnp.tile(jnp.arange(10)[jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis, ...], reps=(3,100,100,10,1)).astype('float32'),
            beta : 2. * jnp.ones((3,100,100,10)).astype('float32')
        })

        diff_data = SJDict({
            A : 0.1 * jnp.ones((3,10)).astype('float32'),
            B : jnp.zeros((3,10)).astype('float32'),
            C : 0.2 * jnp.ones((3,10,10)).astype('float32')
        })

        last_concs, concs_hist = self.exp2.simulate_segments(concs_data=concs_data,
                                                            rate_constant_data=rate_constant_data,
                                                            diff_data=diff_data,
                                                            segments=5,
                                                            scan_length=10)
        
        self.assertIsInstance(last_concs, SJDict)
        self.assertIsInstance(concs_hist, SJDict)

        self.assertEqual(last_concs.dict.keys(), {A, B, C})
        self.assertEqual(concs_hist.dict.keys(), {A, B, C})

        self.assertEqual(concs_hist[A].shape, (6,3,100,100,10))
        self.assertEqual(concs_hist[B].shape, (6,3,100,100,10))
        self.assertEqual(concs_hist[C].shape, (6,3,100,100,10,10))

        self.assertTrue(jnp.all(concs_data[A] == concs_hist[A][0]))
        self.assertTrue(jnp.all(concs_data[B] == concs_hist[B][0]))
        self.assertTrue(jnp.all(concs_data[C] == concs_hist[C][0]))

        self.assertTrue(jnp.all(last_concs[A] == concs_hist[A][-1]))
        self.assertTrue(jnp.all(last_concs[B] == concs_hist[B][-1]))
        self.assertTrue(jnp.all(last_concs[C] == concs_hist[C][-1]))

    def test_simulate_time(self):
        A, B, C = many_species("A, B, C")
        alpha, beta = many_rate_constants("alpha, beta")

        concs_data = SJDict({
            A : jnp.ones(10).astype('float32'),
            B : jnp.arange(10).astype('float32'),
            C : jnp.zeros((10,10)).astype('float32')
        })

        rate_constant_data = SJDict({
            alpha : jnp.tile(jnp.arange(10)[jnp.newaxis,...], reps=(10,1)).astype('float32'),
            beta : 2. * jnp.ones(10)
        })

        last_concs, concs_hist = self.exp1.simulate_time(concs_data=concs_data,
                                                         rate_constant_data=rate_constant_data,
                                                         diff_data={},
                                                         time=1,
                                                         sample_num=100)
        
        self.assertIsInstance(last_concs, SJDict)
        self.assertIsInstance(concs_hist, SJDict)

        self.assertEqual(last_concs.dict.keys(), {A, B, C})
        self.assertEqual(concs_hist.dict.keys(), {A, B, C})

        self.assertEqual(concs_hist[A].shape, (101,10))
        self.assertEqual(concs_hist[B].shape, (101,10))
        self.assertEqual(concs_hist[C].shape, (101,10,10))

        self.assertTrue(jnp.all(concs_data[A] == concs_hist[A][0]))
        self.assertTrue(jnp.all(concs_data[B] == concs_hist[B][0]))
        self.assertTrue(jnp.all(concs_data[C] == concs_hist[C][0]))

        self.assertTrue(jnp.all(last_concs[A] == concs_hist[A][-1]))
        self.assertTrue(jnp.all(last_concs[B] == concs_hist[B][-1]))
        self.assertTrue(jnp.all(last_concs[C] == concs_hist[C][-1]))

        concs_data = SJDict({
            A : jnp.ones((3,100,100,10)).astype('float32'),
            B : jnp.tile(jnp.arange(10)[jnp.newaxis, jnp.newaxis, jnp.newaxis, ...], reps=(3,100,100,1)).astype('float32'),
            C : jnp.zeros((3,100,100,10,10)).astype('float32')
        })

        rate_constant_data = SJDict({
            alpha : jnp.tile(jnp.arange(10)[jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis, ...], reps=(3,100,100,10,1)).astype('float32'),
            beta : 2. * jnp.ones((3,100,100,10)).astype('float32')
        })

        diff_data = SJDict({
            A : 0.1 * jnp.ones((3,10)).astype('float32'),
            B : jnp.zeros((3,10)).astype('float32'),
            C : 0.2 * jnp.ones((3,10,10)).astype('float32')
        })

        last_concs, concs_hist = self.exp2.simulate_time(concs_data=concs_data,
                                                         rate_constant_data=rate_constant_data,
                                                         diff_data=diff_data,
                                                         time=10,
                                                         sample_num=100)
        
        self.assertIsInstance(last_concs, SJDict)
        self.assertIsInstance(concs_hist, SJDict)

        self.assertEqual(last_concs.dict.keys(), {A, B, C})
        self.assertEqual(concs_hist.dict.keys(), {A, B, C})

        self.assertEqual(concs_hist[A].shape, (101,3,100,100,10))
        self.assertEqual(concs_hist[B].shape, (101,3,100,100,10))
        self.assertEqual(concs_hist[C].shape, (101,3,100,100,10,10))

        self.assertTrue(jnp.all(concs_data[A] == concs_hist[A][0]))
        self.assertTrue(jnp.all(concs_data[B] == concs_hist[B][0]))
        self.assertTrue(jnp.all(concs_data[C] == concs_hist[C][0]))

        self.assertTrue(jnp.all(last_concs[A] == concs_hist[A][-1]))
        self.assertTrue(jnp.all(last_concs[B] == concs_hist[B][-1]))
        self.assertTrue(jnp.all(last_concs[C] == concs_hist[C][-1]))


    def test_dict_builder(self):
        A, B, C = many_species("A, B, C")
        alpha, beta = many_rate_constants("alpha, beta")

        concs_spec = {
            A : 1,
            B : range(10)
        }

        rate_constant_spec = {
            alpha : jnp.arange(10),
            beta : 2.
        }

        concs_data, rate_constant_data, _ = self.exp1.dict_builder(concs_spec=concs_spec, 
                                                                   rate_constant_spec=rate_constant_spec, 
                                                                   diff_spec={})

        self.assertIsInstance(concs_data, SJDict)
        self.assertIsInstance(rate_constant_data, SJDict)

        self.assertEqual(concs_data, SJDict({
            A : jnp.ones(10),
            B : jnp.arange(10),
            C : jnp.zeros((10,10))
        }))

        self.assertEqual(rate_constant_data, SJDict({
            alpha : jnp.tile(jnp.arange(10)[jnp.newaxis,...], reps=(10,1)),
            beta : 2. * jnp.ones(10)
        }))

        concs_spec = {
            A : 1,
            B : range(10)
        }

        rate_constant_spec = {
            alpha : jnp.arange(10),
            beta : 2.
        }

        diff_spec = {
            A : 0.1,
            C : 0.2
        }

        concs_data, rate_constant_data, diff_data = self.exp2.dict_builder(concs_spec=concs_spec, 
                                                                           rate_constant_spec=rate_constant_spec, 
                                                                           diff_spec=diff_spec, 
                                                                           batch_size=3)

        self.assertIsInstance(concs_data, SJDict)
        self.assertIsInstance(rate_constant_data, SJDict)
        self.assertIsInstance(diff_data, SJDict)

        self.assertEqual(concs_data, SJDict({
            A : jnp.ones((3,100,100,10)),
            B : jnp.tile(jnp.arange(10)[jnp.newaxis, jnp.newaxis, jnp.newaxis, ...], reps=(3,100,100,1)),
            C : jnp.zeros((3,100,100,10,10))
        }))

        self.assertEqual(rate_constant_data, SJDict({
            alpha : jnp.tile(jnp.arange(10)[jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis, ...], reps=(3,100,100,10,1)),
            beta : 2. * jnp.ones((3,100,100,10))
        }))

        self.assertEqual(diff_data, SJDict({
            A : 0.1 * jnp.ones((3,10)),
            B : jnp.zeros((3,10)),
            C : 0.2 * jnp.ones((3,10,10))
        }))