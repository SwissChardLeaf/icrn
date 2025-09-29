import unittest
import icrn.numerics as numerics
import jax
from jax import numpy as jnp
from icrn.dict_utils import load_sjdict, SJDict, sjdict_allclose, load_dict_yaml
from icrn.representation import relu, many_species, many_index_symbols, many_rate_constants, BulkReaction, FastReaction, ICRN
import os
import numpy as np

class NumericsFunctions(unittest.TestCase):

    def setUp(self):
        A, B, C, D, E, F, G = many_species("A, B, C, D, E, F, G")
        alpha, beta, gamma = many_rate_constants("alpha, beta, gamma")
        i, j, k = many_index_symbols("i, j, k", 10)

        self.rxns1 = [
            BulkReaction(A[i,j]+2*B[j,k], A[i,j] + C[i,k], alpha[i]),
            BulkReaction(D + E, F, relu(gamma[i,j])),
            BulkReaction(0, B[i,j], beta),
            BulkReaction(A[1,2], 2*B[2,3], 2.),
            FastReaction(D + 2*F, 3*G),
            FastReaction(A[i,j] + C[i,j], 0),
        ]
        self.icrn1 = ICRN(self.rxns1)

        self.conc_data = SJDict({
            A : jnp.arange(100).reshape((10,10)),
            B : 2 * jnp.arange(100).reshape((10,10)),
            C : 3 * jnp.arange(100).reshape((10,10)),
            D : jnp.array(10.1),
            E : jnp.array(11.2),
            F : jnp.array(12.3),
            G : jnp.array(12.3),
        })

        self.rate_constant_data = SJDict({
            alpha : 1.1 * jnp.arange(10),
            beta : jnp.array(10.1),
            gamma : 0.01 * jnp.arange(-50, 50, 1).reshape((10,10))
        })

        self.rxns2 = [
            BulkReaction(A[i,j]+2*B[j,k], A[i,j] + C[i,k], alpha[i]),
            BulkReaction(D + E, F, relu(gamma[i,j])),
            FastReaction(D + 2*F, 3*G)
        ]
        self.icrn2 = ICRN(self.rxns2)

    def test_compute_lap_op(self):
        computed_lap_op1 = numerics.compute_lap_op((4,3), 1, 1)
        
        target_lap_op1 = jnp.array([
            [ -0.       ,  -4.3864913,  -4.3864913],
            [ -2.4674013,  -6.8538923,  -6.8538923],
            [ -9.869605 , -14.256096 , -14.256096 ],
            [ -2.4674013,  -6.8538923,  -6.8538923]
        ])

        self.assertTrue(jnp.allclose(computed_lap_op1, target_lap_op1))

        computed_lap_op2 = numerics.compute_lap_op((2,3), 0.5, 2.0)

        target_lap_op2 = jnp.array([
            [ -0.       ,  -1.0966228,  -1.0966228],
            [-39.47842  , -40.575043 , -40.575043 ]
        ])

        self.assertTrue(jnp.allclose(computed_lap_op2, target_lap_op2))

        computed_lap_op3 = numerics.compute_lap_op((1,4), 1, 1)
        target_lap_op3 = jnp.array([[-0.       , -2.4674013, -9.869605 , -2.4674013]])
        self.assertTrue(jnp.allclose(computed_lap_op3, target_lap_op3))

    def test_spectral_per_species_diffuse(self):
        lap_op = numerics.compute_lap_op((5,5), 1, 1)

        initial_state1 = jnp.array([
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
        ])
        target_state1 = jnp.array([
            [0.03760443, 0.03843227, 0.03896552, 0.03843227, 0.03760443],
            [0.03843226, 0.0406484,  0.04309084, 0.0406484,  0.03843226],
            [0.03896552, 0.04309085, 0.05130513, 0.04309085, 0.03896552],
            [0.03843226, 0.0406484,  0.04309084, 0.0406484,  0.03843226],
            [0.03760443, 0.03843226, 0.03896552, 0.03843226, 0.03760443]
        ])

        computed_state1 = numerics._spectral_species_diffuse(initial_state1, kd=jnp.array(10.0), lap_op=lap_op, dt=2.0)
        self.assertTrue(jnp.allclose(computed_state1, target_state1))

        initial_state2 = jnp.repeat(initial_state1[..., jnp.newaxis], repeats=3, axis=-1)

        target_state2 = jnp.stack([
            numerics._spectral_species_diffuse(initial_state1, kd=jnp.array(1), lap_op=lap_op, dt=2.0),
            numerics._spectral_species_diffuse(initial_state1, kd=jnp.array(2), lap_op=lap_op, dt=2.0),
            numerics._spectral_species_diffuse(initial_state1, kd=jnp.array(10), lap_op=lap_op, dt=2.0)
        ], axis=-1)

        computed_state2 = numerics._spectral_species_diffuse(initial_state2, kd=jnp.array([1, 2, 10]), lap_op=lap_op, dt=2.0)
        self.assertTrue(jnp.allclose(computed_state2, target_state2))
        self.assertTrue(jnp.allclose(computed_state2[..., 2], target_state1))

    def test_spectral_diffuse(self):
        lap_op = numerics.compute_lap_op((5,5), 1, 1)

        initial_state = jnp.array([
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
        ])

        target_state = jnp.array([
            [0.03760443, 0.03843227, 0.03896552, 0.03843227, 0.03760443],
            [0.03843226, 0.0406484,  0.04309084, 0.0406484,  0.03843226],
            [0.03896552, 0.04309085, 0.05130513, 0.04309085, 0.03896552],
            [0.03843226, 0.0406484,  0.04309084, 0.0406484,  0.03843226],
            [0.03760443, 0.03843226, 0.03896552, 0.03843226, 0.03760443]
        ])

        A, B, C = many_species("A, B, C")

        initial_state_sjdict = SJDict({
            A : initial_state,
            B : jnp.tile(initial_state[..., jnp.newaxis], (1, 1, 3)),
            C : jnp.tile(initial_state[..., jnp.newaxis, jnp.newaxis], (1, 1, 2, 3))
        })

        target_state_sjdict = SJDict({
            A : target_state,
            B: jnp.stack([
                    numerics._spectral_species_diffuse(initial_state, kd=jnp.array(1), lap_op=lap_op, dt=2.0),
                    numerics._spectral_species_diffuse(initial_state, kd=jnp.array(2), lap_op=lap_op, dt=2.0),
                    numerics._spectral_species_diffuse(initial_state, kd=jnp.array(10), lap_op=lap_op, dt=2.0)
                ], axis=-1),
            C : jnp.tile(target_state[..., jnp.newaxis, jnp.newaxis], (1, 1, 2, 3))
        })

        kd_sjdict = SJDict({
            A : jnp.array(10),
            B : jnp.array([1, 2, 10]),
            C : jnp.array([
                [10, 10, 10],
                [10, 10, 10]
            ])
        })

        computed_state = numerics.spectral_diffuse(initial_state_sjdict, kd_sjdict, lap_op, dt=2.0)
        self.assertTrue(sjdict_allclose(computed_state, target_state_sjdict))

    def test_conv_species_diffuse(self):
        # scalar species
        initial_state = jnp.array([
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
        ])

        target_state = jnp.array([
            [0.        , 0.        , 0.        , 0.        , 0.        ],
            [0.        , 0.01666667, 0.06666667, 0.01666667, 0.        ],
            [0.        , 0.06666667, 0.6666666 , 0.06666667, 0.        ],
            [0.        , 0.01666667, 0.06666667, 0.01666667, 0.        ],
            [0.        , 0.        , 0.        , 0.        , 0.        ]
        ])

        computed_state = numerics._conv_species_diffuse(initial_state, jnp.array(1.), dt=0.1, dh=1, dw=1)
        self.assertTrue(jnp.allclose(computed_state, target_state))
        
        # indexed species
        init_small = np.zeros((5,5,3))
        init_small[1,1,0] = 1
        init_small[2,2,1] = 1
        init_small[3,3,2] = 1

        initial_state = jnp.array(init_small)

        target_state_index0 = jnp.array([
            [0.01666667, 0.06666667, 0.01666667, 0.        , 0.        ],
            [0.06666667, 0.6666666 , 0.06666667, 0.        , 0.        ],
            [0.01666667, 0.06666667, 0.01666667, 0.        , 0.        ],
            [0.        , 0.        , 0.        , 0.        , 0.        ],
            [0.        , 0.        , 0.        , 0.        , 0.        ]
        ])

        target_state_index1 = jnp.array([
            [0.        , 0.        , 0.        , 0.        , 0.        ],
            [0.        , 0.03333334, 0.13333334, 0.03333334, 0.        ],
            [0.        , 0.13333334, 0.3333333 , 0.13333334, 0.        ],
            [0.        , 0.03333334, 0.13333334, 0.03333334, 0.        ],
            [0.        , 0.        , 0.        , 0.        , 0.        ]
        ])

        target_state_index2 = jnp.array([
            [0.  , 0.  , 0.  , 0.  , 0.  ],
            [0.  , 0.  , 0.  , 0.  , 0.  ],
            [0.  , 0.  , 0.05, 0.2 , 0.05],
            [0.  , 0.  , 0.2 , 0.  , 0.2 ],
            [0.  , 0.  , 0.05, 0.2 , 0.05]
        ])

        target_state = jnp.stack(
            [target_state_index0, target_state_index1, target_state_index2],
            axis=-1
        )

        computed_state = numerics._conv_species_diffuse(initial_state, jnp.array([1., 2., 3.]), dt=0.1, dh=1, dw=1)
        self.assertTrue(jnp.allclose(computed_state, target_state))

        # indexed species with spatially varying rate constant
        init_state = np.zeros((5,5,3))
        init_state[1:4,1,0] = 1
        init_state[1:4,2,1] = 1
        init_state[1:4,3,2] = 1
        initial_state = jnp.array(init_state)

        diff_const = np.zeros((5,5,3))
        diff_const[1,1:4,0] = np.arange(3)/2
        diff_const[2,1:4,1] = np.arange(3)/2
        diff_const[3,1:4,2] = np.arange(3)/2
        diff_constant = jnp.array(diff_const)

        target_state_index0 = jnp.array([
            [0.        , 0.        , 0.        , 0.        , 0.        ],
            [0.        , 1.        , 0.04166667, 0.        , 0.        ],
            [0.        , 1.        , 0.        , 0.        , 0.        ],
            [0.        , 1.        , 0.        , 0.        , 0.        ],
            [0.        , 0.        , 0.        , 0.        , 0.        ]
        ])

        target_state_index1 = jnp.array([
            [0.        , 0.        , 0.        , 0.        , 0.        ],
            [0.        , 0.        , 1.        , 0.        , 0.        ],
            [0.        , 0.        , 0.90000004, 0.1       , 0.        ],
            [0.        , 0.        , 1.        , 0.        , 0.        ],
            [0.        , 0.        , 0.        , 0.        , 0.        ]
        ])

        target_state_index2 = jnp.array([
            [0.        , 0.        , 0.        , 0.        , 0.        ],
            [0.        , 0.        , 0.        , 1.        , 0.        ],
            [0.        , 0.        , 0.        , 1.        , 0.        ],
            [0.        , 0.        , 0.04166667, 0.73333335, 0.        ],
            [0.        , 0.        , 0.        , 0.        , 0.        ]
        ])

        target_state = jnp.stack(
            [target_state_index0, target_state_index1, target_state_index2],
            axis=-1
        )

        computed_state = numerics._conv_species_diffuse(initial_state, diff_constant, dt=0.1, dh=1, dw=1)
        self.assertTrue(jnp.allclose(computed_state, target_state))

    def test_fast_react(self):
        A, B, C, D, E, F, G = many_species("A, B, C, D, E, F, G")

        fast_dynamics, _ = self.icrn1.dynamics(None, False)
        computed_state = numerics.fast_react(self.conc_data, fast_dynamics)
        
        target_state = SJDict({
            A : jnp.zeros((10, 10)),
            B : 2 * jnp.arange(100).reshape((10,10)),
            C : 2 * jnp.arange(100).reshape((10,10)),
            D : jnp.array(10.1 - 12.3/2),
            E : jnp.array(11.2),
            F : jnp.array(0),
            G : jnp.array(12.3 + 3 * 12.3/2)
        })

        self.assertTrue(sjdict_allclose(computed_state, target_state))

    def test_euler(self):
        A, B, C, D, E, F, G = many_species("A, B, C, D, E, F, G")

        _, normal_dynamics = self.icrn2.dynamics(None, False)
        computed_state = numerics.euler(self.conc_data, self.rate_constant_data, 1.5, normal_dynamics)
        
        target_state = SJDict({
            A : jnp.arange(100).reshape((10,10)),
            B : 2 * jnp.arange(100).reshape((10,10)) - 2 * 1.5 * jnp.einsum("ij,jk,i->jk", 
                                                                            jnp.arange(100).reshape((10,10)),
                                                                            (2 * jnp.arange(100).reshape((10,10))) ** 2,
                                                                            1.1 * jnp.arange(10)),
            C : 3 * jnp.arange(100).reshape((10,10)) +  1.5 * jnp.einsum("ij,jk,i->ik", 
                                                                            jnp.arange(100).reshape((10,10)),
                                                                            (2 * jnp.arange(100).reshape((10,10))) ** 2,
                                                                            1.1 * jnp.arange(10)),
            D : jnp.array(10.1) - 1.5 * 10.1 * 11.2 * jnp.sum(jax.nn.relu(0.01 * jnp.arange(-50, 50, 1).reshape((10,10)))),
            E : jnp.array(11.2) - 1.5 * 10.1 * 11.2 * jnp.sum(jax.nn.relu(0.01 * jnp.arange(-50, 50, 1).reshape((10,10)))),
            F : jnp.array(12.3) + 1.5 * 10.1 * 11.2 * jnp.sum(jax.nn.relu(0.01 * jnp.arange(-50, 50, 1).reshape((10,10)))),
            G : jnp.array(12.3),
        })

        self.assertTrue(sjdict_allclose(computed_state, target_state))

    def test_RK4(self):
        A, B = many_species("A, B")
        alpha, beta = many_rate_constants("alpha, beta")
        i, j = many_index_symbols("i, j", 3)

        test_icrn = ICRN([
            BulkReaction(A[i,j]+2*B[j], A[i,j], alpha[i]),
            BulkReaction(A[i,j], B[i], relu(beta[i]))
        ])

        conc_data = SJDict({
            A : jnp.arange(9).reshape((3,3)),
            B : jnp.arange(3)
        })

        rate_constant_data = SJDict({
            alpha : jnp.arange(3),
            beta : -jnp.arange(3),
        })

        _, normal_dynamics = test_icrn.dynamics(None, False)
        computed_state = numerics.RK4(conc_data, rate_constant_data, 1.5, normal_dynamics)

        def A_dynamics(A_concs):
            return - jnp.einsum("ij,i->ij", A_concs, jax.nn.relu(rate_constant_data[beta]))
        
        def B_dynamics(A_concs, B_concs):
            return jnp.einsum("ij,i->i", A_concs, jax.nn.relu(rate_constant_data[beta])) \
                   - 2 * jnp.einsum("ij,j,i->j", A_concs, B_concs ** 2, rate_constant_data[alpha])
        
        def net_dynamics(conc):
            return SJDict({
                A : A_dynamics(conc[A]), 
                B : B_dynamics(conc[A], conc[B])
            })
        
        k1 = net_dynamics(conc_data)
        k2 = net_dynamics(conc_data + k1 * 1.5 * 0.5)
        k3 = net_dynamics(conc_data + k2 * 1.5 * 0.5)
        k4 = net_dynamics(conc_data + k3 * 1.5)

        target = 1.5 * (k1 + 2*k2 + 3*k3 + k4) / 6
        self.assertTrue(sjdict_allclose(computed_state - conc_data, target))