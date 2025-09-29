import unittest
from icrn import many_species, many_rate_constants, many_index_symbols, MassActionReaction, ICRN, FastReaction, Experiment, relu, SJDict, map1
rxn = MassActionReaction
import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax.random as jax_random

def max_abs_error(a, b):
    return jnp.max(jnp.abs(a-b))

def max_rel_error(a, b):
    return jnp.max(jnp.abs(a-b) / b)

class EndToEnd(unittest.TestCase):
    def test_dimerization_network(self):
        M, D = many_species("M, D")
        K_1, K_2 = many_rate_constants("K_1, K_2")

        n = 10

        i, j = many_index_symbols("i, j", n)

        dn_crn = ICRN([
            rxn(M[i] + M[j], D[i,j], K_1[i,j]),
            rxn(D[i,j], M[i] + M[j], K_2[i,j])
        ])

        dn_exp_params = {
            "dt" : 1e-5,
            "batch" : False,
            "integration_method" : "euler",
            "spatial_dim" : None
        }

        dn_exp = Experiment(dn_crn, dn_exp_params)

        test_path = os.path.join("test", "dimerization")

        init_M_path = os.path.join(test_path, "init_M.npy")
        init_D_path = os.path.join(test_path, "init_D.npy")

        conc_data = SJDict({M: jnp.load(init_M_path), D: jnp.load(init_D_path)})

        K_1_path = os.path.join(test_path, "K_1.npy")
        K_2_path = os.path.join(test_path, "K_2.npy")

        rate_data = SJDict({K_1: jnp.load(K_1_path), K_2: jnp.load(K_2_path)})

        sim_concs, _ = dn_exp.simulate_time(conc_data, rate_data, {}, time=1.0)

        target_M_path = os.path.join(test_path, "target_M.npy")
        target_D_path = os.path.join(test_path, "target_D.npy")

        target_M = jnp.load(target_M_path)
        target_D = jnp.load(target_D_path)

        M_max_rel_error = max_rel_error(sim_concs[M], target_M)
        D_max_rel_error = max_rel_error(sim_concs[D], target_D)

        self.assertTrue(M_max_rel_error < 0.01)
        self.assertTrue(D_max_rel_error < 0.01)

    def test_winner_take_all(self):
        X, W, XF, P, S, SG, A, RG, YF, Y, Rep, F = many_species("X, W, XF, P, S, SG, A, RG, YF, Y, Rep, F")

        alpha = many_rate_constants("alpha")

        n = 100
        m = 3

        i = many_index_symbols("i", n)
        j, k = many_index_symbols("j, k", m)

        rxn = MassActionReaction

        wta_crn = ICRN([
            rxn(X[i] + W[i,j] + XF[i], X[i] + P[i,j], 36.),
            rxn(P[i,j] + SG[j], S[j], 36.),
            rxn(S[j] + S[k] + A[j,k], 0, alpha[j,k]),
            rxn(S[j] + RG[j] + YF[k], S[j] + Y[j], 1.8e-4),
            rxn(Y[j] + Rep[j], F[j], 3.6)
        ])

        wta_exp_params = {
            "dt" : 1e-4,
            "batch" : True,
            "integration_method" : "euler",
            "spatial_dim" : None
        }

        wta_exp = Experiment(wta_crn, wta_exp_params)

        test_dir = os.path.join("test", "winner_take_all")
        img_batch_path = os.path.join(test_dir, "img_batch.npy")
        avg_img = os.path.join(test_dir, "avg_img.npy")

        img_batch = jnp.load(img_batch_path)
        avg_img = jnp.load(avg_img)

        concs_spec = {
            X : 5. * img_batch,
            W : 100. * avg_img,
            XF : 2. * np.sum(100. * avg_img, axis=-1),
            SG : 100.,
            RG : 100.,
            A : 400.,
            YF :  200.,
            Rep : 200.,
            # below should start at zero
            P : 0.,
            S : 0.,
            Y : 0.,
            F : 0.
        }

        rate_data_spec = {
            alpha : 3.6e-3 * (np.ones((m,m)) - np.identity(m))
        }

        concs, rate_data, diff_data = wta_exp.dict_builder(concs_spec, rate_data_spec, {}, 9)
        sim_concs, _ = wta_exp.simulate_time(concs, rate_data, diff_data, 10.)

        target_F_path = os.path.join(test_dir, "target_F.npy")
        target_F = jnp.load(target_F_path)

        F_max_rel_error = max_rel_error(sim_concs[F], target_F)

        self.assertTrue(F_max_rel_error < 0.01)

    def test_gray_scott(self):
        U, V = many_species("U, V")
        F, k = many_rate_constants("F, k")

        gs_crn = ICRN([
            rxn(U + 2*V, 3*V, 1),
            rxn(V, 0, F+k),
            rxn(0, U, F),
            rxn(U, 0, F)
        ])

        gs_exp_params = {
            "dt" : 1,
            "dh" : 1,
            "batch" : False,
            "integration_method" : "relu_RK4",
            "spatial_dim" : (101,101)
        }

        gs_exp = Experiment(gs_crn, gs_exp_params)

        test_dir = os.path.join("test", "gray_scott")

        init_U_path = os.path.join(test_dir, "init_U.npy")
        init_V_path = os.path.join(test_dir, "init_V.npy")

        init_concs = SJDict({U: jnp.load(init_U_path), V: jnp.load(init_V_path)})

        rate_data = SJDict({
            F : 0.037,
            k : 0.06
        })

        diff_data_spec = {
            U : 0.2,
            V : 0.1
        }

        _, _, diff_data = gs_exp.dict_builder({}, {}, diff_data_spec)
        sim_concs, _ = gs_exp.simulate_time(init_concs, rate_data, diff_data, time=5000)

        target_U_path = os.path.join(test_dir, "target_U.npy")
        target_V_path = os.path.join(test_dir, "target_V.npy")

        target_U = jnp.load(target_U_path)
        target_V = jnp.load(target_V_path)

        def normalise(channel):
            return (channel - channel.min()) / (channel.max() - channel.min())

        r = normalise(sim_concs[V])
        g = normalise(sim_concs[U])
        b = 1 - (r + g)/2
        img = jnp.stack([r, g, b], axis=-1)

        img_path = os.path.join(test_dir, "pink_maze_on_green.png")
        plt.imsave(img_path, img)

        U_max_rel_error = max_rel_error(sim_concs[U], target_U)
        V_max_rel_error = max_rel_error(sim_concs[V], target_V)


    def test_spatial_gray_scott(self):
        U, V = many_species("U, V")
        F, k = many_rate_constants("F, k")

        gs_crn = ICRN([
            rxn(U + 2*V, 3*V, 1),
            rxn(V, 0, F+k),
            rxn(0, U, F),
            rxn(U, 0, F)
        ])

        gs_exp_params = {
            "dt" : 1,
            "dh" : 1,
            "batch" : False,
            "integration_method" : "relu_RK4",
            "spatial_dim" : (1000, 1000),
            "spatial_rate_constant" : True
        }

        gs_exp = Experiment(gs_crn, gs_exp_params)

        test_dir = os.path.join("test", "gray_scott")

        key = jax_random.key(12)
    
        init_concs = SJDict({
            U: jnp.zeros((1000, 1000)), 
            V: 0.9 + 0.1 * jax_random.uniform(key, ((1000,1000)))

        })

        rate_data = SJDict({
            F : jnp.broadcast_to(jnp.linspace(0.08, 0.01, num=1000)[..., jnp.newaxis], (1000, 1000)),
            k : jnp.broadcast_to(jnp.linspace(0.03, 0.07, num=1000)[jnp.newaxis, ...], (1000, 1000))
        })

        diff_data_spec = {
            U : 0.2,
            V : 0.1
        }

        _, _, diff_data = gs_exp.dict_builder({}, {}, diff_data_spec)
        sim_concs, _ = gs_exp.simulate_time(init_concs, rate_data, diff_data, time=5000)

        def normalise(channel):
            return (channel - channel.min()) / (channel.max() - channel.min())

        r = normalise(sim_concs[V])
        g = normalise(sim_concs[U])
        b = 1 - (r + g)/2
        img = jnp.stack([r, g, b], axis=-1)

        img_path = os.path.join(test_dir, "spatial_gray_scott.png")
        plt.imsave(img_path, img)
    
    def test_hopfield(self):
        n = 256

        Up, Un = many_species("Up, Un")
        Wp, Wn, Up_deg, Un_deg = many_rate_constants("Wp, Wn, Up_deg, Un_deg")
        i, j = many_index_symbols("i, j", n)

        hf_crn = ICRN([
            rxn(Up[i], Up[i]+Up[j], relu(Wp[i,j])),
            rxn(Up[i], Up[i]+Un[j], relu(-Wp[i,j])),
            rxn(Un[i], Un[i]+Up[j], relu(Wn[i,j])),
            rxn(Un[i], Un[i]+Un[j], relu(-Wn[i,j])),

            rxn(3*Up[i], 2*Up[i], Up_deg[i]),
            rxn(3*Un[i], 2*Un[i], Un_deg[i]),

            FastReaction(Up[i]+Un[i], 0) # fast reactions use up the limiting reagent
        ])

        hf_exp_params = {
            "dt" : 0.1,
            "dh" : 1.,
            "batch" : False,
            "integration_method" : "relu_RK4",
            "spatial_dim" : (100, 100)
        }

        hf_exp = Experiment(hf_crn, hf_exp_params)

        test_dir = os.path.join("test", "hopfield")

        seed_path = os.path.join(test_dir, "seed.npy")
        concs_spec = {
            Up : jnp.load(seed_path),
            Un : 0
        }

        Wp_path = os.path.join(test_dir, "Wp.npy")
        Wn_path = os.path.join(test_dir, "Wn.npy")
        Up_deg_path = os.path.join(test_dir, "Up_deg.npy")
        Un_deg_path = os.path.join(test_dir, "Un_deg.npy")

        rate_data_spec = {
            Wp : jnp.load(Wp_path),
            Wn : jnp.load(Wn_path),
            Up_deg : jnp.load(Up_deg_path),
            Un_deg : jnp.load(Un_deg_path),
        }

        Up_path = os.path.join(test_dir, "Up.npy")
        Un_path = os.path.join(test_dir, "Un.npy")

        diff_data_spec = {
            Up : jnp.load(Up_path),
            Un : jnp.load(Un_path)
        }

        init_concs, rate_data, diff_data = hf_exp.dict_builder(concs_spec, rate_data_spec, diff_data_spec)

        sim_concs, _ = hf_exp.simulate_time(init_concs, rate_data, diff_data, 12.5)

        target_Up_path = os.path.join(test_dir, "target_Up.npy")
        target_Un_path = os.path.join(test_dir, "target_Un.npy")

        target_Up = jnp.load(target_Up_path)
        target_Un = jnp.load(target_Un_path)

        img = (sim_concs[Up]-sim_concs[Un])[...,:3]
        img = jnp.maximum(img, 0)
        img = img / img.max()
        img_path = os.path.join(test_dir, "turing.png")
        plt.imsave(img_path, img)

        Up_max_abs_error = max_abs_error(sim_concs[Up], target_Up)
        Un_max_abs_error = max_abs_error(sim_concs[Un], target_Un)

    def test_heat_gradient_catalytic_species(self):
        pass

    def test_heat_gradient_spatially_varying_rate_constant(self):
        pass

if __name__ == "__main__":
    unittest.main()