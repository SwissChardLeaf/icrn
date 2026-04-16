import unittest
from ..representation.symbols import (
    many_species,
    many_rate_constants,
    many_index_symbols,
    TensorFunction,
)
from ..representation.reactions import MassActionReaction, FastReaction
from ..utils.dict_utils import dict_allclose
from ..representation.solver import solve_well_mixed, solve_reaction_diffusion
import os
from jax import numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np


class TestSolveWellMixed(unittest.TestCase):
    def test_exponential_decay(self):
        A = many_species("A")
        k = many_rate_constants("k")

        rxns = [
            MassActionReaction(A, 0, k),
        ]

        conc_vals = {A: jnp.array(1.0)}
        rate_constant_vals = {k: jnp.array(1.0)}
        times = jnp.array([0, jnp.log(2), 1.0])
        dt = 0.005
        key = None
        checkpoint_length = None
        reaction_solver = "RK4"
        mode = None

        result = solve_well_mixed(
            rxns,
            conc_vals,
            rate_constant_vals,
            times,
            dt,
            key,
            checkpoint_length,
            reaction_solver,
            mode,
        )
        target = {A: jnp.array([1.0, 0.5, jnp.exp(-1.0)])}

        self.assertTrue(dict_allclose(result, target))

    def test_two_decay(self):
        A = many_species("A")
        k = many_rate_constants("k")
        i = many_index_symbols("i")

        rxns = [
            MassActionReaction(A[i], 0, k[i]),
        ]

        conc_vals = {A: jnp.array([1.0, 2.0])}
        rate_constant_vals = {k: jnp.array([1.0, 2.0])}
        times = jnp.array([0, jnp.log(2), 1.0])
        dt = 0.01
        key = None
        checkpoint_length = None
        reaction_solver = "RK4"
        mode = None

        result = solve_well_mixed(
            rxns,
            conc_vals,
            rate_constant_vals,
            times,
            dt,
            key,
            checkpoint_length,
            reaction_solver,
            mode,
        )
        target = {
            A: jnp.array(
                [
                    [1.0, 2.0],  # initial conditions
                    [0.50000554, 0.50315726],  # half life
                    [0.3667276, 0.27067068],  # e^-1, 2e^-2
                ]
            )
        }

        self.assertTrue(dict_allclose(result, target))

    def test_steady_state(self):
        A, B = many_species("A, B")
        k1, k2 = many_rate_constants("k1, k2")
        i = many_index_symbols("i")

        rxns = [
            MassActionReaction(A[i], 0, 1.0),
            MassActionReaction(B[i], 0, 1.0),
            MassActionReaction(0, A[i], k1[i]),
            MassActionReaction(0, B[i], k2[i]),
        ]

        conc_vals = {
            A: jnp.array([1.0, 1.0, 1.0]),
            B: jnp.array([1.0, 1.0, 1.0]),
        }

        rate_constant_vals = {
            k1: jnp.array([1.0, 2.0, 3.0]),
            k2: jnp.array([1.5, 3.5, 4.5]),
        }

        times = jnp.array([0, 5, 10.0])
        dt = 0.01
        key = None
        checkpoint_length = None
        reaction_solver = "RK4"
        mode = None

        result = solve_well_mixed(
            rxns,
            conc_vals,
            rate_constant_vals,
            times,
            dt,
            key,
            checkpoint_length,
            reaction_solver,
            mode,
        )
        target = {A: jnp.array([1.0, 2.0, 3.0]), B: jnp.array([1.5, 3.0, 4.5])}

        # print(result)
        self.assertTrue(dict_allclose(result, target))

    def test_dimerization(self):
        M, D = many_species("M, D")
        K_1, K_2 = many_rate_constants("K_1, K_2")

        n = 10

        i, j = many_index_symbols("i, j", n)

        dimer_rxns = [
            MassActionReaction(M[i] + M[j], D[i, j], K_1[i, j]),
            MassActionReaction(D[i, j], M[i] + M[j], K_2[i, j]),
        ]

        test_path = os.path.join("test", "dimerization")

        conc_vals = {
            M: jnp.load(os.path.join(test_path, "init_M.npy")),
            D: jnp.load(os.path.join(test_path, "init_D.npy")),
        }

        rate_constant_vals = {
            K_1: jnp.load(os.path.join(test_path, "K_1.npy")),
            K_2: jnp.load(os.path.join(test_path, "K_2.npy")),
        }

        times = jnp.array([1.0])
        dt = 1e-5
        key = None
        checkpoint_length = None
        reaction_solver = "Euler"
        mode = None

        sim_concs = solve_well_mixed(
            dimer_rxns,
            conc_vals,
            rate_constant_vals,
            times,
            dt,
            key,
            checkpoint_length,
            reaction_solver,
            mode,
        )

        target_M = jnp.load(os.path.join(test_path, "target_M.npy"))
        target_D = jnp.load(os.path.join(test_path, "target_D.npy"))

        M_max_rel_error = jnp.max(jnp.abs(sim_concs[M][-1] - target_M))
        D_max_rel_error = jnp.max(jnp.abs(sim_concs[D][-1] - target_D))

        self.assertTrue(M_max_rel_error < 0.01)
        self.assertTrue(D_max_rel_error < 0.01)

    def test_winner_take_all(self):
        X, W, XF, P, S, SG, A, RG, YF, Y, Rep, F = many_species(
            "X, W, XF, P, S, SG, A, RG, YF, Y, Rep, F"
        )

        alpha = many_rate_constants("alpha")

        n = 100
        m = 3

        i = many_index_symbols("i", n)
        j, k = many_index_symbols("j, k", m)

        wta_rxns = [
            MassActionReaction(X[i] + W[i, j] + XF[i], X[i] + P[i, j], 36.0),
            MassActionReaction(P[i, j] + SG[j], S[j], 36.0),
            MassActionReaction(S[j] + S[k] + A[j, k], 0, alpha[j, k]),
            MassActionReaction(S[j] + RG[j] + YF[k], S[j] + Y[j], 1.8e-4),
            MassActionReaction(Y[j] + Rep[j], F[j], 3.6),
        ]

        test_dir = os.path.join("test", "winner_take_all")
        img_batch_path = os.path.join(test_dir, "img_batch.npy")
        avg_img = os.path.join(test_dir, "avg_img.npy")

        img_batch = jnp.load(img_batch_path)
        avg_img = jnp.load(avg_img)

        batch_size = img_batch.shape[0]

        conc_vals = {
            X: 5.0 * img_batch,
            W: jnp.broadcast_to(100.0 * avg_img, (batch_size, n, m)),
            XF: jnp.broadcast_to(
                2.0 * jnp.sum(100.0 * avg_img, axis=-1), (batch_size, n)
            ),
            SG: 100.0
            * jnp.ones(
                (
                    batch_size,
                    m,
                )
            ),
            RG: 100.0
            * jnp.ones(
                (
                    batch_size,
                    m,
                )
            ),
            A: 400.0 * jnp.ones((batch_size, m, m)),
            YF: 200.0
            * jnp.ones(
                (
                    batch_size,
                    m,
                )
            ),
            Rep: 200.0
            * jnp.ones(
                (
                    batch_size,
                    m,
                )
            ),
            # below should start at zero
            P: jnp.zeros((batch_size, n, m)),
            S: jnp.zeros(
                (
                    batch_size,
                    m,
                )
            ),
            Y: jnp.zeros(
                (
                    batch_size,
                    m,
                )
            ),
            F: jnp.zeros(
                (
                    batch_size,
                    m,
                )
            ),
        }

        rate_constant_vals = {
            alpha: 3.6e-3 * (jnp.ones((m, m)) - jnp.identity(m))
        }

        times = jnp.array([10.0])
        dt = 1e-4
        key = None
        checkpoint_length = None
        reaction_solver = "Euler"
        mode = None

        batch_solve = jax.vmap(
            solve_well_mixed,
            in_axes=(None, 0, None, None, None, None, None, None, None),
        )
        sim_concs = batch_solve(
            wta_rxns,
            conc_vals,
            rate_constant_vals,
            times,
            dt,
            key,
            checkpoint_length,
            reaction_solver,
            mode,
        )

        target_F_path = os.path.join(test_dir, "target_F.npy")
        target_F = jnp.load(target_F_path)

        F_max_rel_error = jnp.max(jnp.abs(sim_concs[F][:, -1] - target_F))

        self.assertTrue(F_max_rel_error < 0.01)

    def test_jit(self):
        A, B = many_species("A, B")
        k = many_rate_constants("k")

        rxns = (
            MassActionReaction(A, B, k),
        )

        conc_vals = {A: jnp.array(1.0), B: jnp.array(0.0)}
        rate_constant_vals = {k: jnp.array(1.0)}

        times = jnp.array([0, 1, 2, 3])
        dt = 1.0

        @jax.jit
        def solve_well_mixed_jit(conc_vals, rate_constant_vals):
            return solve_well_mixed(
                rxns,
                conc_vals,
                rate_constant_vals,
                times,
                dt
            )

        sim_concs_jit = solve_well_mixed_jit(conc_vals, rate_constant_vals)

        print(sim_concs_jit[A][-1])



class TestSolveReactionDiffusion(unittest.TestCase):
    def setUp(self):
        U, V = many_species("U, V")
        F, k = many_rate_constants("F, k")

        self.gs_rxns = [
            MassActionReaction(U + 2 * V, 3 * V, 1),
            MassActionReaction(V, 0, F + k),
            MassActionReaction(0, U, F),
            MassActionReaction(U, 0, F),
        ]

    def test_gray_scott(self):
        U, V = many_species("U, V")
        F, k = many_rate_constants("F, k")

        rxns = [
            MassActionReaction(U + 2 * V, 3 * V, 1),
            MassActionReaction(V, 0, F + k),
            MassActionReaction(0, U, F),
            MassActionReaction(U, 0, F),
        ]

        times = jnp.array([5e3])
        dt = 1
        key = None
        checkpoint_length = None
        reaction_solver = "RK4"
        mode = "relu"
        dspaces = (1.0, 1.0)
        spatial_dims = (101, 101)

        test_dir = os.path.join("test", "gray_scott")
        init_U_path = os.path.join(test_dir, "init_U.npy")
        init_V_path = os.path.join(test_dir, "init_V.npy")

        conc_vals = {U: jnp.load(init_U_path), V: jnp.load(init_V_path)}
        rate_constant_vals = {F: jnp.array(0.037), k: jnp.array(0.06)}
        diffusion_constant_vals = {U: jnp.array(0.2), V: jnp.array(0.1)}

        sim_concs = solve_reaction_diffusion(
            rxns,
            conc_vals,
            rate_constant_vals,
            diffusion_constant_vals,
            times,
            dt,
            spatial_dims,
            dspaces,
            key,
            checkpoint_length,
            reaction_solver,
            mode=mode,
        )

        target_U_path = os.path.join(test_dir, "target_U.npy")
        target_V_path = os.path.join(test_dir, "target_V.npy")

        target_U = jnp.load(target_U_path)
        target_V = jnp.load(target_V_path)

        def normalise(channel):
            return (channel - channel.min()) / (channel.max() - channel.min())

        r = normalise(sim_concs[V][-1])
        g = normalise(sim_concs[U][-1])
        b = 1 - (r + g) / 2
        img = jnp.stack([r, g, b], axis=-1)

        img_path = os.path.join(test_dir, "pink_maze_on_green.png")
        plt.imsave(img_path, img)

        U_mean_error = jnp.mean(jnp.abs(sim_concs[U][-1] - target_U))
        V_mean_error = jnp.mean(jnp.abs(sim_concs[V][-1] - target_V))

        self.assertTrue(U_mean_error < 0.2)
        self.assertTrue(V_mean_error < 0.2)

    def test_spatial_gray_scott(self):

        U, V = many_species("U, V")
        F, k = many_rate_constants("F, k")

        rxns = [
            MassActionReaction(U + 2 * V, 3 * V, 1),
            MassActionReaction(V, 0, F + k),
            MassActionReaction(0, U, F),
            MassActionReaction(U, 0, F),
        ]

        times = jnp.array([1e1])
        dt = 1
        key = None
        checkpoint_length = None
        reaction_solver = "RK4"
        mode = "relu"
        spatial_rate_constants = True
        spatial_dims = (1000, 1000)
        dspaces = (1.0, 1.0)
        key = jax.random.key(12)

        conc_vals = {
            U: jnp.zeros(spatial_dims),
            V: 0.9 + 0.1 * jax.random.uniform(key, spatial_dims),
        }

        rate_constant_vals = {
            F: jnp.broadcast_to(
                jnp.linspace(0.08, 0.01, num=1000)[..., jnp.newaxis],
                (1000, 1000),
            ),
            k: jnp.broadcast_to(
                jnp.linspace(0.03, 0.07, num=1000)[jnp.newaxis, ...],
                (1000, 1000),
            ),
        }
        print(rate_constant_vals[F].shape)
        print(rate_constant_vals[k].shape)

        diffusion_constant_vals = {U: jnp.array(0.2), V: jnp.array(0.1)}

        sim_concs = solve_reaction_diffusion(
            rxns,
            conc_vals,
            rate_constant_vals,
            diffusion_constant_vals,
            times,
            dt,
            spatial_dims,
            dspaces,
            key,
            checkpoint_length,
            reaction_solver,
            spatial_rate_constants=spatial_rate_constants,
            mode=mode,
        )

        self.assertTrue(sim_concs[V][-1].shape == (1000, 1000))
        self.assertTrue(sim_concs[U][-1].shape == (1000, 1000))

        def normalise(channel):
            return (channel - channel.min()) / (channel.max() - channel.min())

        r = normalise(sim_concs[V])
        g = normalise(sim_concs[U])
        b = 1 - (r + g) / 2
        img = jnp.stack([r, g, b], axis=-1)

        test_dir = os.path.join("test", "gray_scott")
        img_path = os.path.join(test_dir, "spatial_gray_scott.png")
        plt.imsave(img_path, img)

    def test_turing_hopfield(self):
        n = 256

        Up, Un = many_species("Up, Un")
        Wp, Wn, Up_deg, Un_deg = many_rate_constants("Wp, Wn, Up_deg, Un_deg")
        i, j = many_index_symbols("i, j", n)

        def relu(x):
            return TensorFunction(jax.nn.relu, (x,))

        rxns = [
            MassActionReaction(Up[i], Up[i] + Up[j], relu(Wp[i, j])),
            MassActionReaction(Up[i], Up[i] + Un[j], relu(-Wp[i, j])),
            MassActionReaction(Un[i], Un[i] + Up[j], relu(Wn[i, j])),
            MassActionReaction(Un[i], Un[i] + Un[j], relu(-Wn[i, j])),
            MassActionReaction(3 * Up[i], 2 * Up[i], Up_deg[i]),
            MassActionReaction(3 * Un[i], 2 * Un[i], Un_deg[i]),
            FastReaction(
                Up[i] + Un[i], 0
            ),  # fast reactions use up the limiting reagent
        ]

        spatial_dims = (100, 100)
        dspaces = (1.0, 1.0)
        key = None
        mode = "relu"
        reaction_solver = "RK4"
        dt = 0.1
        times = jnp.array([12.5])
        checkpoint_length = None

        test_dir = os.path.join("test", "hopfield")
        seed_path = os.path.join(test_dir, "seed.npy")
        conc_vals = {Up: jnp.load(seed_path), Un: jnp.zeros((*spatial_dims, n))}

        rate_constant_vals = {
            Wp: jnp.load(os.path.join(test_dir, "Wp.npy")),
            Wn: jnp.load(os.path.join(test_dir, "Wn.npy")),
            Up_deg: jnp.load(os.path.join(test_dir, "Up_deg.npy")),
            Un_deg: jnp.load(os.path.join(test_dir, "Un_deg.npy")),
        }

        diffusion_constant_vals = {
            Up: jnp.load(os.path.join(test_dir, "Up.npy")),
            Un: jnp.load(os.path.join(test_dir, "Un.npy")),
        }

        sim_concs = solve_reaction_diffusion(
            rxns,
            conc_vals,
            rate_constant_vals,
            diffusion_constant_vals,
            times,
            dt,
            spatial_dims,
            dspaces,
            key,
            checkpoint_length,
            reaction_solver,
            mode=mode,
        )

        target_Up_path = os.path.join(test_dir, "target_Up.npy")
        target_Un_path = os.path.join(test_dir, "target_Un.npy")

        target_Up = jnp.load(target_Up_path)
        target_Un = jnp.load(target_Un_path)

        img = (sim_concs[Up][-1] - sim_concs[Un][-1])[..., :3]
        img = jnp.maximum(img, 0)
        img = img / img.max()
        img_path = os.path.join(test_dir, "turing.png")
        plt.imsave(img_path, img)

        Up_max_abs_error = (sim_concs[Up][-1] - target_Up).max()
        Un_max_abs_error = (sim_concs[Un][-1] - target_Un).max()

        self.assertTrue(Up_max_abs_error < 0.01)
        self.assertTrue(Un_max_abs_error < 0.01)