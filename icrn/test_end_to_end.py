import unittest

from icrn import (
    many_species,
    many_rate_constants,
    many_index_symbols,
    MassActionReaction,
    FastReaction,
)

rxn = MassActionReaction
frxn = FastReaction


class GrayScott(unittest.TestCase):
    def setUp(self):
        U, V = many_species("U, V")
        F, k = many_rate_constants("F, k")

        self.rxns = [
            rxn(U + 2 * V, 3 * V, 1),
            rxn(V, 0, F + k),
            rxn(0, U, F),
            rxn(U, 0, F),
        ]


class WinnerTakeAll(unittest.TestCase):
    def setUp(self):
        X, W, XF, P, S, SG, A, RG, YF, Y, Rep, F = many_species(
            "X, W, XF, P, S, SG, A, RG, YF, Y, Rep, F"
        )
        alpha = many_rate_constants("alpha")
        i, j, k = many_index_symbols("i, j, k")

        self.rxns = [
            rxn(X[i] + W[i, j] + XF[i], X[i] + P[i, j], 36.0),
            rxn(P[i, j] + SG[j], S[j], 36.0),
            rxn(S[j] + S[k] + A[j, k], 0, alpha[j, k]),
            rxn(S[j] + RG[j] + YF[k], S[j] + Y[j], 1.8e-4),
            rxn(Y[j] + Rep[j], F[j], 3.6),
        ]


class Dimerization(unittest.TestCase):
    def setUp(self):
        M, D = many_species("M, D")
        K_1, K_2 = many_rate_constants("K_1, K_2")
        i, j = many_index_symbols("i, j", n)

        self.rxns = [
            rxn(M[i] + M[j], D[i, j], K_1[i, j]),
            rxn(D[i, j], M[i] + M[j], K_2[i, j]),
        ]


class Hopfield(unittest.TestCase):
    def setUp(self):
        Up, Un = many_species("Up, Un")
        Wp, Wn, Up_deg, Un_deg = many_rate_constants("Wp, Wn, Up_deg, Un_deg")
        i, j = many_index_symbols("i, j", n)

        self.rxns = [
            rxn(Up[i], Up[i] + Up[j], relu(Wp[i, j])),
            rxn(Up[i], Up[i] + Un[j], relu(-Wp[i, j])),
            rxn(Un[i], Un[i] + Up[j], relu(Wn[i, j])),
            rxn(Un[i], Un[i] + Un[j], relu(-Wn[i, j])),
            rxn(3 * Up[i], 2 * Up[i], Up_deg[i]),
            rxn(3 * Un[i], 2 * Un[i], Un_deg[i]),
            frxn(
                Up[i] + Un[i], 0
            ),  # fast reactions use up the limiting reagent
        ]
