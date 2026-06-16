"""Load reaction networks from the example YAML files in this directory.

Run from the repository root:

    python test/reaction_yaml/load_examples.py
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from icrn import (
    AbstractReaction,
    many_rate_constants,
    many_species,
    solve_well_mixed,
)
from icrn.reactions import fast_rxns_to_update_f

HERE = Path(__file__).resolve().parent


def load_dimerization():
    rxns, symbols = AbstractReaction.load_yaml(HERE / "dimerization.yaml")
    A, B, C = many_species("A, B, C")
    k_f, k_r = many_rate_constants("k_f, k_r")
    return (
        rxns,
        symbols,
        {
            A: jnp.array(1.0),
            B: jnp.array(2.0),
            C: jnp.array(0.0),
        },
        {
            k_f: jnp.array(0.5),
            k_r: jnp.array(0.1),
        },
    )


def load_exponential_decay():
    rxns, symbols = AbstractReaction.load_yaml(HERE / "exponential_decay.yaml")
    A = many_species("A")
    k = many_rate_constants("k")
    return rxns, symbols, {A: jnp.array(1.0)}, {k: jnp.array(1.0)}


def load_indexed_dimerization():
    rxns, symbols = AbstractReaction.load_yaml(
        HERE / "indexed_dimerization.yaml"
    )
    M, D = many_species("M, D")
    K1, K2 = many_rate_constants("K1, K2")
    n = 5
    return (
        rxns,
        symbols,
        {
            M: jnp.ones(n),
            D: jnp.zeros((n, n)),
        },
        {
            K1: jnp.ones((n, n)) * 0.1,
            K2: jnp.ones((n, n)) * 0.05,
        },
    )


def load_fast_annihilation():
    rxns, symbols = AbstractReaction.load_yaml(HERE / "fast_annihilation.yaml")
    A, B = many_species("A, B")
    return (
        rxns,
        symbols,
        {
            A: jnp.array([1.0, 2.0, 1.5]),
            B: jnp.array([2.0, 0.0, 1.5]),
        },
        {},
    )


def main():
    examples = [
        ("dimerization.yaml", load_dimerization),
        ("exponential_decay.yaml", load_exponential_decay),
        ("indexed_dimerization.yaml", load_indexed_dimerization),
        ("fast_annihilation.yaml", load_fast_annihilation),
    ]

    times = jnp.array([0.0, 1.0])

    for name, loader in examples:
        rxns, symbols, conc_vals, rate_vals = loader()
        if rate_vals:
            traj = solve_well_mixed(
                rxns,
                conc_vals=conc_vals,
                rate_constant_vals=rate_vals,
                times=times,
                dt=0.01,
            )
        else:
            state = {k: v.copy() for k, v in conc_vals.items()}
            fast_rxns_to_update_f(rxns)(state)
            traj = {k: jnp.stack([conc_vals[k], state[k]]) for k in state}
        print(f"{name}: loaded {len(rxns)} reaction(s)")
        print(f"  symbol species: {sorted(symbols.species)}")
        summary = {}
        for key, series in traj.items():
            final = series[-1]
            summary[str(key)] = float(final) if final.ndim == 0 else final.shape
        print(f"  final concentrations: {summary}")


if __name__ == "__main__":
    main()
