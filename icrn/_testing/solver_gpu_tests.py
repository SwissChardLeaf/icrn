"""GPU device-placement checks for the public solver API."""

from __future__ import annotations


def well_mixed_runs_on_gpu() -> None:
    import jax.tree_util as jax_tree
    from jax import numpy as jnp

    from icrn.reactions import MassActionReaction
    from icrn.solver import solve_well_mixed
    from icrn.symbols import many_rate_constants, many_species

    A = many_species("A")
    k = many_rate_constants("k")
    rxns = [MassActionReaction(A, 0, k)]

    result = solve_well_mixed(
        rxns,
        conc_vals={A: jnp.array(1.0)},  # type: ignore[dict-item]
        rate_constant_vals={k: jnp.array(1.0)},  # type: ignore[dict-item]
        times=jnp.array([0.0, 1.0]),
        dt=0.01,
    )

    platforms = {leaf.device.platform for leaf in jax_tree.tree_leaves(result)}
    assert platforms == {"gpu"}, (
        f"expected all result leaves on GPU, got {platforms}"
    )


def reaction_diffusion_runs_on_gpu() -> None:
    import jax.tree_util as jax_tree
    from jax import numpy as jnp

    from icrn.reactions import MassActionReaction
    from icrn.solver import solve_reaction_diffusion
    from icrn.symbols import many_rate_constants, many_species

    U, V = many_species("U, V")
    F, k = many_rate_constants("F, k")

    rxns = [
        MassActionReaction(U + 2 * V, 3 * V, 1),
        MassActionReaction(V, 0, F + k),
        MassActionReaction(0, U, F),
        MassActionReaction(U, 0, F),
    ]

    spatial_dims = (16, 16)
    conc_vals = {
        U: jnp.ones(spatial_dims),
        V: jnp.zeros(spatial_dims),
    }
    rate_constant_vals = {F: jnp.array(0.037), k: jnp.array(0.06)}
    diffusion_constant_vals = {U: jnp.array(0.2), V: jnp.array(0.1)}

    result = solve_reaction_diffusion(
        rxns,
        conc_vals,
        rate_constant_vals,
        diffusion_constant_vals,
        times=jnp.array([1.0]),
        dt=0.1,
        spatial_dims=spatial_dims,
        dspaces=(1.0, 1.0),
        mode="relu",
    )

    platforms = {leaf.device.platform for leaf in jax_tree.tree_leaves(result)}
    assert platforms == {"gpu"}, (
        f"expected all result leaves on GPU, got {platforms}"
    )
