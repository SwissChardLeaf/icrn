from typing import Iterable
from .reactions import AbstractReaction, FastReaction
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from .operator import to_well_mixed_ops, to_reaction_diffusion_ops
from ._internal._solve import _solve_with_ops
from .symbols import Species, TensorSymbol, Numeric
from jax import numpy as jnp


def solve_well_mixed(
    rxns: Iterable[AbstractReaction | FastReaction],
    conc_vals: dict[Species, jnp.ndarray],
    rate_constant_vals: dict[TensorSymbol, jnp.ndarray],
    times: Numeric,
    dt: Numeric,
    key=None,
    checkpoint_length=None,
    reaction_solver="RK4",
    mode: str | None = None,
):
    """Integrate a well-mixed reaction network as a system of ODEs.

    Parameters
    ----------
    rxns : iterable of AbstractReaction or FastReaction
        <TODO: the reaction network to integrate.>
    conc_vals : dict[Species, jax.Array]
        <TODO: initial concentrations, keyed by `Species`. Array shape
        must match the species' index sets.>
    rate_constant_vals : dict[TensorSymbol, jax.Array]
        <TODO: numeric values for every `RateConstant` referenced in
        `rxns`.>
    times : Numeric
        <TODO: total integration time.>
    dt : Numeric
        <TODO: integrator step size.>
    key : jax.random.PRNGKey, optional
        <TODO: required only by stochastic integrators; ignored
        otherwise.>
    checkpoint_length : int, optional
        <TODO: trade memory for time when differentiating long
        trajectories.>
    reaction_solver : {"RK4", "Euler"}, optional
        <TODO: integrator used for reaction dynamics.>
    mode : str or None, optional
        <TODO>

    Returns
    -------
    dict[Species, jax.Array]
        <TODO: time-series of concentrations, with a leading time axis.>

    Examples
    --------
    ```python
    # <TODO: minimal exponential-decay example.>
    ```

    See Also
    --------
    [`solve_reaction_diffusion`][icrn.solve_reaction_diffusion] : Spatial
        counterpart with diffusion.
    """
    ops = to_well_mixed_ops(rxns, reaction_solver, mode)
    return solve_with_ops(
        ops=ops,
        state=conc_vals,
        non_state=rate_constant_vals,
        dt=dt,
        times=times,
        key=key,
        checkpoint_length=checkpoint_length,
    )


def solve_reaction_diffusion(
    rxns: Iterable[AbstractReaction | FastReaction],
    conc_vals: dict[Species, jnp.ndarray],
    rate_constant_vals: dict[TensorSymbol, jnp.ndarray],
    diffusion_constant_vals: dict[TensorSymbol, jnp.ndarray],
    times: Numeric,
    dt: Numeric,
    spatial_dims: tuple[int],
    dspaces: tuple[float, ...],
    key=None,
    checkpoint_length=None,
    reaction_solver="RK4",
    boundary_condition="neumann",
    splitting="LieTrotter",
    spatial_rate_constants: bool = False,
    mode: str | None = None,
):
    """Integrate a reaction-diffusion PDE on a regular grid.

    Reactions are integrated with the chosen method, diffusion solved with a spectral method,
    and the two are combined via the chosen splitting scheme.

    Parameters
    ----------
    rxns : iterable of AbstractReaction or FastReaction
        <TODO>
    conc_vals : dict[Species, jax.Array]
        <TODO: initial concentrations on the spatial grid; array shape
        must include both the the spatial axes and the species' index axes.>
    rate_constant_vals : dict[TensorSymbol, jax.Array]
        <TODO>
    diffusion_constant_vals : dict[TensorSymbol, jax.Array]
        <TODO: diffusion coefficient for each species, keyed by the same
        symbols used to declare diffusion in `rxns`.>
    times : Numeric
        <TODO>
    dt : Numeric
        <TODO>
    spatial_dims : tuple of int
        <TODO: shape of the spatial grid, e.g. `(64, 64)`.>
    dspaces : tuple of float
        <TODO: grid spacing along each spatial axis; same length as
        `spatial_dims`.>
    key : jax.random.PRNGKey, optional
        <TODO>
    checkpoint_length : int, optional
        <TODO>
    reaction_solver : {"RK4", "Euler"}, optional
        <TODO>
    splitting : {"LieTrotter", "Strang"}, optional
        <TODO: operator-splitting scheme combining reaction and diffusion
        sub-steps.>
    spatial_rate_constants : bool, optional
        <TODO: when `True`, rate constants vary in space and must be
        supplied as full spatial arrays.>
    mode : str or None, optional
        <TODO>

    Returns
    -------
    dict[Species, jax.Array]
        <TODO: time-series of concentrations, with leading time
        axis followed by spatial and index axes.>

    Examples
    --------
    ```python
    # <TODO: minimal Gray-Scott or Turing example.>
    ```

    See Also
    --------
    [`solve_well_mixed`][icrn.solve_well_mixed] : Non-spatial counterpart.
    """
    ops = to_reaction_diffusion_ops(
        rxns,
        spatial_dims,
        dspaces,
        reaction_solver,
        splitting,
        boundary_condition=boundary_condition,
        mode=mode,
        spatial_rate_constants=spatial_rate_constants,
    )
    combined_vals = (rate_constant_vals, diffusion_constant_vals)
    return solve_with_ops(
        ops=ops,
        state=conc_vals,
        non_state=combined_vals,
        dt=dt,
        times=times,
        key=key,
        checkpoint_length=checkpoint_length,
    )


def solve_with_ops(*, ops, state, non_state, dt, times, key, checkpoint_length):
    """Repeatedly apply a sequence of operators.

    todo

    Parameters
    ----------
    ops : sequence
        <TODO: sequence of operators to apply
    state : dict[Species, jax.Array]
        <TODO: initial state is updated by the operators.>
    non_state : Any
        <TODO: auxiliary data (rate constants, diffusion constants, etc.)
        passed through unchanged.>
    dt : Numeric
        <TODO>
    times : Numeric
        <TODO>
    key : jax.random.PRNGKey or None
        <TODO>
    checkpoint_length : int or None
        <TODO>

    Returns
    -------
    dict[Species, jax.Array]
        <TODO: trajectory.>

    See Also
    --------
    [`solve_well_mixed`][icrn.solve_well_mixed]
    [`solve_reaction_diffusion`][icrn.solve_reaction_diffusion]
    """
    return _solve_with_ops(
        ops, state, non_state, dt, key, times, checkpoint_length
    )