from typing import Iterable

from jax import numpy as jnp

from ._internal._solve import _solve_with_ops
from .operator import to_reaction_diffusion_ops, to_well_mixed_ops
from .reactions import AbstractReaction, FastReaction
from .symbols import Numeric, Species, TensorSymbol


def solve_well_mixed(
    rxns: Iterable[AbstractReaction | FastReaction],
    conc_vals: dict[Species, jnp.ndarray],
    rate_constant_vals: dict[TensorSymbol, jnp.ndarray],
    times: Numeric,
    dt: Numeric,
    checkpoint_length=None,
    reaction_solver="RK4",
    mode: str | None = None,
    pre_computed_state: dict[Species, jnp.ndarray] | None = None,
):
    """Integrate a well-mixed reaction network as a system of ODEs.

    Fast reactions are applied first, followed by the chosen reaction
    integrator for the remaining network dynamics.

    Parameters
    ----------
    rxns : iterable of AbstractReaction or FastReaction
        FastReaction dynamics occur first, followed by the AbstractReaction
        dynamics.
    conc_vals : dict[Species, jax.Array]
        Initial concentrations keyed by `Species`. The dimensions of each
        array must match the species' index axes.
    rate_constant_vals : dict[TensorSymbol, jax.Array]
        Rate constants for the reactions. The dimensions of each array must
        match the species' index axes.
    times : Numeric
        The time points at which to evaluate the solution. The sequence must be
        strictly increasing.
    dt : Numeric
        The time step size. Must be positive.
    checkpoint_length : int, optional
        The number of solver steps to checkpoint.
    reaction_solver : {"RK4", "Euler", "PatankarEuler", "MPE"}, optional
        The method used to integrate the reaction dynamics.
    mode : str or None, optional
        Must be one of "strict", "relu", or None.
    pre_computed_state : dict[Species, jax.Array], optional
        A pre-computed state where leading dimensions are the same length as the
        time sequence and trailing dimensions are the same as the `conc_vals`
        dimensions.

    Returns
    -------
    dict[Species, jax.Array]
        A time-series of concentrations, with a leading time axis followed by
        index axes.

    See Also
    --------
    [`solve_reaction_diffusion`][icrn.solve_reaction_diffusion] : Spatial
        counterpart with diffusion.
    [`solve_with_ops`][icrn.solve_with_ops] : Low-level operator driver.
    """
    ops = to_well_mixed_ops(rxns, reaction_solver, mode)
    return solve_with_ops(
        ops=ops,
        state=conc_vals,
        non_state=rate_constant_vals,
        dt=dt,
        times=times,
        key=None,
        checkpoint_length=checkpoint_length,
        pre_computed_state=pre_computed_state,
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
    checkpoint_length=None,
    reaction_solver="RK4",
    boundary_condition="periodic",
    splitting="LieTrotter",
    spatial_rate_constants: bool = False,
    mode: str | None = None,
    pre_computed_state: dict[Species, jnp.ndarray] | None = None,
):
    """Integrate a reaction-diffusion PDE on a regular grid.

    Reactions are integrated with the chosen method, diffusion solved with a
    spectral method, and the two are combined via the chosen splitting scheme.

    Parameters
    ----------
    rxns : iterable of AbstractReaction or FastReaction
        FastReaction dynamics occur first, followed by the AbstractReaction
        dynamics.
    conc_vals : dict[Species, jax.Array]
        Initial concentrations on the spatial grid. The dimensions of the array
        must match the spatial dimensions and the species' index axes, in
        that order.
    rate_constant_vals : dict[TensorSymbol, jax.Array]
        Rate constants for the reactions. The dimensions of the array must match
        the species' index axes. If `spatial_rate_constants` is `True`, the
        dimensions of the array must match the spatial dimensions and the
        species' index axes, in that order.
    diffusion_constant_vals : dict[TensorSymbol, jax.Array]
        Diffusion coefficients for the species. The dimensions of the array must
        match the species' index axes.
    times : Numeric
        The time points at which to evaluate the solution. The sequence must be
        strictly increasing.
    dt : Numeric
        The time step size. Must be positive.
    spatial_dims : tuple of int
        Shape of the spatial grid, e.g. `(64, 64)`.
    dspaces : tuple of float
        Grid spacing along each spatial axis. This mus be the same length as
        `spatial_dims`.
    checkpoint_length : int, optional
        The number of solver steps to checkpoint.
    reaction_solver : {"RK4", "Euler", "PatankarEuler", "MPE"}, optional
        The method used to integrate the reaction dynamics.
    boundary_condition : {"neumann", "dirichlet", "periodic"}, optional
        The boundary condition to apply to the spatial grid.
    splitting : {"LieTrotter", "Strang"}, optional
        The operator-splitting scheme combining reaction and diffusion
        sub-steps.
    spatial_rate_constants : bool, optional
        When `True`, rate constants vary in space and must be supplied as full
        spatial arrays.
    mode : str or None, optional
        Must be one of "strict", "relu", or None.
    pre_computed_state : dict[Species, jax.Array], optional
        A pre-computed state where leading dimensions are the same length as the
        time sequence and trailing dimensions are the same as the `conc_vals`
        dimensions.

    Returns
    -------
    dict[Species, jax.Array]
        A time-series of concentrations, with leading time axis followed by
        spatial and index axes.

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
        key=None,
        checkpoint_length=checkpoint_length,
        pre_computed_state=pre_computed_state,
    )


def solve_with_ops(
    *,
    ops,
    state,
    non_state,
    dt,
    times,
    key,
    checkpoint_length,
    pre_computed_state=None,
):
    """Repeatedly apply a sequence of operators over a fixed time step.

    This is the low-level driver used by
    [`solve_well_mixed`][icrn.solve_well_mixed]
    and [`solve_reaction_diffusion`][icrn.solve_reaction_diffusion]. The `ops`
    sequence is built with ``to_well_mixed_ops`` or
    ``to_reaction_diffusion_ops`` and applied at every solver step; the
    trajectory is sampled at `times`.

    Parameters
    ----------
    ops : sequence
        Operators to apply in order at each integration step.
    state : dict[Species, jax.Array]
        Initial concentrations keyed by `Species`. The operators update this
        state at every step.
    non_state : Any
        Auxiliary data passed unchanged to each operator on every step. For
        well-mixed simulations this is `rate_constant_vals`; for
        reaction-diffusion it is
        `(rate_constant_vals, diffusion_constant_vals)`.
    dt : Numeric
        The time step size. Must be positive.
    times : Numeric
        The time points at which to evaluate the solution. The sequence must be
        strictly increasing.
    key : jax.random.PRNGKey or None
        A key for the random number generator. Only required by operators.
    checkpoint_length : int or None
        The number of solver steps to checkpoint.
    pre_computed_state : dict[Species, jax.Array], optional
        A pre-computed state where leading dimensions are the same length as the
        time sequence and trailing dimensions are the same as the `state`
        dimensions.

    Returns
    -------
    dict[Species, jax.Array]
        A time-series of concentrations, with a leading time axis followed by
        the trailing axes of the corresponding entries in `state`.

    See Also
    --------
    [`solve_well_mixed`][icrn.solve_well_mixed]
    [`solve_reaction_diffusion`][icrn.solve_reaction_diffusion]
    """
    return _solve_with_ops(
        ops,
        state,
        non_state,
        dt,
        key,
        times,
        checkpoint_length,
        pre_computed_state,
    )
