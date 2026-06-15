from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

import jax

from ._internal._convolutional_diffusion import _conv_diffuse
from ._internal._reaction_numerics import (
    _euler_step,
    _mpe_step,
    _patankar_euler_step,
    _RK4_step,
)
from ._internal._spectral_diffusion import _compute_lap_op, _spectral_diffuse
from .reactions import (
    AbstractReaction,
    FastReaction,
    fast_rxns_to_update_f,
    rxns_to_dynamics_f,
    rxns_to_mpe_dynamics_f,
    rxns_to_pd_dynamics_f,
)

_VALID_BOUNDARY_CONDITIONS = ("neumann", "dirichlet", "periodic")

"""
An operator is function that takes in a state and a non-state and returns a
new state.

In the most common case, the state is a dict mapping species to their tensor
of concentrations.
The non-state is a dict mapping rate constants to their values or species to
their diffusion coefficients.
"""

# should be jaxtyped?
# def _ops_to_solver(
#     ops: list,
#     initial_state: PyTree[Float[Array, "?h *?foo"], "T"],
#     rate_constants: PyTree[Float[Array, "*?foo"], "T"],
#     diffusion_constants: PyTree[Float[Array, "*?foo"], "T"],
#     dt: float,
# ):
#     def solver(state, non_state, dt, key):

#         for op in ops.values():
#             state = op(state, non_state, dt, key)
#         return state

#         fori_loop()

#     return solver

# def _rxns_to_ops_lst(
#     rxns: list[AbstractReaction],
#     reaction_solver,
#     spatial_info,
#     splitting,
#     diffusion_solver,
# ):
#     if not spatial_info:
#         return _rxn_well_mixed_to_ops_lst(rxns, reaction_solver)
#     else:
# return _rxn_reaction_diffusion_to_ops_lst(rxns, reaction_solver,
# spatial_info, splitting, diffusion_solver)

# def _ops_to_f(ops: list[AbstractOperator]):
#     def f(state, non_state, dt, key):
#         for op in ops:
#             key, state = op(state, non_state, dt, key)
#         return state
#     return f


def to_well_mixed_ops(
    reactions,
    reaction_solver="RK4",
    mode: str | None = None,
    mpe_split: str = "uniform",
):
    non_fast_rxns = [
        rxn for rxn in reactions if isinstance(rxn, AbstractReaction)
    ]
    fast_rxns = [rxn for rxn in reactions if isinstance(rxn, FastReaction)]
    rxn_op = ReactionsOperator(
        mode,
        non_fast_rxns,
        reaction_solver,
        mpe_split=mpe_split,
    )

    ops: list[AbstractOperator] = [rxn_op]
    if fast_rxns:
        fast_rxn_op = FastReactionsOperator(fast_rxns)
        ops.insert(0, fast_rxn_op)

    return ops


def to_reaction_diffusion_ops(
    reactions,
    spatial_dims,
    dspaces,
    reaction_solver="RK4",
    splitting="LieTrotter",
    boundary_condition="neumann",
    spatial_rate_constants=False,
    mode: str | None = None,
    mpe_split: str = "uniform",
):
    if boundary_condition not in _VALID_BOUNDARY_CONDITIONS:
        raise ValueError(
            f"Invalid boundary_condition: {boundary_condition!r}; "
            f"expected one of {_VALID_BOUNDARY_CONDITIONS}"
        )

    non_fast_rxns = [
        rxn for rxn in reactions if isinstance(rxn, AbstractReaction)
    ]
    fast_rxns = [rxn for rxn in reactions if isinstance(rxn, FastReaction)]

    if boundary_condition == "periodic":

        def make_diffusion_op(dt_scale=1.0):
            return SpectralDiffusionOperator(
                mode, spatial_dims, dspaces, dt_scale=dt_scale
            )
    else:

        def make_diffusion_op(dt_scale=1.0):
            return ConvolutionalDiffusionOperator(
                mode,
                spatial_dims,
                dspaces,
                boundary_condition=boundary_condition,
                dt_scale=dt_scale,
            )

    def make_reaction_op(dt_scale=1.0):
        return ReactionsOperator(
            mode,
            non_fast_rxns,
            reaction_solver,
            spatial_axes=len(spatial_dims),
            spatial_rate_constants=spatial_rate_constants,
            dt_scale=dt_scale,
            mpe_split=mpe_split,
        )

    ops = []
    if fast_rxns:
        ops.append(FastReactionsOperator(fast_rxns, len(spatial_dims)))

    if splitting == "LieTrotter":
        ops += [make_reaction_op(), make_diffusion_op()]
    elif splitting == "Strang":
        ops += [
            make_reaction_op(dt_scale=0.5),
            make_diffusion_op(dt_scale=0.5),
            make_reaction_op(dt_scale=0.5),
        ]
    else:
        raise ValueError(f"Invalid splitting: {splitting}")

    return ops


class AbstractOperator(ABC):
    # def __init__(self, mode: str | None, stochastic=False):
    #     update_f = self.update_state

    #     else:
    #         raise ValueError(f"Invalid mode: {mode}")

    #     self._stochastic = stochastic

    @abstractmethod
    def update_state(self, solver_state, non_state, dt):
        pass

    @abstractmethod
    def get_mode(self):
        pass

    @abstractmethod
    def get_is_stochastic(
        self,
    ):  # if stochastic, update_state returns a tuple of (state, key)
        pass

    # @property
    # @abstractmethod
    # def dt_scale(self):
    #     pass

    def __repr__(self):
        return f"{self.__class__.__name__}(aux={self.aux}, mode={self.mode})"


class ReactionsOperator(AbstractOperator):
    def __init__(
        self,
        mode: str | None,
        rxns: Iterable[AbstractReaction],
        reaction_solver,
        spatial_axes=0,
        spatial_rate_constants=False,
        dt_scale=1.0,
        mpe_split="uniform",
    ):
        self._mode = mode
        self._dt_scale = dt_scale

        rxns = list(rxns)
        for rxn in rxns:
            if not isinstance(rxn, AbstractReaction):
                raise ValueError(
                    f"rxns must be an iterator of AbstractReactions, "
                    f"got {rxn} of type {type(rxn)}"
                )

        if reaction_solver == "RK4":
            reaction_solver_f = _RK4_step
            _rxns_dynamics_f = rxns_to_dynamics_f(rxns)
        elif reaction_solver == "Euler":
            reaction_solver_f = _euler_step
            _rxns_dynamics_f = rxns_to_dynamics_f(rxns)
        elif reaction_solver == "PatankarEuler":
            reaction_solver_f = _patankar_euler_step
            _rxns_dynamics_f = rxns_to_pd_dynamics_f(rxns)
        elif reaction_solver == "MPE":
            reaction_solver_f = _mpe_step
            _rxns_dynamics_f = rxns_to_mpe_dynamics_f(rxns, mpe_split)
        else:
            raise ValueError(f"Invalid reaction solver: {reaction_solver}")

        if spatial_axes > 0:

            def spatial_rxn_update_f(state, non_state, dt):
                rate_constant_vals = non_state[0]
                return reaction_solver_f(
                    state, rate_constant_vals, _rxns_dynamics_f, dt
                )

            in_axes = (0, None, None)
            if spatial_rate_constants:
                in_axes = (0, (0, None), None)

            for _ in range(spatial_axes):
                spatial_rxn_update_f = jax.vmap(
                    spatial_rxn_update_f, in_axes=in_axes
                )
            self._rxn_update_f = spatial_rxn_update_f
        else:

            def non_spatial_rxn_update_f(state, rate_constant_vals, dt):
                return reaction_solver_f(
                    state, rate_constant_vals, _rxns_dynamics_f, dt
                )

            self._rxn_update_f = non_spatial_rxn_update_f

    def update_state(self, solver_state, non_state, dt):
        return self._rxn_update_f(solver_state, non_state, self._dt_scale * dt)

    def get_mode(self):
        return self._mode

    def get_is_stochastic(self):
        return False

    def __repr__(self):
        return f"{self.__class__.__name__}(rxns={self.rxns}, reaction_solver={
            self.reaction_solver
        })"


class FastReactionsOperator(AbstractOperator):
    def __init__(self, fast_rxns: Iterable[FastReaction], spatial_axes=0):

        fast_rxns = list(fast_rxns)
        for rxn in fast_rxns:
            if not isinstance(rxn, FastReaction):
                raise ValueError(
                    f"fast_rxns must be an iterator of FastReactions, "
                    f"got {rxn} of type {type(rxn)}"
                )

        fast_rxns_update_f = fast_rxns_to_update_f(fast_rxns)

        for _ in range(spatial_axes):
            fast_rxns_update_f = jax.vmap(fast_rxns_update_f, in_axes=(0,))

        self._fast_rxns_update_f = fast_rxns_update_f

    def get_mode(self):
        return None

    def get_is_stochastic(self):
        return False

    def update_state(self, state, non_state, dt):
        return self._fast_rxns_update_f(state)

    def __repr__(self):
        return f"{self.__class__.__name__}(fast_rxns={self.fast_rxns})"


class SpectralDiffusionOperator(AbstractOperator):
    def __init__(self, mode: str | None, spatial_dims, dspaces, dt_scale=1.0):
        self.mode = mode
        self.spatial_dims = spatial_dims
        self.lap_op = _compute_lap_op(spatial_dims, dspaces)
        self.dt_scale = dt_scale

    def update_state(self, state, non_state, dt):
        diff_constant_vals = non_state[1]
        return _spectral_diffuse(
            self.lap_op, state, diff_constant_vals, self.dt_scale * dt
        )

    def get_mode(self):
        return self.mode

    def get_is_stochastic(self):
        return False

    def __repr__(self):
        return f"{self.__class__.__name__}(xs={self.xs}, dxs={
            self.dxs
        }, diffusion_solver={self.diffusion_solver})"


class ConvolutionalDiffusionOperator(AbstractOperator):
    def __init__(
        self,
        mode: str | None,
        spatial_dims,
        dspaces,
        boundary_condition="neumann",
        dt_scale=1.0,
    ):
        if boundary_condition not in _VALID_BOUNDARY_CONDITIONS:
            raise ValueError(
                f"Invalid boundary_condition: {boundary_condition!r}; "
                f"expected one of {_VALID_BOUNDARY_CONDITIONS}"
            )
        self.mode = mode
        self.spatial_dims = spatial_dims
        self.dspaces = dspaces
        self.boundary_condition = boundary_condition
        self.dt_scale = dt_scale

    def update_state(self, state, non_state, dt):
        diff_constant_vals = non_state[1]
        return _conv_diffuse(
            state,
            diff_constant_vals,
            self.dt_scale * dt,
            self.dspaces,
            self.boundary_condition,
        )

    def get_mode(self):
        return self.mode

    def get_is_stochastic(self):
        return False

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(spatial_dims={self.spatial_dims}, "
            f"dspaces={self.dspaces}, "
            f"boundary_condition={self.boundary_condition})"
        )


# def _rxns_to_op(
#     rxns: list[AbstractReaction],
#     spatial_dims: int = 0,
#     reaction_solver="RK4"
# ):
#     dyn_f = _rxns_to_dynamics(rxns)

#     solver_f = None
#     if reaction_solver == "RK4":
#         solver_f = RK4_step
#     elif reaction_solver == "Euler":
#         solver_f = euler_step
#     else:
#         raise ValueError(f"Invalid reaction solver: {reaction_solver}")

#     res_f =

#     if spatial_dims > 0:
#         vmap_solver_f = solver_f

#         for _ in range(spatial_dims):
#             vmap_solver_f = jax.vmap(
#                 vmap_solver_f, in_axes=(0, None, None, None)
#             )

#         def spatial_rxns_op(state, non_state, dt, key):
#             return fori_loop(
#                 0,
#                 spatial_dims,
#                 lambda i, x: (vmap_solver_f(x, non_state, dt, key), key),
#                 state,
#             )

#         return spatial_rxns_op

#     else:
#         def rxns_op(state, non_state, dt, key):
#             return solver_f(state, non_state, dyn_f, dt)

#         return rxns_op

# def _fast_rxns_to_op(state, non_state, dyn_f, dt):
#     pass


# def _diffusion_to_op(
#     spatial_dim, diffusion_solver="spectral", boundary_conditions=None,
# ):
#     lap = _compute_lap_op(spatial_dim)

#     solver_f = None
#     if diffusion_solver == "spectral":

#         def spectral_diffusion_op(state, non_state, dt, key):
#             return (_spectral_diffuse(lap, state, non_state, dt), key)

#         return spectral_diffusion_op
#     elif diffusion_solver == "convolutional":

#         def convolutional_diffusion_op(state, non_state, dt, key):
#             return (_convolutional_diffuse(state, non_state, dt), key)

#         return convolutional_diffusion_op
#     else:
#         raise ValueError(f"Invalid diffusion solver: {diffusion_solver}")

# def _to_lie_trotter_ops(
#     self, space, reaction_solver, diffusion_solver, boundary_conditions,
# ):
#     return [
#         _rxns_to_op(self.reactions, reaction_solver),
#         _diffusion_to_op(
#             self.spatial_info, diffusion_solver, boundary_conditions,
#         ),
#     ]

# def _to_strang_ops(
#     self, space, reaction_solver, diffusion_solver, boundary_conditions,
# ):
#     return [
#         _rxns_to_op(self.reactions, reaction_solver),
#         _diffusion_to_op(
#             self.spatial_info, diffusion_solver, boundary_conditions,
#         ),
#         _rxns_to_op(self.reactions, reaction_solver)
#     ]

# def _to_well_mixed_ops(self, reaction_solver):
#     return [
#         _fast_rxns_to_op(self.)
#         _rxns_to_op(self.reactions, reaction_solver)
#     ]
