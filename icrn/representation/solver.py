from typing import Iterable
from .reactions import AbstractReaction, FastReaction
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from .operator import to_well_mixed_ops, to_reaction_diffusion_ops
from .._numerics._solve import _solve_with_ops
from ..representation.symbols import Species, TensorSymbol, Numeric
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
    """
    Solve the IVP for well mixed system specified by the reactions.
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
    splitting="LieTrotter",
    spatial_rate_constants: bool = False,
    mode: str | None = None,
):
    ops = to_reaction_diffusion_ops(
        rxns,
        spatial_dims,
        dspaces,
        reaction_solver,
        splitting,
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
    return _solve_with_ops(
        ops, state, non_state, dt, key, times, checkpoint_length
    )


# def solve(
#     rxns: Iterable[AbstractReaction],
#     conc_vals: dict[Species, jnp.ndarray],
#     rate_constant_vals: dict[TensorSymbol, jnp.ndarray],
#     diffusion_constant_vals: dict[TensorSymbol, jnp.ndarray] | None = None,
#     times: Numeric,
#     dt: Numeric,
#     key=None,
#     checkpoint_length = None,
#     interpolation_method: str = "linear",
#     reaction_solver="RK4",
#     spatial_info: tuple[int, ...] | None = None,
#     splitting="LieTrotter",
#     diffusion_solver="spectral",
#     mode: str | None = None
# ):
#     '''
#     Args:
#         rxns: A list of AbstractReactions.
#         state: A dict mapping Species to their tensor of concentrations.
#         non_state: A dict mapping TensorSymbols to their values.
#         times: A Numeric array of times.
#         dt: A Numeric scalar of the time step.
#         key: A jnp.ndarray of the random key.
#         reaction_solver: A string of the reaction solver.
#         spatial_info: A tuple of the spatial information.
#         splitting: A string of the splitting.
#         diffusion_solver: A string of the diffusion solver.

#         The following can be traced through jax.jit:
#         - conc_vals
#         - rate_constant_vals
#         - diffusion_constant_vals
#         - key

#         The following are static:
#         - rxns
#         - times
#         - dt
#         - checkpoint_length
#         - interpolation_method
#         - reaction_solver
#         - spatial_info
#         - splitting
#         - diffusion_solver
#     '''

#     for rxn in rxns:
#         if not isinstance(rxn, AbstractReaction):
#             raise ValueError(f"rxns must be a list of AbstractReactions, got {rxn} of type {type(rxn)}")

#     if spatial_info is not None:
#         if not isinstance(spatial_info, tuple):
#             raise ValueError(f"spatial_info must be a tuple of ints, got {spatial_info} of type {type(spatial_info)}")
#         if len(spatial_info) == 0:
#             raise ValueError(f"spatial_info must be a non-empty tuple, got {spatial_info}")
#         if len(spatial_info) > 3:
#             raise ValueError(f"spatial_info must be a tuple of at most 3 ints, got {spatial_info} with more than 3 dimensions")
#         for dim in spatial_info:
#             if not isinstance(dim, int):
#                 raise ValueError(f"spatial_info must be a tuple of ints, got {spatial_info} of type {type(spatial_info)}")
#             if dim <= 0:
#                 raise ValueError(f"spatial_info must be a tuple of positive ints, got {spatial_info} with a dimension less than or equal to 0")

#     if not isinstance(times, Numeric):
#         raise ValueError(f"times must be a Numeric, got {times} of type {type(times)}")
#     if jnp.array(times).ndim !- 1:
#         raise ValueError(f"times must be a 1D array, got {times} of shape {jnp.array(times).shape}")

#     if not isinstance(dt, Numeric):
#         raise ValueError(f"dt must be a Numeric, got {dt} of type {type(dt)}")
#     if jnp.array(dt).ndim != 0:
#         raise ValueError(f"dt must be a scalar, got {dt} of shape {jnp.array(dt).shape}")
#     if dt <= 0:
#         raise ValueError(f"dt must be a positive float, got {dt} which is less than or equal to 0")
#     if key is not None and not isinstance(key, jnp.ndarray):
#         raise ValueError(f"key must be a jnp.ndarray, got {key} of type {type(key)}")

#     if reaction_solver not in ["RK4", "Euler"]:
#         raise ValueError(f"Invalid reaction solver: {reaction_solver}")
#     if splitting not in ["LieTrotter", "Strang"]:
#         raise ValueError(f"Invalid splitting: {splitting}")
#     if diffusion_solver not in ["spectral", "convolutional"]:
#         raise ValueError(f"Invalid diffusion solver: {diffusion_solver}")
#     if mode not in ["none", "relu", "strict"]:
#         raise ValueError(f"Invalid mode: {mode}")

#     ops = _rxns_to_ops_lst(rxns, reaction_solver, spatial_info, splitting, diffusion_solver, mode)
#     opf_f = _function_from_ops_lst(ops)
#     return _solve_with_ops_f(ops_f, state, non_state, times, dt, key, spatial_info)

# @dataclass(frozen=True)
# class WellMixedSolver:
#     reactions: Iterable[AbstractReaction]
#     reaction_solver: str = "RK4"
#     mode: str = "relu"
#     fast_rxn_threshold: float | None = None
#     ops: list[AbstractOperator] = field(init=False)

#     def __post_init__(self):
#         object.__setattr__(
#             self,
#             "ops",
#             _to_reaction_diffusion_ops(self.reactions, self.reaction_solver, self.mode),
#         )

#     def solve(self, state, non_state, dt, key=None):
#         ops_f = _function_from_ops_lst(self.ops)
#         return _solve_with_ops_f(ops_f, state, non_state, dt, key)


# @dataclass(frozen=True)
# class ReactionDiffusionSolver:
#     reactions: Iterable[AbstractReaction]
#     dxs: tuple[float, ...]
#     reaction_solver: str = "RK4"
#     splitting: str = "LieTrotter"
#     diffusion_solver: str = "spectral"
#     mode: str = "relu"
#     fast_rxn_threshold: float | None = None
#     ops: list[AbstractOperator] = field(init=False)

#     def __post_init__(self):
#         object.__setattr__(
#             self,
#             "ops",
#             _to_reaction_diffusion_ops(
#                 self.reactions,
#                 self.dxs,
#                 self.reaction_solver,
#                 self.splitting,
#                 self.diffusion_solver,
#                 self.mode,
#             ),
#         )

#     def solve(self, state, non_state, dt, key=None):
#         ops_f = _function_from_ops_lst(self.ops)
#         return _solve_with_ops_f(ops_f, state, non_state, dt, key)


# @dataclass
# class ReactionDiffusionSolver:
#     reaction_solver: str = "RK4"
#     splitting: str = "LieTrotter"
#     diffusion_solver: str = "spectral"
#     strict: bool = False
#     key: jnp.ndarray = None

#     def solve(self, state, non_state, time, dt):
#         ops = _to_reaction_diffusion_ops(state.reactions, state.dxs, self.reaction_solver, self.splitting, self.diffusion_solver)
#         return solve_with_ops(ops, state, non_state, time, dt)
# think of this as an outline for an initial value problem

# class AbstractProblem(ABC):
#     def to_ops(self, splitting="Lie", reaction_solver="RK4", diffusion_solver="spectral"):
#         pass

#     @abstractmethod
#     def solve(self, dt, dx, splitting="Lie", reaction_solver="RK4", diffusion_solver="spectral"):
#         pass

# class WellMixedProblem(AbstractProblem):
#     def __init__(self, reactions: Iterable[AbstractReaction]):
#         self.reactions = reactions

#     def to_ops(self, splitting="Lie", reaction_solver="RK4", diffusion_solver="spectral"):
#         return _to_well_mixed_ops(self.reactions, reaction_solver)
# class AbstractProblem(ABC):
#     @abstractmethod
#     def to_ops(self) -> list[Callable]:
#         pass

#     @abstractmethod
#     def solve(self, state, non_state, time, dt, key=None):
#         ops = self.to_ops()
#         return solve(self, state, non_state, time, dt, key)

#     def solver(self, time, dt):
#         ops = self.to_ops()
#         return solver_from_ops(ops, time, dt)

# if you think about this as an IVP, then we need time and space specified


# def _problem_to_ops(
#     problem, reaction_solver="RK4", splitting="LieTrotter", diffusion_solver="spectral"
# ):
#     if isinstance(problem, WellMixedProblem):
#         return _to_well_mixed_ops(problem.reactions, reaction_solver)
#     elif isinstance(problem, ReactionDiffusionProblem):
#         return _to_reaction_diffusion_ops(
#             problem.reactions, problem.dxs, reaction_solver, splitting, diffusion_solver
#         )
#     else:
#         raise ValueError(f"Invalid problem: {problem}")


# def _to_well_mixed_ops(reactions, reaction_solver="RK4"):
#     pass


# def _to_reaction_diffusion_ops(
#     reactions,
#     dxs,
#     reaction_solver="RK4",
#     splitting="LieTrotter",
#     diffusion_solver="spectral",
# ):
#     if splitting == "LieTrotter":
#         return _to_lie_trotter_ops(reactions, dxs, reaction_solver, diffusion_solver)
#     elif splitting == "Strang":
#         return _to_strang_trotter_ops(reactions, dxs, reaction_solver, diffusion_solver)
#     else:
#         raise ValueError(f"Invalid splitting: {splitting}")


# class ReactionDiffusionProblem:
#     def __init__(
#         self,
#         reactions: Iterable[AbstractReaction],
#         xs: tuple[int, ...] | None = None,
#         dxs: tuple[float, ...] | None = None
#     ):
#         self.reactions = reactions
#         self.dxs = dxs
#         self.ops = dict()
#         # self._ops = dict()

#     def to_ops(self, reaction_solver="RK4", splitting="LieTrotter", diffusion_solver="spectral"):
#         spec = (reaction_solver, splitting, diffusion_solver)

#         if spec not in self.ops:
#             self.ops[spec] = _to_reaction_diffusion_ops(self.reactions, self.dxs, reaction_solver, splitting, diffusion_solver)

#         return self.ops[spec]

#         # if splitting == "LieTrotter":
#         #     return _to_lie_trotter_ops(self.reactions, self.spatial_info, reaction_solver, diffusion_solver)
#         # elif splitting == "Strang":
#         #     return _to_strang_trotter_ops(self.reactions, self.spatial_info, reaction_solver, diffusion_solver)
#         # else:
#         #     raise ValueError(f"Invalid splitting: {splitting}")

#     def solve(self, state, non_state, time, dt, key=None, strict=False):
#         ops = self.to_ops()
#         return solve(self, state, non_state, time, dt, key)

# class WellMixedProblem(ReactionDiffusionProblem):
#     def __init__(self, reactions: Iterable[AbstractReaction]):
#         super().__init__(reactions, None, None)

#     def to_ops(self, reaction_solver="RK4") -> list[Callable]:
#         return _to_well_mixed_ops(self.reactions, reaction_solver)

# def to_solver(self, dt, dx, splitting="Lie", reaction_solver="RK4", diffusion_solver="spectral"):
#     solve_spec = (dt, dx, splitting, reaction_solver, diffusion_solver)

#     if solve_spec in self._ops:
#         return self._ops[solve_spec]
#     else:
#         ops = self.to_ops(splitting, reaction_solver, diffusion_solver)
#         self._ops[solve_spec] = _solve_with_ops(ops, self.initial_state, self.rate_constants, self.diffusion_constants, dt)
#         return ops

# def _build_problem(
#     reactions: Iterable[AbstractReaction], spatial_info: tuple[int, int]
# ):
#     return Problem(reactions, spatial_info)
