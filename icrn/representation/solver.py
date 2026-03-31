from collections import Iterable
from .reactions import AbstractReaction
from .._numerics._reaction_numerics import RK4
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class WellMixedSolver:
    reactions: Iterable[AbstractReaction]
    reaction_solver: str = "RK4"
    mode: str = "relu"
    fast_rxn_threshold: float | None = None
    ops: list[AbstractOperator] = field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "ops",
            _to_reaction_diffusion_ops(self.reactions, self.reaction_solver, self.mode),
        )

    def solve(self, state, non_state, dt, key=None):
        return self.solver_f(state, non_state, time, dt, key)

    def __call__(self, state, non_state, time, dt, key=None):
        return self.solve(state, non_state, time, dt, key)


@dataclass(frozen=True)
class ReactionDiffusionSolver:
    reactions: Iterable[AbstractReaction]
    dxs: tuple[float, ...]
    reaction_solver: str = "RK4"
    splitting: str = "LieTrotter"
    diffusion_solver: str = "spectral"
    mode: str = "relu"
    fast_rxn_threshold: float | None = None
    ops: list[AbstractOperator] = field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "ops",
            _to_reaction_diffusion_ops(
                self.reactions,
                self.dxs,
                self.reaction_solver,
                self.splitting,
                self.diffusion_solver,
                self.mode,
            ),
        )

    def solve(self, state, non_state, dt, key=None):
        return self.solve_with_ops(state, non_state, time, dt)

    def __call__(self, state, non_state, time, dt, key=None):
        return self.solve(state, non_state, dt, key)


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


def _problem_to_ops(
    problem, reaction_solver="RK4", splitting="LieTrotter", diffusion_solver="spectral"
):
    if isinstance(problem, WellMixedProblem):
        return _to_well_mixed_ops(problem.reactions, reaction_solver)
    elif isinstance(problem, ReactionDiffusionProblem):
        return _to_reaction_diffusion_ops(
            problem.reactions, problem.dxs, reaction_solver, splitting, diffusion_solver
        )
    else:
        raise ValueError(f"Invalid problem: {problem}")


def _to_well_mixed_ops(reactions, reaction_solver="RK4"):
    pass


def _to_reaction_diffusion_ops(
    reactions,
    dxs,
    reaction_solver="RK4",
    splitting="LieTrotter",
    diffusion_solver="spectral",
):
    if splitting == "LieTrotter":
        return _to_lie_trotter_ops(reactions, dxs, reaction_solver, diffusion_solver)
    elif splitting == "Strang":
        return _to_strang_trotter_ops(reactions, dxs, reaction_solver, diffusion_solver)
    else:
        raise ValueError(f"Invalid splitting: {splitting}")


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
