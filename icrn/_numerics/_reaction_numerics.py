from abc import ABC, abstractmethod
# from ._operator import Operator
from jaxtyping import Float, Array, PyTree, jaxtyped
from typeguard import typechecked
from jax_tree import tree_map

# class ExplicitReactionSolver(Operator):
#     @abstractmethod
#     def step(self, dyn_f, dt):
#         pass

# class Euler(ExplicitReactionSolver):
#     pass
    
# class RK4(ExplicitReactionSolver):
#     pass

@jaxtyped(typechecker=typechecked)
def _euler_step(
        state[PyTree[Float[Array, "?h *?foo"], "T"]], 
        non_state[PyTree[Float[Array, "*?foo"], "T"]], 
        dyn_f: Callable[[PyTree[Float[Array, "?h *?foo"], "T"], PyTree[Float[Array, "*?foo"], "T"]], PyTree[Float[Array, "?h *?foo"], "T"]], 
        dt: float,
        dt_fraction: float
    ) -> PyTree[Float[Array, "?h *?foo"], "T"]:

    state_change = _scalar_mul(dt, dyn_f(state, non_state))
    next_step = _state_add(state, state_change)

    return next_step

@jaxtyped(typechecker=typechecked)
def _RK4_step(
        state[PyTree[Float[Array, "?h *?foo"], "T"]], 
        non_state[PyTree[Float[Array, "*?foo"], "T"]], 
        dyn_f: Callable[[PyTree[Float[Array, "?h *?foo"], "T"], PyTree[Float[Array, "*?foo"], "T"]], PyTree[Float[Array, "?h *?foo"], "T"]], 
        dt: float
    ) -> PyTree[Float[Array, "?h *?foo"], "T"]:

    k1 = dyn_f(state, non_state)
    k2 = dyn_f(_state_sum(state, _scalar_mul(dt * 0.5, k1)), non_state)
    k3 = dyn_f(_state_sum(state, _scalar_mul(dt * 0.5, k2)), non_state)
    k4 = dyn_f(_state_sum(state, _scalar_mul(dt, k3)), non_state)

    next_step = _scalar_mul(dt/6,_state_sum(
        k1, _scalar_mul(k2, 2), _scalar_mul(k3, 2), k4
    ))

    return next_step