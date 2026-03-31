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

def tree_mul():
    pass

def tree_add():
    pass

@jaxtyped(typechecker=typechecked)
def _euler_step(
        state[PyTree[Float[Array, "?h *?foo"], "T"]], 
        non_state[PyTree[Float[Array, "*?foo"], "T"]], 
        dyn_f: Callable[[PyTree[Float[Array, "?h *?foo"], "T"], PyTree[Float[Array, "*?foo"], "T"]], PyTree[Float[Array, "?h *?foo"], "T"]], 
        dt: float
    ) -> PyTree[Float[Array, "?h *?foo"], "T"]:

    return state + dt * dyn_f(state, non_state)
    return tree_add(state, tree_mul(dt, dyn_f(state, non_state)))

@jaxtyped(typechecker=typechecked)
def _RK4_step(
        state[PyTree[Float[Array, "?h *?foo"], "T"]], 
        non_state[PyTree[Float[Array, "*?foo"], "T"]], 
        dyn_f: Callable[[PyTree[Float[Array, "?h *?foo"], "T"], PyTree[Float[Array, "*?foo"], "T"]], PyTree[Float[Array, "?h *?foo"], "T"]], 
        dt: float
    ) -> PyTree[Float[Array, "?h *?foo"], "T"]:

    k1 = dyn_f(state, non_state)
    k2 = dyn_f(tree_add(state, tree_mul(k1, dt * 0.5)), non_state)
    k3 = dyn_f(tree_add(state, tree_mul(k2, dt * 0.5)), non_state)
    k4 = dyn_f(state + k3 * dt, non_state)
    return tree_add(
        state, 
        state_mul(
            state_add(
                state_add(
                    k1, 
                    state_mul(k2, 2)
                ), 
                state_mul(
                    k3, 
                    2
                )
            ), 
            state_mul(dt, 1/6)
        )
    )

@jaxtyped(typechecker=typechecked)
def _fast_reaction_step(
        state[PyTree[Float[Array, "?h *?foo"], "T"]], 
        non_state[PyTree[Float[Array, "*?foo"], "T"]], 
        dyn_f: Callable[[PyTree[Float[Array, "?h *?foo"], "T"], PyTree[Float[Array, "*?foo"], "T"]], PyTree[Float[Array, "?h *?foo"], "T"]], 
        dt: float
    ) -> PyTree[Float[Array, "?h *?foo"], "T"]:

    return state + dt * dyn_f(state, non_state)