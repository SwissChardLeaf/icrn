from abc import ABC, abstractmethod
from curses import KEY_A1
# from ._operator import Operator
# from jaxtyping import Float, Array, PyTree, jaxtyped .
# from typeguard import typechecked
# import jax.tree as jax_tree
from ..utils.dict_utils import arr_mul, dict_add, dict_sum

# class ExplicitReactionSolver(Operator):
#     @abstractmethod
#     def step(self, dyn_f, dt):
#         pass

# class Euler(ExplicitReactionSolver):
#     pass
    
# class RK4(ExplicitReactionSolver):
#     pass

def _euler_step(
        state, 
        non_state, 
        dyn_f,
        dt: float
    ):

    state_change = arr_mul(dyn_f(state, non_state), dt)
    next_step = dict_add(state, state_change)

    return next_step

def _RK4_step(
    state, 
    non_state, 
    dyn_f,
    dt: float,
    return_dynamics: bool = False
):
    k1 = dyn_f(state, non_state)
    k2 = dyn_f(dict_add(state, arr_mul(k1, dt * 0.5)), non_state)
    k3 = dyn_f(dict_add(state, arr_mul(k2, dt * 0.5)), non_state)
    k4 = dyn_f(dict_add(state, arr_mul(k3, dt)), non_state)

    state_change = arr_mul(dict_sum(k1, arr_mul(k2, 2), arr_mul(k3, 2), k4), dt/6)
    next_step = dict_add(state, state_change)
    
    if return_dynamics:
        return next_step, k1
    else:   
        return next_step