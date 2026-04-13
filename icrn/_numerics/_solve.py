# input can be the well-mixed ODE or the spatial PDE
# assume in the latter case that the spatial discretization is set
# coordinates sending ops to scan
# responsible for setting up time

from pickle import NONE
from typing import Any, Callable


import opcode
import jax
import jax.numpy as jnp
from jax.experimental import checkify
import jax.tree_util as jax_tree

from ._loop import _loop_with_checkpointing

# this not jax_jit compatible immediately because problem
# def solve_with_reactions(
#     reactions,
#     state,
#     non_state
#     times,
#     dt,
#     splitting="Lie",
#     reaction_solver="RK4",
#     diffusion_solver="spectral",
# ):
#     ops = problem.to_ops(splitting, reaction_solver, diffusion_solver)
#     return solve_with_ops(ops, state_data, non_state_data, dt)


# def solve_well_mixed(
#     reactions,
#     state,
#     non_state,
#     dt,
#     key,
#     times=None,
#     inner_scan_length=None,
#     outer_scan_length=None,
# ):
#     ops = to_well_mixed_ops(problem.reactions, problem.reaction_solver)
#     return _solve_with_ops(
#         ops, state, non_state, dt, key, times, inner_scan_length, outer_scan_length
#     )


# def solve_reaction_diffusion(
#     reactions,
#     state,
#     non_state,
#     dt,
#     key,
#     times=None,
#     inner_scan_length=None,
#     outer_scan_length=None,
# ):
#     ops = to_reaction_diffusion_ops(
#         reactions,
#         problem.dxs,
#         problem.reaction_solver,
#         problem.splitting,
#         problem.diffusion_solver,
#     )
#     return _solve_with_ops(
#         ops, state, non_state, dt, key, times, inner_scan_length, outer_scan_length
#     )

def _check_state_is_non_negative(state):
    leaves = jax_tree.tree_leaves(
        jax_tree.tree_map(lambda x: jnp.all(x >= 0), state)
    )
    if not leaves:
        return jnp.array(True)
    return jnp.all(jnp.stack(leaves))

def _to_mod_op(op):

    if op.mode == "strict":
        def checked_update_f(key, state, non_state, dt):
            new_dt = dt * op.dt_scale
            aux = None
            if op.has_aux:
                state_out, aux = op.update_state(key, state, non_state, new_dt)
                checkify.check(
                    _check_state_is_non_negative(state_out), f"state is negative after {op.__class__.__name__}"
                )
                return state_out, aux
            else:
                state_out = op.update_state(key, state, non_state, new_dt)
                checkify.check(
                    _check_state_is_non_negative(state_out), f"state is negative after {op.__class__.__name__}"
                )
                return state_out

        checked_f = checkify.checkify(checked_update_f)

        def new_update_f(key, state, non_state, dt):
            err, out = checked_f(key, state, non_state, dt)
            checkify.check_error(err)
            return out

        return new_update_f

    elif op.mode == "relu":
        def relu_update_f(key, state, non_state, dt):
            if op.has_aux:
                state_out, aux = op.update_state(key, state, non_state, dt)
                return jax_tree.tree_map(jax.nn.relu, state_out), aux
            else:
                state_out = op.update_state(key, state, non_state, dt)
                return jax_tree.tree_map(jax.nn.relu, state_out)

        return relu_update_f

    elif op.mode is None:
        def no_mode_update_f(key, state, non_state, dt):
            new_dt = dt * op.dt_scale
            return op.update_state(key, state, non_state, new_dt)
        return no_mode_update_f

def _ops_to_f(ops):
    mod_ops = list(map(_to_mod_op, ops))

    def f(state, non_state, dt, key):
        for op in mod_ops:
            state = op.update_state(state, non_state, dt, key)
        return state
    
    return f

def _solve_with_ops(
    ops,
    state,
    non_state,
    dt,
    key,
    times,
    checkpoint_length=None,
    interpolation_method: str = "linear",
):
    ops_f = _ops_to_f(ops)
    return _solve_with_f(ops_f, state, non_state, dt, key, times, checkpoint_length, interpolation_method)

def _solve_with_f(
    ops_f,
    state,
    non_state,
    dt,
    key,
    times,
    checkpoint_length=None,
    interpolation_method: str = "linear",
):
    return _loop_with_checkpointing(ops_f, times, key, state, dt, checkpoint_length, interpolation_method)
    # err, out = _loop_with_checkpointing(ops_f, times, key, state, dt, checkpoint_length, interpolation_f)
    # checkify.check_error(err)
    # return out
    # total_steps = jnp.ceil(times[-1] / dt).astype(int)

    # if times:
    #     return _linear_interpolation_eval_hist(times, eval_times, eval_hist)
    # else:
    #     if not inner_scan_length or not outer_scan_length:
    #         raise ValueError(
    #             f"Inner and outer scan lengths must be provided if times are not provided"
    #         )
    #     return _scan_by_segments_with_checkpointing(
    #         ops_f, state, non_state, dt, key, inner_scan_length, outer_scan_length
    #     )
# these are not jax_jit compatible immediately because of the ops
# def solve_with_ops(
#     ops: list[Callable], conc_data, rate_constant_data, diff_data, time, dt, debug=False
# ):
#     solver = solver_from_ops(ops)
#     return solver(conc_data, rate_constant_data, diff_data, time, dt)

# def scan_helper(x, x):
#     for op in ops.values():
#         x = op(x, rate_constant_data, diff_data)
#     return x

# return _scan_by_segments(
#     scan_helper,
#     conc_data,
#     rate_constant_data,
#     diff_data,
# )

# the return value is a jax compatible function
# def _solver_from_ops(ops):

#     def solver(state, non_state, dt, key):
#         return _scan_by_segments( _function_from_ops(ops), state, non_state, dt, key)

#     return solver

# def get_solver_f(
#     reactions,
#     dxs=None,
#     reaction_solver="RK4",
#     splitting="LieTrotter",
#     diffusion_solver="spectral",
#     spatial_dims=None,
#     mode="relu") -> Callable:

#     if dxs:
#         ops = _to_reaction_diffusion_ops(reactions, dxs, reaction_solver, splitting, diffusion_solver, mode)
#     else:
#         ops = _to_well_mixed_ops(reactions, reaction_solver, mode)

#     return _solver_from_ops(ops)
