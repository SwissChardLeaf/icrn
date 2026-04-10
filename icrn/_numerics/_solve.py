# input can be the well-mixed ODE or the spatial PDE
# assume in the latter case that the spatial discretization is set
# coordinates sending ops to scan
# responsible for setting up time

from ._scan import (
    _scan_by_segments_linear_interpolation,
    _scan_by_segments_with_checkpointing,
)
from jax.lax import fori_loop

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


def _solve_with_ops_f(
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
