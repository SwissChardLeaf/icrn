from jax import lax, checkpoint, vmap
import jax.numpy as jnp

from icrn._numerics._integrator import euler, relu_euler, RK4, relu_RK4

from icrn._core._convolutional_diffusion import _conv_diffuse
from icrn._numerics._spectral_diffusion import _compute_lap_op, _spectral_diffuse
from icrn.utils.dict_utils import map1, map2

# INT_METHOD_DICT = {
#     "euler": euler,
#     "relu_euler": relu_euler,
#     "RK4": RK4,
#     "relu_RK4": relu_RK4,
# }


# def _build_bulk_forward_step(icrn, exp_params):
#     reaction_groups = icrn.reaction_groups(spatial_dim, spatial_rate_constant)

#     integration_method = exp_params["integration_method"]
#     diffusion_method = exp_params["diffusion_method"]

#     rxn_integrator = INT_METHOD_DICT[integration_method]

#     def wm_f(conc_data, rate_data, _, dt):
#         for group in reaction_groups:
#             conc_data = group(conc_data, rate_data, dt)
#         return conc_data

#     res_f = wm_f

#     spatial_dim = exp_params["spatial_dim"]
#     batch = exp_params["batch"]

#     if spatial_dim is not None:
#         if diffusion_method == "spectral":
#             lap_op = _compute_lap_op(
#                 spatial_dim, dh=exp_params["dh"], dw=exp_params["dw"]
#             )

#             def spectral_rd_f(conc_data, rate_data, diff_data, dt):
#                 conc_data = wm_f(conc_data, rate_data, diff_data, dt)
#                 return _spectral_diffuse(conc_data, diff_data, lap_op, dt)

#             res_f = spectral_rd_f
#         else:

#             def conv_rd_f(conc_data, rate_data, diff_data, dt):
#                 conc_data = wm_f(conc_data, rate_data, diff_data, dt)
#                 return _conv_diffuse(conc_data, diff_data, dt)

#             res_f = conv_rd_f

#     if batch:
#         reaction_in_axes = (0, 0, 0, None)
#         return vmap(res_f, in_axes=reaction_in_axes)
#     else:
#         return res_f


def _linear_interpolation_from_hist(times, hist, dt):
    pass


def _scan_linear_interpolation(
    step_f: Callable,
    times: jnp.ndarray,
    state: dict[Species, jnp.ndarray],
    non_state: dict[TensorBacked, jnp.ndarray],
    dt: float,
    key: jnp.ndarray,
    checkpoint_length: int | None = None,
) -> dict[Species, jnp.ndarray]:

    eval_steps = jnp.floor(times / dt).astype(int)
    fractional_dt = times - eval_steps * dt

    def inner_scan_helper(state_key_pair, _):
        state, key, time = state_key_pair
        new_state_key = step_f(state, non_state, time, dt, key)
        new_state, new_key = new_state_key_pair
        return new_state_key_pair, new_state  # return the new state and the new key

    if checkpoint_length:

        @checkpoint
        def outer_scan_helper(state_key_pair, inner_scan_length):
            inner_state_key_pair, inner_hist = lax.scan(
                inner_scan_helper, init=state_key_pair, length=checkpoint_length
            )
            inner_state, inner_key = inner_state_key_pair
            return inner_state_key_pair, inner_state

        state, hist = lax.scan(
            outer_scan_helper, init=(state, key), length=eval_steps[-1] + 1
        )

    def interpolate_hist(hist):
        return (
            hist[eval_steps]
            + (hist[eval_steps + 1] - hist[eval_steps]) * fractional_dt * dt
        )

    state_at_times = jax.tree_map(interpolate_hist, hist)

    return state_at_times


def _scan_by_segments_with_checkpointing(
    step_f,
    state: dict[Species, jnp.ndarray],
    non_state: dict[TensorBacked, jnp.ndarray],
    dt: float,
    key: jnp.ndarray,
    inner_scan_lengths: int,
    outer_scan_length: int,
) -> dict[Species, jnp.ndarray]:
    def inner_scan_helper(state_key_pair, _):
        state, key, time = state_key_pair
        new_state_key = step_f(state, non_state, time, dt, key)
        new_state, new_key = new_state_key_pair
        return new_state_key_pair, new_state  # return the new state and the new key

    @checkpoint
    def outer_scan_helper(state_key_pair, inner_scan_length):
        inner_state_key_pair, inner_hist = lax.scan(
            inner_scan_helper, init=state_key_pair, length=inner_scan_length
        )
        inner_state, inner_key = inner_state_key_pair
        return inner_state_key_pair, inner_state

    outer_state, outer_hist = lax.scan(
        outer_scan_helper, init=(state, key), xs=inner_scan_lengths
    )

    return outer_hist
