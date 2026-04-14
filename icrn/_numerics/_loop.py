from jax import lax, checkpoint, vmap, jit
import jax.numpy as jnp
import jax

from ..representation.symbols import Species, TensorSymbol
from functools import partial

# general purpuse, quite agnositic scan function
def _scan(scan_f, key, state, *args, length=length):
    def _helper(key_state_pair, x):
        key, state = key_state_pair
        return scan_f(key, state, x, *args)

    return lax.scan(_helper, init=(key, state), length=length)
    

def _times_to_steps(times, dt, length):
    n = len(times)
    steps = jnp.floor(times / dt).astype(int)
    segment = steps // length
    max_segment = segment[-1]

    segment_time = dt * length

    segment_steps = [[]] * (max_segment + 1)
    segment_dt_fractions = [[]] * (max_segment + 1)
    
    for i in range(n):
        segment_index = segment[i]
        mod_time = times[i] - segment_index * segment_time
        mod_step = mod_time // dt
        mod_dt_fraction = (mod_time % dt) / dt

        segment_steps[segment_index].append(mod_step)
        segment_dt_fractions[segment_index].append(mod_dt_fraction)

    return zip(segment_steps, segment_dt_fractions)

def _loop_with_checkpointing(step_f, times, key, state, dt, checkpoint_length=None, interpolation_f=_linear_interpolation):
    if checkpoint_length is None:
        max_step = jnp.ceil(times[-1] / dt).astype(int)
        checkpoint_length = max_step + 1

    segmment_pairs = _times_to_steps(times, dt, checkpoint_length)

    interpolated_hist = []

    for steps, dt_fractions in segment_pairs:
        key_state, hist = _scan(step_f, key, state, dt, length=checkpoint_length)
        key, state = key_state

        interpolated_hist.append(interpolation_f(hist, pair_lst))

    return jnp.concatenate(interpolated_hist)