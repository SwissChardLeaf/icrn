from jax import lax, checkpoint, vmap, jit
import jax.numpy as jnp
import jax
import jax.tree as jax_tree

from ..representation.symbols import Species, TensorSymbol
from functools import partial
from ._interpolation import _linear_interpolation

# general purpuse, quite agnositic scan function
# scan function returns key, state pair even if no operators are
# stochastic

def _scan(scan_f, key, state, non_state, dt, length):
    def _helper(key_state_pair, _):
        key, state = key_state_pair
        new_key, new_state = scan_f(key, state, non_state, dt)
        return (new_key, new_state), new_state

    last_state, hist = lax.scan(_helper, init=(key, state), length=length)
    hist_with_init = jax_tree.map(lambda x, y: jnp.concatenate([x[None, ...], y], axis=0), state, hist)
    return last_state, hist_with_init
    

# times has to be positive
def _times_to_steps(times, dt, length):
    n = len(times)
    steps = jnp.floor(times / dt).astype(int)
    segment = jnp.ceil(times / dt / length).astype(int) - 1
    max_segment = segment[-1]

    segment_time = dt * length

    segment_steps = [[] for _ in range(int(max_segment + 1))]
    segment_dt_fractions = [[] for _ in range(int(max_segment + 1))]
    
    for i in range(n):
        segment_index = segment[i]
        mod_time = times[i] - segment_index * segment_time
        mod_step = mod_time // dt
        mod_dt_fraction = (mod_time % dt) / dt

        segment_steps[segment_index].append(mod_step)
        segment_dt_fractions[segment_index].append(mod_dt_fraction)

    segment_steps = list(map(lambda x: jnp.array(x).astype(int), segment_steps))
    segment_dt_fractions = list(map(jnp.array, segment_dt_fractions))
    return segment_steps, segment_dt_fractions

def _loop_with_checkpointing(step_f, times, key, state, non_state, dt, checkpoint_length=None):
    if checkpoint_length is None:
        max_step = jnp.ceil(times[-1] / dt).astype(int)
        checkpoint_length = max_step + 1

    interpolated_hist = []

    if times[0] == 0:
        interpolated_hist.append(jax_tree.map(lambda x: x[None, ...], state))
        times = times[1:]

    segment_steps, segment_dt_fractions = _times_to_steps(times, dt, checkpoint_length)
    print("segment_steps", segment_steps)
    print("segment_dt_fractions", segment_dt_fractions)

    for i in range(len(segment_steps)):
        steps = segment_steps[i]
        dt_fractions = segment_dt_fractions[i]
        (key, state), hist = _scan(step_f, key, state, non_state, dt, length=checkpoint_length)
        interpolated_hist.append(_linear_interpolation(steps, dt_fractions, hist))

    return jax_tree.map(lambda *int_hists: jnp.concatenate([*int_hists]), *interpolated_hist)