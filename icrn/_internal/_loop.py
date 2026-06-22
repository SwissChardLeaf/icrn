import jax.numpy as jnp
import jax.tree as jax_tree
import numpy as np
from jax import checkpoint, lax

from ._interpolation import _linear_interpolation

# general purpuse, quite agnositic scan function
# scan function returns key, state pair even if no operators are
# stochastic


def _scan_segment(
    scan_f, key, state, non_state, dt, length, segment_pre_computed_state=None
):
    """Advance the state by `length` fixed steps within one checkpoint segment.

    Wraps `jax.lax.scan` to apply `scan_f` repeatedly, threading the
    `(key, state)` carry through each step and stacking every intermediate
    state into a history array. The scan function always returns a
    `(key, state)` pair, even when no operators are stochastic, so this
    helper stays agnostic to the underlying dynamics.

    Parameters
    ----------
    scan_f : callable
        Step function with signature `scan_f((key, state), non_state, dt)`
        returning a new `(key, state)` pair. If pre-computed state is provided,
        'scan_f' can use data in the pre-computed state, but the pre-computed
        state is not modified. i.e. the pre-computed state is catalytic.
    key : jax.Array
        PRNG key forming the first half of the scan carry. Passed through
        unchanged by deterministic steps.
    state : PyTree
        Initial state (e.g. a dict mapping species to concentration arrays).
        Forms the second half of the scan carry and is prepended to the
        returned history.
    non_state : Any
        Auxiliary data (rate constants, diffusion constants, etc.) passed to
        `scan_f` unchanged at every step.
    dt : Numeric
        Solver step size.
    length : int
        Number of steps to scan over in this segment.
    segment_pre_computed_state : dict, optional
        Per-step values to inject into the state. Must be a dict whose
        leaves have a leading axis of size `length`; slice `i` is merged
        into the state (via `state | slice`) before step `i`.

    Returns
    -------
    last_state : tuple
        The final `(key, state)` carry after `length` steps.
    hist_with_init : dict
        Dictionary of stacked state histories with a leading time axis of size
        `length + 1` for each species; the initial `state` is prepended to the
        `length` stacked outputs for each species.
    """

    def _helper(key_state_pair, pre_computed_state):
        key, state = key_state_pair
        if pre_computed_state is not None:
            state = state | pre_computed_state

        new_key, new_state = scan_f((key, state), non_state, dt)
        return (new_key, new_state), new_state

    last_state, hist = lax.scan(
        _helper, init=(key, state), xs=segment_pre_computed_state, length=length
    )
    hist_with_init = jax_tree.map(
        lambda x, y: jnp.concatenate([x[None, ...], y], axis=0), state, hist
    )
    return last_state, hist_with_init


# times has to be positive
def _times_to_steps(times, dt, length):
    """Map query times onto per-segment step indices and fractional offsets.

    Time is broken into contiguous segments of `length` steps.
    For each requested time this computes which segment it falls in,
    the integer step index *within* that segment, and the fractional position
    between that step and the next (used for interpolation).

    Each segment spans `segment_time = dt * length` units of time. A time `t`
    is assigned to segment `ceil(t / segment_time) - 1`, which makes the
    segments **left-open, right-closed**: segment `k` owns the interval
    `(k * segment_time, (k + 1) * segment_time]`. Two consequences follow:

    - A time landing exactly on a boundary `k * segment_time` belongs to the
      *earlier* segment `k - 1` (as its final step `length`), not to the start
      of segment `k`. (`t = 0` is not handled here; the caller strips a leading
      zero, so all times must be strictly positive.)
    - The number of segments is fixed by the last (largest) time. If no query
      time falls inside an intermediate segment, that segment's entry is empty
      (`jnp.array([])`); the caller still scans it but it contributes no
      interpolated samples.

    Within its segment a time `t` is reduced to `mod_time = t - k *
    segment_time`, then split into `mod_step = floor(mod_time / dt)` and
    `mod_dt_fraction = (mod_time mod dt) / dt`.

    Parameters
    ----------
    times : array_like
        Strictly positive query times at which output is desired. Assumed to
        be sorted in increasing order; the last entry determines the number
        of segments. A time 't' is assigned to segment 'i' so that 't' is in
        the interval '(i * dt * length, (i + 1) * dt * length]'.
    dt : float
        Solver step size.
    length : int
        Number of steps per segment.

    Returns
    -------
    segment_steps : list of jax.Array
        One integer array per segment giving the in-segment step index of
        each query time that falls in that segment.
    segment_dt_fractions : list of jax.Array
        One float array per segment, aligned with `segment_steps`, giving
        the fractional offset in `[0, 1)` of each query time between its
        step and the following step.

    Examples
    --------
    With `dt = 0.5` and `length = 4`, each segment spans
    `segment_time = 2.0` (4 steps of 0.5):

    ```python
    times = [0.5, 2.0, 4.25, 5.0]
    segment_steps, segment_dt_fractions = _times_to_steps(times, 0.5, 4)
    # segment_steps        == [[1, 4], [], [0, 2]]
    # segment_dt_fractions == [[0.0, 0.0], [], [0.5, 0.0]]
    ```

    - `0.5` -> segment 0, step 1, fraction 0.0 (lands exactly on a step).
    - `2.0` -> segment 0, step 4, fraction 0.0. This is a *boundary* time; the
      right-closed convention keeps it in segment 0 at its last step rather
      than starting segment 1.
    - `4.25` -> segment 2 (`ceil(4.25 / 2.0) - 1`), `mod_time = 0.25`, so step 0
      with fraction 0.5 (halfway between steps 0 and 1).
    - `5.0` -> segment 2, `mod_time = 1.0`, step 2, fraction 0.0.

    Segment 1 receives no query times, so it is returned empty.
    ```
    """
    n = len(times)

    np.floor(times / dt).astype(int)
    segment = np.ceil(times / dt / length).astype(int) - 1
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


def _split_pre_computed_state_segments(pre_computed_state, checkpoint_length):
    """Split a precomputed trajectory into overlapping checkpoint segments.

    The leaves of `pre_computed_state` share a leading axis whose length equals
    the total number of solver steps. This breaks that axis into checkpoint
    segments so each chunk can be fed to `_scan_segment` as its
    `segment_pre_computed_state` (the scanned `xs`).

    Each segment is treated as a **closed interval** that contains both its
    start and end point. A segment therefore spans `checkpoint_length` steps but
    `checkpoint_length + 1` states, and consecutive segments overlap by one
    state: the end of one segment is the start of the next. This matches the
    per-segment history produced by `_scan_segment`, which prepends the initial
    state and so also has a leading axis of `checkpoint_length + 1`.

    The segments collectively span `num_segments * checkpoint_length + 1`
    states; the trajectory is zero-padded at the end to reach that length so
    that every returned segment has a leading axis of exactly
    `checkpoint_length + 1`.

    Parameters
    ----------
    pre_computed_state : dict
        Precomputed per-step values
        Every leaf must share the same leading axis of size
        `num_solver_steps`; trailing axes may differ between leaves.
    checkpoint_length : int
        Number of solver steps per segment.

    Returns
    -------
    list of dicts
        A list of `ceil(num_solver_steps / checkpoint_length)` PyTrees, each
        matching the structure of `pre_computed_state` with every leaf having a
        leading axis of size `checkpoint_length + 1`. Boundary states are
        shared between adjacent segments, and the final segment is zero-padded
        at the end as needed.

    Examples
    --------
    Five solver steps split into segments of two. Each segment holds three
    states (two steps, endpoints included), and adjacent segments share a
    boundary state (`2` and `4` below):

    ```python
    pre_computed_state = {"C": jnp.arange(5)}  # [0, 1, 2, 3, 4]
    segments = _split_pre_computed_state_segments(pre_computed_state, 2)
    # segments == [
    #     {"C": jnp.array([0, 1, 2])},
    #     {"C": jnp.array([2, 3, 4])},
    #     {"C": jnp.array([4, 0, 0])},   # final segment zero-padded at the end
    # ]
    ```
    """
    leaves = jax_tree.leaves(pre_computed_state)
    num_solver_steps = leaves[0].shape[0]

    num_segments = -(-num_solver_steps // checkpoint_length)
    # closed-interval segments overlap by one state, so they collectively span
    # num_segments * checkpoint_length + 1 states
    padded_length = num_segments * checkpoint_length + 1
    pad = padded_length - num_solver_steps

    def _pad_leading(x):
        pad_width = [(0, pad)] + [(0, 0)] * (x.ndim - 1)
        return jnp.pad(x, pad_width)

    padded = jax_tree.map(_pad_leading, pre_computed_state)

    return [
        jax_tree.map(
            lambda x, start=i * checkpoint_length: x[
                start : start + checkpoint_length + 1
            ],
            padded,
        )
        for i in range(num_segments)
    ]


def _loop_with_checkpointing(
    step_f,
    times,
    key,
    state,
    non_state,
    dt,
    checkpoint_length=None,
    pre_computed_state=None,
):
    """Apply `step_f` repeatedly over time steps `dt` and sample the trajectory
    at `times`.

    Runs the dynamics in contiguous segments of `checkpoint_length` steps,
    scanning each segment with `_scan_segment` and linearly interpolating the
    resulting history onto the requested `times`. When `checkpoint_length` is
    set, each segment scan is wrapped with ``jax.checkpoint`` so reverse-mode
    differentiation rematerializes segment internals instead of storing them.

    Parameters
    ----------
    step_f : callable
        Step function with signature `step_f((key, state), non_state, dt)`
        returning a new `(key, state)` pair.
    times : numpy.ndarray
        Sorted, non-negative output times. A leading `0` is handled
        specially by prepending the initial `state` directly.
    key : jax.Array
        PRNG key threaded through the scan carry.
    state : dict
        Initial state (e.g. a dict mapping species to concentration arrays).
    non_state : dict
        Auxiliary data (rate constants, diffusion constants, etc.) passed to
        `step_f` unchanged.
    dt : float
        Solver step size.
    checkpoint_length : int, optional
        Number of steps per segment. When set, each segment is wrapped with
        ``jax.checkpoint`` for gradient checkpointing. When `None`, a single
        segment spanning the whole trajectory is used
        (`ceil(times[-1] / dt) + 1` steps) with no checkpointing.
    pre_computed_state : dict, optional
        Externally supplied trajectory for a subset of species, injected during
        the scan so the dynamics see it while the species itself follows the
        supplied values rather than the network dynamics. Its leading axis must
        have length `max_step + 1`, where `max_step = ceil(times[-1] / dt)`.
        This corresponds to one entry per solver step, including the initial
        (0th) step.

    Returns
    -------
    PyTree
        The interpolated trajectory with a leading time axis aligned with
        ``times``, matching the structure of ``state``.
    """
    max_step = np.ceil(times[-1] / dt).astype(int)

    if pre_computed_state is not None:
        expected_length = int(max_step) + 1
        for leaf in jax_tree.leaves(pre_computed_state):
            if leaf.shape[0] != expected_length:
                raise ValueError(
                    "pre_computed_state must have a leading axis of length "
                    f"max_step + 1 = {expected_length} "
                    "(max_step = ceil(times[-1] / dt)), but got a leaf with "
                    f"leading axis {leaf.shape[0]}"
                )

    use_gradient_checkpoint = checkpoint_length is not None
    if checkpoint_length is None:
        checkpoint_length = max_step + 1

    scan_segment = (
        checkpoint(_scan_segment, static_argnums=(0, 5))
        if use_gradient_checkpoint
        else _scan_segment
    )

    interpolated_hist = []

    if times[0] == 0:
        interpolated_hist.append(jax_tree.map(lambda x: x[None, ...], state))
        times = times[1:]

    segment_steps, segment_dt_fractions = _times_to_steps(
        times, dt, checkpoint_length
    )

    if pre_computed_state is not None:
        pre_computed_states = _split_pre_computed_state_segments(
            pre_computed_state, checkpoint_length
        )
    else:
        pre_computed_states = [None] * len(segment_steps)

    for i in range(len(segment_steps)):
        steps = segment_steps[i]
        dt_fractions = segment_dt_fractions[i]

        segment_pre_computed_state = pre_computed_states[i]
        if segment_pre_computed_state is not None:
            # segments are length (checkpoint_length + 1) but the scan needs
            # checkpoint_length prefix.
            scan_pre_computed_state = jax_tree.map(
                lambda x: x[:checkpoint_length], segment_pre_computed_state
            )
        else:
            scan_pre_computed_state = None

        (key, state), hist = scan_segment(
            step_f,
            key,
            state,
            non_state,
            dt,
            checkpoint_length,
            scan_pre_computed_state,
        )

        if segment_pre_computed_state is not None:
            hist = hist | segment_pre_computed_state

        interpolated_hist.append(
            _linear_interpolation(steps, dt_fractions, hist)
        )

    return jax_tree.map(
        lambda *int_hists: jnp.concatenate([*int_hists]), *interpolated_hist
    )
