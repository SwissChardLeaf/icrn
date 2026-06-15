# from ._operator import Operator
# from jaxtyping import Float, Array, PyTree, jaxtyped .
# from typeguard import typechecked
# import jax.tree as jax_tree
import jax.numpy as jnp

from ..utils.dict_utils import arr_mul, dict_add, dict_div, dict_mul, dict_sum

# class ExplicitReactionSolver(Operator):
#     @abstractmethod
#     def step(self, dyn_f, dt):
#         pass

# class Euler(ExplicitReactionSolver):
#     pass

# class RK4(ExplicitReactionSolver):
#     pass


def _euler_step(state, non_state, dyn_f, dt: float):

    state_change = arr_mul(dyn_f(state, non_state), dt)
    next_step = dict_add(state, state_change)

    return next_step


def _patankar_euler_step(state, non_state, pd_f, dt: float):
    production, destruction = pd_f(state, non_state)

    numerator = dict_add(state, arr_mul(production, dt))
    denominator = dict_add(state, arr_mul(destruction, dt))

    return dict_mul(state, dict_div(numerator, denominator))


def _mpe_step(state, non_state, mpe_f, dt: float):
    """Advance one step of the modified Patankar (Euler) method.

    The modified Patankar method treats destruction implicitly and weights
    each production term by its source concentration, yielding a
    linearly-implicit update that is unconditionally positive: starting from
    positive concentrations the result stays positive for any ``dt``. The
    update is obtained by assembling and solving the linear system
    ``A @ c_next = b``, where ``A``'s diagonal carries the lumped destruction
    and its off-diagonal carries the source-attributed production.

    Parameters
    ----------
    state : dict[Species, jax.Array]
        Current concentrations, keyed by base species. Each value is assumed
        to be a scalar (per cell, under spatial batching); indexed species
        are not yet supported. All concentrations should be strictly
        positive; see Notes.
    non_state : Any
        Auxiliary data (e.g. rate constants) forwarded to ``mpe_f``.
    mpe_f : callable
        A dynamics function, as built by
        [`rxns_to_mpe_dynamics_f`][icrn.rxns_to_mpe_dynamics_f], returning
        ``(destruction, pairs, explicit)`` for the current state.
    dt : float
        Integrator step size.

    Returns
    -------
    dict[Species, jax.Array]
        Updated concentrations, keyed by the same base species as ``state``.

    Notes
    -----
    Self-production (a species produced from itself, e.g. autocatalysis) and
    sourceless influx are added to the right-hand side explicitly, which
    keeps ``A`` a diagonally dominant M-matrix and preserves positivity.

    Zero concentrations are not supported: the implicit weights divide by the
    old concentration on both the diagonal and off-diagonal entries. A
    species at zero with zero destruction (e.g. an empty product or inert
    species) yields ``0/0``; a reactant at zero with a nonzero attributed
    rate yields ``x/0``. Either case produces ``NaN`` or ``inf`` and
    corrupts the linear solve. Use strictly positive initial data, or add a
    denominator guard if zeros are required.

    See Also
    --------
    _patankar_euler_step : The lumped, elementwise Patankar-Euler step.
    """
    species = sorted(state.keys())
    pos = {s: i for i, s in enumerate(species)}
    n = len(species)

    c = jnp.stack([jnp.asarray(state[s]) for s in species])
    b = c

    destruction, pairs, explicit = mpe_f(state, non_state)

    diag = jnp.stack([jnp.asarray(destruction[s]) for s in species])
    A = jnp.eye(n) + jnp.diag(dt * diag / c)

    for (prod_s, react_s), rate in pairs.items():
        i = pos[prod_s]
        j = pos[react_s]
        rate = jnp.asarray(rate)
        if i == j:
            b = b.at[i].add(dt * rate)
        else:
            A = A.at[i, j].add(-dt * rate / c[j])

    b = b + dt * jnp.stack([jnp.asarray(explicit[s]) for s in species])

    c_next = jnp.linalg.solve(A, b)

    return {s: c_next[pos[s]] for s in species}


def _RK4_step(
    state, non_state, dyn_f, dt: float, return_dynamics: bool = False
):
    k1 = dyn_f(state, non_state)
    k2 = dyn_f(dict_add(state, arr_mul(k1, dt * 0.5)), non_state)
    k3 = dyn_f(dict_add(state, arr_mul(k2, dt * 0.5)), non_state)
    k4 = dyn_f(dict_add(state, arr_mul(k3, dt)), non_state)

    state_change = arr_mul(
        dict_sum(k1, arr_mul(k2, 2), arr_mul(k3, 2), k4), dt / 6
    )
    next_step = dict_add(state, state_change)

    if return_dynamics:
        return next_step, k1
    else:
        return next_step
