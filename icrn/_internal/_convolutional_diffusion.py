import jax.tree_util as jax_tree
from jax import numpy as jnp

_PAD_MODES = {
    "dirichlet": ("constant", {"constant_values": 0.0}),
    "neumann": ("edge", {}),
    "periodic": ("wrap", {}),
}


def _pad_for_bc(a, axis, boundary_condition):
    if boundary_condition not in _PAD_MODES:
        raise ValueError(
            f"unknown boundary_condition: {boundary_condition!r}; "
            f"expected one of {tuple(_PAD_MODES)}"
        )
    mode, kwargs = _PAD_MODES[boundary_condition]
    pad_widths = [(0, 0)] * a.ndim
    pad_widths[axis] = (1, 1)
    return jnp.pad(a, pad_widths, mode=mode, **kwargs)


def _second_derivative(a, axis, dspace, boundary_condition):
    a_padded = _pad_for_bc(a, axis, boundary_condition)
    sl_minus = [slice(None)] * a.ndim
    sl_zero = [slice(None)] * a.ndim
    sl_plus = [slice(None)] * a.ndim
    sl_minus[axis] = slice(0, -2)
    sl_zero[axis] = slice(1, -1)
    sl_plus[axis] = slice(2, None)
    return (
        a_padded[tuple(sl_plus)]
        - 2.0 * a_padded[tuple(sl_zero)]
        + a_padded[tuple(sl_minus)]
    ) / (dspace**2)


def _laplacian(a, dspaces, boundary_condition):
    result = _second_derivative(a, 0, dspaces[0], boundary_condition)
    for i in range(1, len(dspaces)):
        result = result + _second_derivative(
            a, i, dspaces[i], boundary_condition
        )
    return result


def _conv_species_diffuse(
    conc,
    diff_constant,
    dt,
    dspaces,
    boundary_condition="neumann",
):
    return (
        conc
        + diff_constant * _laplacian(conc, dspaces, boundary_condition) * dt
    )


def _conv_diffuse(
    concs,
    diff_data,
    dt,
    dspaces,
    boundary_condition="neumann",
):
    return jax_tree.tree_map(
        lambda x, y: _conv_species_diffuse(
            x, y, dt, dspaces, boundary_condition
        ),
        concs,
        diff_data,
    )
