from jax import numpy as jnp
import jax
from jax.numpy.fft import fftfreq, fftn, ifftn
import jax.tree as jax_tree


def _compute_lap_op(spatial_dims, dspace):
    Ks = jax_tree.map(
        lambda dim, d: fftfreq(dim, d=d) * 2 * jnp.pi, spatial_dims, dspace
    )
    grids = jnp.meshgrid(*Ks, indexing="ij")
    grids_sq = jax_tree.map(lambda grid: grid**2, grids)

    acc = grids_sq[0]
    for grid_sq in grids_sq[1:]:
        acc += grid_sq
    return -acc


def _spectral_species_diffuse(conc_vals, diffuson_constant_val, lap_op, dt):
    spatial_dims = len(lap_op.shape)
    spatial_axes = list(range(spatial_dims))
    x_hat = fftn(conc_vals, axes=spatial_axes)
    broadcast_shape = lap_op.shape + diffuson_constant_val.shape
    for i in range(len(diffuson_constant_val.shape)):
        lap_op = jnp.expand_dims(lap_op, axis=-1)
    x_hat = x_hat / (
        1
        - dt
        * jnp.broadcast_to(diffuson_constant_val, broadcast_shape)
        * jnp.broadcast_to(lap_op, broadcast_shape)
    )
    return ifftn(x_hat, axes=spatial_axes).real


def _spectral_diffuse(lap_op, conc_vals, diffuson_constant_vals, dt):

    return jax_tree.map(
        lambda c, kd: _spectral_species_diffuse(c, kd, lap_op, dt),
        conc_vals,
        diffuson_constant_vals,
    )
