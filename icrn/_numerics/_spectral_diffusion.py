from jax import numpy as jnp
import jax
from jax.numpy.fft import fftfreq, fftn, ifftn
import jax.tree as jax_tree
from jaxtyping import Array, Float, PyTree, jaxtyped
from typeguard import typechecked


@jaxtyped(typechecker=typechecked)
def _compute_lap_op(spatial_dims, dspace):
    Ks = jax_tree.map(lambda dim, d: fftfreq(dim, d=d) * 2 * jnp.pi, spatial_dims, dspace)
    grids = jnp.meshgrid(*Ks, indexing="ij")
    grids_sq = jax_tree.map(lambda grid: grid**2, grids)

    acc = grids_sq[0]
    for grid_sq in grids_sq[1:]:
        acc += grid_sq
    return -acc


@jaxtyped(typechecker=typechecked)
def _spectral_species_diffuse(
    conc: Float[Array, "h w *dims"],
    kd: Float[Array, "*dims"],
    lap_op: Float[Array, "h w"],
    dt: float,
) -> Float[Array, "h w *dims"]:

    x_hat = fftn(conc, axes=[0, 1])
    broadcast_shape = lap_op.shape + kd.shape
    for i in range(len(kd.shape)):
        lap_op = jnp.expand_dims(lap_op, axis=-1)
    x_hat = x_hat / (
        1
        - dt
        * jnp.broadcast_to(kd[None, None, ...], broadcast_shape)
        * jnp.broadcast_to(lap_op, broadcast_shape)
    )
    return ifftn(x_hat, axes=[0, 1]).real


@jaxtyped(typechecker=typechecked)
def _spectral_diffuse(
    lap_op: Float[Array, "h w"],
    state: PyTree[Float[Array, "h w *?dims"], "T"],  # type: ignore
    non_state: PyTree[Float[Array, "*?dims"], "T"],  # type: ignore
    dt: float,
    dxs: tuple[float, ...],
):

    return jax_tree.tree_map(
        lambda c, kd: _spectral_species_diffuse(c, kd, lap_op, dt), concs, diff_data
    )


@jaxtyped(typechecker=typechecked)
def _test(
    x: PyTree[Float[jax.Array, "?h *?foo"], "T"],  # type: ignore
    y: PyTree[Float[jax.Array, "?h *?foo"], "T"],  # type: ignore
) -> PyTree[Float[jax.Array, "*?foo"], "T"]:  # type: ignore

    add_tree = jax_tree.tree_map(jnp.add, x, y)
    # return add_tree
    sum_tree = jax_tree.tree_map(lambda z: jnp.sum(z, axis=0), add_tree)
    return sum_tree
