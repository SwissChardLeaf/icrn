NINE_POINT_STENCIL = jnp.array(
    [[1 / 6, 4 / 6, 1 / 6], [4 / 6, -20 / 6, 4 / 6], [1 / 6, 4 / 6, 1 / 6]],
    dtype="float32",
)[..., jnp.newaxis, jnp.newaxis]

STENCIL_H = jnp.array(
    [[-1 / 3, -2 / 3, 1 / 3], [0, 0, 0], [1 / 3, 2 / 3, 1 / 3]], dtype="float32"
)[..., jnp.newaxis, jnp.newaxis]

STENCIL_W = jnp.array(
    [[-1 / 3, 0, 1 / 3], [-2 / 3, 0, 2 / 3], [-1 / 3, 0, 1 / 3]], dtype="float32"
)[..., jnp.newaxis, jnp.newaxis]

DIM_NUM = lax.conv_dimension_numbers(
    (0, 0, 0, 0), (0, 0, 0, 0), ("HWNC", "HWIO", "HWNC")
)


dn = lax.conv_dimension_numbers(
    (0, 0, 0, 0),  # only ndim matters, not shape
    (0, 0, 0, 0),  # only ndim matters, not shape
    ("NHWC", "HWIO", "NHWC"),
)  # the important bit


def _da_dspace(a):
    change_h = lax.conv_general_dilated(
        a, STENCIL_H, (1, 1), "SAME", dimension_numbers=DIM_NUM
    )
    change_w = lax.conv_general_dilated(
        a, STENCIL_W, (1, 1), "SAME", dimension_numbers=DIM_NUM
    )
    return change_h, change_w


def _d2a_dspace2(a):
    return lax.conv_general_dilated(
        a, NINE_POINT_STENCIL, (1, 1), "SAME", dimension_numbers=DIM_NUM
    )


def _dCdt_spatially_varying(conc, diff_constant):
    dDdh, dDdw = _da_dspace(diff_constant)
    dCdh, dCdw = _da_dspace(conc)
    out = _d2a_dspace2(conc)

    return dDdh * dCdh + dDdw * dCdw + diff_constant * out


def _dCdt_constant(conc, diff_constant):
    return diff_constant * _d2a_dspace2(conc)


def _reshaped_conc_diff_constant(
    conc, diff_constant, spatially_varying_diffusion_constant
):
    original_conc_shape = conc.shape
    conc_reshape = None
    diff_constant_reshape = None

    indexed_species = len(conc.shape) > 2

    if indexed_species:
        prod = 1
        for i in range(len(conc.shape[2:])):
            prod *= conc.shape[2:][i]
        new_shape = conc.shape[:2] + (prod, 1)
        conc_reshape = jnp.reshape(conc, new_shape)
    else:
        conc_reshape = conc[..., jnp.newaxis, jnp.newaxis]

    if spatially_varying_diffusion_constant:
        diff_constant_reshape = jnp.reshape(diff_constant, conc_reshape.shape)
    else:
        diff_constant_broadcasted = jnp.broadcast_to(diff_constant, original_conc_shape)
        diff_constant_reshape = jnp.reshape(
            diff_constant_broadcasted, conc_reshape.shape
        )

    return conc_reshape, diff_constant_reshape


def _conv_species_diffuse(
    conc, diff_constant, dt, dh=1, dw=1, spatially_varying_diffusion_constant=False
):
    # currently assumes dh == dw
    diff = None

    # spatial_dim = conc.shape[:2]
    conc_original_shape = conc.shape
    conc_reshape, diff_constant_reshape = _reshaped_conc_diff_constant(
        conc, diff_constant, spatially_varying_diffusion_constant
    )

    # print(conc_reshape.shape)
    # print(diff_constant_reshape.shape)

    if spatially_varying_diffusion_constant:
        diff = _dCdt_spatially_varying(conc_reshape, diff_constant_reshape)
    else:
        diff = _dCdt_constant(conc_reshape, diff_constant_reshape)

    diff_reshape = jnp.reshape(diff, conc_original_shape)
    conc += diff_reshape * dt / (dh**2 * dw**2)
    return conc


def _conv_diffuse(concs, diff_data, dt):
    return jax_tree.tree_map(
        lambda x, y: _conv_species_diffuse(x, y, dt), concs, diff_data
    )
