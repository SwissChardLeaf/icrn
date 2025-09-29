import jax
from jax import lax
# from jax.nn import relu
from jax import numpy as jnp
from jax import vmap
from jax.numpy.fft import fftfreq, fftn, ifftn
import jax.tree_util as jax_tree
from .dict_utils import SJDict

NINE_POINT_STENCIL = jnp.array([
    [1/6, 4/6, 1/6],
    [4/6, -20/6, 4/6],
    [1/6, 4/6, 1/6]
], dtype="float32")[..., jnp.newaxis, jnp.newaxis]

STENCIL_H = jnp.array([
    [-1/3, -2/3, 1/3],
    [0, 0, 0],
    [1/3, 2/3, 1/3]
], dtype="float32")[..., jnp.newaxis, jnp.newaxis]

STENCIL_W = jnp.array([
    [-1/3, 0, 1/3],
    [-2/3, 0, 2/3],
    [-1/3, 0, 1/3]
], dtype="float32")[..., jnp.newaxis, jnp.newaxis]

DIM_NUM = lax.conv_dimension_numbers((0,0,0,0), (0,0,0,0), ('HWNC', 'HWIO', 'HWNC'))

def compute_lap_op(spatial_dim, dh ,dw):
    h, w = spatial_dim

    kh = fftfreq(h, d=dh) * 2 * jnp.pi
    kw = fftfreq(w, d=dw) * 2 * jnp.pi

    Kh, Kw = jnp.meshgrid(kh, kw)
    Kh = jnp.transpose(Kh, axes=[1, 0])
    Kw = jnp.transpose(Kw, axes=[1, 0])
    return -(Kh**2) - (Kw**2)

def _spectral_species_diffuse(conc, kd, lap_op, dt):
    x_hat = fftn(conc, axes=[0, 1])
    broadcast_shape = lap_op.shape + kd.shape
    for i in range(len(kd.shape)):
        lap_op = jnp.expand_dims(lap_op, axis=-1)
    x_hat = x_hat / (1 - dt * jnp.broadcast_to(kd[None, None, ...], broadcast_shape) * jnp.broadcast_to(lap_op, broadcast_shape))
    return ifftn(x_hat, axes=[0, 1]).real

def spectral_diffuse(concs, diff_data, lap_op, dt):
    return jax_tree.tree_map(lambda c, kd: _spectral_species_diffuse(c, kd, lap_op, dt), concs, diff_data)

dn = lax.conv_dimension_numbers((0,0,0,0),   # only ndim matters, not shape
                                (0,0,0,0),  # only ndim matters, not shape
                                ('NHWC', 'HWIO', 'NHWC'))  # the important bit


def _da_dspace(a):
    change_h = lax.conv_general_dilated(a, STENCIL_H, (1,1), 'SAME', dimension_numbers=DIM_NUM)
    change_w = lax.conv_general_dilated(a, STENCIL_W, (1,1), 'SAME', dimension_numbers=DIM_NUM)
    return change_h, change_w
    
def _d2a_dspace2(a):
    return lax.conv_general_dilated(a, NINE_POINT_STENCIL, (1,1), 'SAME', dimension_numbers=DIM_NUM)

def _dCdt_spatially_varying(conc, diff_constant):
    dDdh, dDdw = _da_dspace(diff_constant)
    dCdh, dCdw = _da_dspace(conc)
    out = _d2a_dspace2(conc)

    return dDdh * dCdh + dDdw * dCdw + diff_constant * out

def _dCdt_constant(conc, diff_constant):
    return diff_constant * _d2a_dspace2(conc)

def _reshaped_conc_diff_constant(conc, diff_constant, spatially_varying_diffusion_constant):
    original_conc_shape = conc.shape
    conc_reshape = None
    diff_constant_reshape = None

    indexed_species = len(conc.shape) > 2

    if indexed_species:
        prod=1
        for i in range(len(conc.shape[2:])):
            prod *= conc.shape[2:][i]
        new_shape = conc.shape[:2] + (prod,1)
        conc_reshape = jnp.reshape(conc, new_shape)
    else:
        conc_reshape = conc[..., jnp.newaxis, jnp.newaxis]

    if spatially_varying_diffusion_constant:
        diff_constant_reshape = jnp.reshape(diff_constant, conc_reshape.shape)
    else:
        diff_constant_broadcasted = jnp.broadcast_to(diff_constant, original_conc_shape)
        diff_constant_reshape = jnp.reshape(diff_constant_broadcasted, conc_reshape.shape)

    return conc_reshape, diff_constant_reshape

def _conv_species_diffuse(conc, diff_constant, dt, dh=1, dw=1, spatially_varying_diffusion_constant=False):
    # currently assumes dh == dw
    diff = None

    # spatial_dim = conc.shape[:2]
    conc_original_shape = conc.shape
    conc_reshape, diff_constant_reshape = _reshaped_conc_diff_constant(conc, diff_constant, spatially_varying_diffusion_constant)

    # print(conc_reshape.shape)
    # print(diff_constant_reshape.shape)

    if spatially_varying_diffusion_constant:
        diff = _dCdt_spatially_varying(conc_reshape, diff_constant_reshape)
    else:
        diff = _dCdt_constant(conc_reshape, diff_constant_reshape)

    diff_reshape = jnp.reshape(diff, conc_original_shape)
    conc += diff_reshape * dt / (dh**2 * dw**2)
    return conc

def conv_diffuse(concs, diff_data, dt):
    return jax_tree.tree_map(lambda x, y: _conv_species_diffuse(x, y, dt), concs, diff_data)

DIFFUSE_METHOD_DICT = {
    "spectral" : spectral_diffuse,
    "conv" : conv_diffuse
}

def fast_react(concs_data, fast_func):
    concs_data.add_with_dict(fast_func(concs_data))
    return concs_data

def euler(concs_data, rate_data, dt, dynamics_func):
    dynamics = concs_data.init_with_dict(dynamics_func(concs_data | rate_data))
    return concs_data + dt * dynamics

def relu_euler(concs_data, rate_data, dt, dynamics_func):
    dynamics = concs_data.init_with_dict(dynamics_func(concs_data | rate_data))
    return jax_tree.tree_map(jax.nn.relu, concs_data + dt * dynamics)

def RK4(concs_data, rate_constant_data, dt, dynamics_func):
    k1 = concs_data.init_with_dict(dynamics_func(concs_data | rate_constant_data))
    k2 = concs_data.init_with_dict(dynamics_func(concs_data + k1 * dt * 0.5 | rate_constant_data))
    k3 = concs_data.init_with_dict(dynamics_func(concs_data + k2 * dt * 0.5 | rate_constant_data))
    k4 = concs_data.init_with_dict(dynamics_func(concs_data + k3 * dt | rate_constant_data))
    return concs_data + (k1 + (k2 * 2) + (k3 * 2) + k4) * (dt / 6)

def relu_RK4(concs_data, rate_constant_data, dt, dynamics_func):
    k1 = concs_data.init_with_dict(dynamics_func(jax_tree.tree_map(jax.nn.relu, concs_data) | rate_constant_data))
    k2 = concs_data.init_with_dict(dynamics_func(jax_tree.tree_map(jax.nn.relu, concs_data) + k1 * dt * 0.5 | rate_constant_data))
    k3 = concs_data.init_with_dict(dynamics_func(jax_tree.tree_map(jax.nn.relu, concs_data) + k2 * dt * 0.5 | rate_constant_data))
    k4 = concs_data.init_with_dict(dynamics_func(jax_tree.tree_map(jax.nn.relu, concs_data) + k3 * dt | rate_constant_data))
    return jax_tree.tree_map(jax.nn.relu, concs_data + (k1 + (k2 * 2) + (k3 * 2) + k4) * (dt / 6))

def RK4_5(concs, dynamics, dt):
    pass

INT_METHOD_DICT ={
    "euler" : euler,
    "relu_euler" : relu_euler,
    "RK4" : RK4,
    "relu_RK4" : relu_RK4
}

def build_forward_step(icrn, spatial_dim, batch, integration_method, diffusion_method="spectral", spatial_rate_constant=False, **kwargs):
    fast_dynamics, normal_dynamics = icrn.dynamics(spatial_dim, spatial_rate_constant)

    rxn_integrator = INT_METHOD_DICT[integration_method]
    diffuse = DIFFUSE_METHOD_DICT[diffusion_method]

    def wm_f(conc_data, rate_data, _, dt):
        conc_data = fast_react(conc_data, fast_dynamics)
        conc_data = rxn_integrator(conc_data, rate_data, dt, normal_dynamics)
        return conc_data
    
    res_f = wm_f
    
    if spatial_dim is not None:
        if diffusion_method == "spectral":
            lap_op = compute_lap_op(spatial_dim, dh=kwargs["dh"], dw=kwargs["dh"])
                
            def spectral_rd_f(conc_data, rate_data, diff_data, dt):
                conc_data = wm_f(conc_data, rate_data, diff_data, dt)
                return diffuse(conc_data, diff_data, lap_op, dt)
            res_f = spectral_rd_f
        else:
            def conv_rd_f(conc_data, rate_data, diff_data, dt):
                conc_data = wm_f(conc_data, rate_data, diff_data, dt)
                return diffuse(conc_data, diff_data, dt)
        
            res_f = conv_rd_f
        
    if batch:
        reaction_in_axes = (0, 0, 0, None)
        return vmap(res_f, in_axes=reaction_in_axes)
    else:
        return res_f