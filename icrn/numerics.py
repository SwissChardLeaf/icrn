import jax
from jax import lax
# from jax.nn import relu
from jax import numpy as jnp
from jax import vmap
from jax.numpy.fft import fftfreq, fftn, ifftn
import jax.tree_util as jax_tree
from .dict_utils import SJDict

def compute_lap_op(spatial_dim, dh ,dw):
    h, w = spatial_dim

    kx = fftfreq(w, d=dw) * 2 * jnp.pi
    ky = fftfreq(h, d=dh) * 2 * jnp.pi

    Kx, Ky = jnp.meshgrid(kx, ky)
    Kx = jnp.transpose(Kx, axes=[1, 0])
    Ky = jnp.transpose(Ky, axes=[1, 0])
    return -(Kx**2) - (Ky**2)

def _species_diffuse(conc, kd, lap_op, dt):
    x_hat = fftn(conc, axes=[0, 1])
    broadcast_shape = lap_op.shape + kd.shape
    for i in range(len(kd.shape)):
        lap_op = jnp.expand_dims(lap_op, axis=-1)
    x_hat = x_hat / (1 - dt * jnp.broadcast_to(kd[None, None, ...], broadcast_shape) * jnp.broadcast_to(lap_op, broadcast_shape))
    return ifftn(x_hat, axes=[0, 1]).real

def diffuse(concs, diff_data, lap_op, dt):
    return jax_tree.tree_map(lambda c, kd: _species_diffuse(c, kd, lap_op, dt), concs, diff_data)

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

def build_forward_step(icrn, spatial_dim, batch, integration_method, spatial_rate_constant=False, **kwargs):
    fast_dynamics, normal_dynamics = icrn.dynamics(spatial_dim, spatial_rate_constant)

    rxn_integrator = INT_METHOD_DICT[integration_method]

    def wm_f(conc_data, rate_data, _, dt):
        conc_data = fast_react(conc_data, fast_dynamics)
        conc_data = rxn_integrator(conc_data, rate_data, dt, normal_dynamics)
        return conc_data
    
    res_f = wm_f
    
    if spatial_dim is not None:
        lap_op = compute_lap_op(spatial_dim, dh=kwargs["dh"], dw=kwargs["dh"])
        def rd_f(conc_data, rate_data, diff_data, dt):
            conc_data = wm_f(conc_data, rate_data, diff_data, dt)
            return diffuse(conc_data, diff_data, lap_op, dt)
        res_f = rd_f
        
    if batch:
        reaction_in_axes = (0, 0, 0, None)
        return vmap(res_f, in_axes=reaction_in_axes)
    else:
        return res_f