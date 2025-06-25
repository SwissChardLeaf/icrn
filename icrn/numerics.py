import jax
from jax import lax
# from jax.nn import relu
from jax import numpy as jnp
from jax.numpy.fft import fftfreq, fftn, ifftn
from .dict_utils import map1, map2

def compute_lap_op(spatial_dim, dh):
    h, w = spatial_dim

    kx = fftfreq(h, d=dh) * 2 * jnp.pi
    ky = fftfreq(w, d=dh) * 2 * jnp.pi

    Kx, Ky = jnp.meshgrid(kx, ky)
    Kx = jnp.transpose(Kx, axes=[1, 0])
    Ky = jnp.transpose(Ky, axes=[1, 0])
    return (-(Kx**2) - (Ky**2))

def _species_diffuse(conc, kd, lap_op, dt):
    # concs shape is (H, W, N)
    x_hat = fftn(conc, axes=[0, 1])
    broadcast_shape = lap_op.shape + kd.shape
    for i in range(len(kd.shape)):
        lap_op = jnp.expand_dims(lap_op, axis=-1)
    x_hat = x_hat / (1 - dt * jnp.broadcast_to(kd[None, None, ...], broadcast_shape) * jnp.broadcast_to(lap_op, broadcast_shape))
    return ifftn(x_hat, axes=[0, 1]).real

def flatten(arr, spatial_dim, batch):
    arr.shape 

def unflatten(arr, spatial_dim, batch):
    return 

def extract_batch_dim(concs, batch_size, sp):
    def helper(a):
        lead_dim = a.shape[0]
        
    return map1(helper, concs)

def return_batch_dim(concs):
    return

def diffuse(concs, diff_data, lap_op, dt):
    # concs shape is (H * W, N)
    return map2(lambda c, kd: _species_diffuse(c, kd, lap_op, dt), concs, diff_data)

def fast_react(concs, fast_func):
    delta = fast_func(concs)
    return concs + delta

def euler(concs, rate_data, dt, dynamics_func):
    dynamics_dict = dynamics_func(rate_data | concs)
    return map1(jax.nn.relu, concs + dynamics_dict * dt)

def RK4(concs, rate_data, dt, dynamics_func):
    k1 = dynamics_func(concs | rate_data)
    k2 = dynamics_func(concs + k1 * dt * 0.5 | rate_data)
    k3 = dynamics_func(concs + k2 * dt * 0.5 | rate_data)
    k4 = dynamics_func(concs + k3 * dt | rate_data)

    return concs + (k1 + (k2 * 2) + (k3 * 2) + k4) * (dt / 6)

def relu_RK4(concs, rate_data, dt, dynamics_func):
    k1 = dynamics_func(map1(jax.nn.relu, concs) | rate_data)
    k2 = dynamics_func(map1(jax.nn.relu, concs + k1 * dt * 0.5) | rate_data)
    k3 = dynamics_func(map1(jax.nn.relu, concs + k2 * dt * 0.5) | rate_data)
    k4 = dynamics_func(map1(jax.nn.relu, concs + k3 * dt) | rate_data)
    return map1(jax.nn.relu, concs + (k1 + (k2 * 2) + (k3 * 2) + k4) * (dt / 6))

def RK4_5(concs, dynamics, dt):
    pass

INT_METHOD_DICT ={
    "euler" : euler,
    "RK4" : RK4,
    "relu_RK4" : relu_RK4
}