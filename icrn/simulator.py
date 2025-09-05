from jax import lax, tree, jit, value_and_grad, checkpoint
import jax.numpy as jnp
import numpy as np
from numpy import empty
import math
from functools import partial
import optax
from .dict_utils import sjdict_builder, map1, map2
from .numerics import build_forward_step

INNER_SCAN_LENGTH = 1e3

class SimulatorError(Exception):
    pass

def _check_reaction_parameters():
    pass

def _check_reaction_dynamics():
    pass

def _check_concs():
    pass

class Experiment():
    def __init__(self, icrn, exp_params) -> None:

        print("compiling forward step function...")
        if exp_params["integration_method"] == "stochastic":
            pass
        else:
            self._forward_step_f = build_forward_step(icrn, **exp_params)
        print("done!")

        self._icrn = icrn 
        self._exp_params = exp_params

    @property
    def forward_step_f(self):
        return self._forward_step_f


    def simulate_segments(self, concs_data, rate_constant_data, diff_data, dt=None, segments=1, scan_length=INNER_SCAN_LENGTH):
        if dt is None:
            dt = self._exp_params['dt']
        
        def scan_helper(x, _):
            new_x = self.forward_step_f(x, rate_constant_data, diff_data, dt)
            return new_x, new_x
        
        @checkpoint
        def scan_inner(x,_ ): 
            scan_inner_state, _ = lax.scan(scan_helper, init=x, length=scan_length)
            return scan_inner_state, scan_inner_state
        
        sim_state, sim_hist = lax.scan(scan_inner, init=concs_data, length=segments)
        concs_expanded = map1(lambda x : jnp.expand_dims(x, 0), concs_data)
        return sim_state, map2(lambda x,y : jnp.concat([x,y]), concs_expanded, sim_hist)
    
    # want this to be jittable
    def simulate_time(self, concs_data, rate_constant_data, diff_data, time, dt=None, sample_num=1):
        if dt is None:
            dt = self._exp_params['dt']

        f_apps = int(math.ceil(time / dt))
        scan_length = int(f_apps / sample_num)

        return self.simulate_segments(concs_data, rate_constant_data, diff_data, segments=sample_num, scan_length=scan_length)
    
    def dict_builder(self, concs_spec={}, rate_constant_spec={}, diff_spec={}, batch_size=None):
        spatial_dim = self._exp_params["spatial_dim"]
        spatial_rate_constant = self._exp_params.get("spatial_rate_constant", False)
        shapes_dict = self._icrn.shapes()
        return sjdict_builder(shapes_dict, concs_spec, rate_constant_spec, diff_spec, batch_size, spatial_dim, spatial_rate_constant)