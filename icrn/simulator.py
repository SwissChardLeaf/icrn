from jax import lax, tree, jit, value_and_grad, checkpoint
import jax.numpy as jnp
from numpy import empty
import math
from .compiler import build_forward_step
from functools import partial
import optax
from .dict_utils import sjdict_builder, map1, map2

INNER_SCAN_LENGTH = 1e3

class SimulatorError(Exception):
    pass

def _check_reaction_parameters():
    pass

def _check_reaction_dynamics():
    pass

def _check_concs():
    pass

def dict_builder():
    pass

def use_loss_fn(weights, sim_fn, loss_fn, xs, ys):
    simulated_state = sim_fn(weights, xs)
    return loss_fn(simulated_state, ys)

def train_step(weights, sim_fn, loss_fn, xs, ys, optimizer, opt_state, train_params):
    grads, state = value_and_grad(use_loss_fn)(weights, sim_fn, loss_fn, xs, ys)
    if optimizer is not None:
        updates, opt_state = optimizer.update(grads, opt_state)

    weights = optax.apply_updates(weights, updates)

    return weights, state

class Experiment():
    def __init__(self, icrn, exp_params) -> None:

        print("compiling forward step function...")
        self.forward_step_f = build_forward_step(icrn, **exp_params)
        print("done!")

        self._icrn = icrn 
        self._exp_params = exp_params

    def forward_step(self, concs, rate_data):
        return self.forward_step_f(concs, rate_data, self._exp_params['dt'])

    def simulate_timeline(self, timeline):
        pass

    def simulate_segments(self, concs, rate_data, diff_data=None, segments=1, scan_length=INNER_SCAN_LENGTH):
        def scan_helper(x, _):
            new_x = self.forward_step_f(x, rate_data, diff_data, self._exp_params['dt'])
            return new_x, new_x
        
        @checkpoint
        def scan_inner(x,_ ): 
            scan_inner_state, _ = lax.scan(scan_helper, init=x, length=scan_length)
            return scan_inner_state, scan_inner_state
        
        return lax.scan(scan_inner, init=concs, length=segments)
    
    # want this to be jittable
    def simulate_time(self, concs, rate_data, diff_data, time, sample_num=1):
        f_apps = int(math.ceil(time / self._exp_params["dt"]))
        scan_length = int(f_apps / sample_num)

        sim_state, sim_hist = self.simulate_segments(concs, rate_data, diff_data, segments=sample_num, scan_length=scan_length)
        concs_expanded = map1(lambda x : jnp.expand_dims(x, 0), concs)
        return sim_state, map2(lambda x,y : jnp.concat([x,y]), concs_expanded, sim_hist)
    
    def dict_builder(self, concs_spec={}, rate_data_spec={}, diff_data_spec={}, batch_size=None):
        spatial_dim = self._exp_params["spatial_dim"]
        shapes_dict = self._icrn.shapes()
        return sjdict_builder(shapes_dict, concs_spec, rate_data_spec, diff_data_spec, batch_size, spatial_dim)
    
    def extract_timeline(self, timeline):
        loss_fn = timeline.loss_fn(self._icrn)
        end_time = timeline.end_time()
        concs_spec_fn, rate_data_spec_fn = timeline.init_tensor_spec

        def tensor_builder(weights, xs):
            concs_spec = concs_spec_fn(xs)
            rate_data_spec = rate_data_spec_fn(xs)
            return sjdict_builder(self._icrn, self._exp_params, concs_spec, rate_data_spec)

        scan_fn = self.simulate_time_samples

        def sim_fn(weights, xs):
            concs, rate_data = tensor_builder(weights, xs)
            return scan_fn(concs, rate_data, end_time, 1)

        return sim_fn, loss_fn
    
    def train(self, xs, ys, weights, timeline, train_params):
        optimizer = train_params["optimizer"]
        opt_state = None
        if optimizer is not None:
            opt_state = optimizer.init()

        best_weights = None
        best_loss = None

        sim_fn, loss_fn = self.extract_timeline(timeline)

        for epoch in range(train_params["epochs"]):
            weights, opt_state, loss_value, simulated_state = train_step(weights, sim_fn, loss_fn, xs, ys, optimizer, opt_state, train_params)

            if best_loss is None or loss_value < best_loss:
                best_loss = loss_value
                best_weights = weights
        
        return weights

def build_train_step():
    pass