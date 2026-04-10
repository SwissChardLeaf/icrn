from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from jax.experimental import checkify
from jax.lax import fori_loop
import jax.tree_util as jax_tree
from dataclasses import dataclass, field

from ..representation.reactions import AbstractReaction

'''
An operator is function that takes in a state and a non-state and returns a new state.

In the most common case, the state is a dict mapping species to their tensor of concentrations.
The non-state is a dict mapping rate constants to their values or species to their diffusion coefficients.
'''

# should be jaxtyped?
# def _ops_to_solver(ops: list, initial_state: PyTree[Float[Array, "?h *?foo"], "T"], rate_constants: PyTree[Float[Array, "*?foo"], "T"], diffusion_constants: PyTree[Float[Array, "*?foo"], "T"], dt: float):
#     def solver(state, non_state, dt, key):

#         for op in ops.values():
#             state = op(state, non_state, dt, key)
#         return state

#         fori_loop()

#     return solver

def _rxns_to_ops_lst(rxns: list[AbstractReaction], reaction_solver, spatial_info, splitting, diffusion_solver):
    if not spatial_info:
        return _rxn_well_mixed_to_ops_lst(rxns, reaction_solver)
    else:
        return _rxn_reaction_diffusion_to_ops_lst(rxns, reaction_solver, spatial_info, splitting, diffusion_solver)

def _function_from_ops_lst(ops: list[AbstractOperator]):
    return lambda state, non_state, dt, key: fori_loop(0, len(ops), lambda i, x: ops[i](x, non_state, dt, key), state)

def _check_state_is_non_negative(state):
    leaves = jax_tree.tree_leaves(
        jax_tree.tree_map(lambda x: jnp.all(x >= 0), state)
    )
    if not leaves:
        return jnp.array(True)
    return jnp.all(jnp.stack(leaves))

class AbstractOperator(ABC):
    def __init__(self, mode: str | None):
        update_f = self.update_state

        if mode == "strict":
            def checked_update_f(key, state, non_state, dt):
                key_out, state_out = update_f(key, state, non_state, dt)
                checkify.check(
                    _check_state_is_non_negative(state_out), "state is negative"
                )
                return key_out, state_out

            checked_f = checkify.checkify(checked_update_f)

            def new_update_f(key, state, non_state, dt):
                err, out = checked_f(key, state, non_state, dt)
                checkify.check_error(err)
                return out

            self._update_f = new_update_f

        elif mode == "relu":
            def relu_update_f(key, state, non_state, dt):
                key_out, state_out = out = update_f(key, state, non_state, dt)
                # update_state returns (key, state); only clamp concentrations, not the key.
                return key_out, jax_tree.tree_map(jax.nn.relu, state_out)

            self._update_f = relu_update_f

        elif mode is None:
            self._update_f = update_f
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def update_with_checks(self, key, state, non_state, dt):
        return self._update_f(key, state, non_state, dt)

    @abstractmethod
    def update_state(self, key, state, non_state, dt):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(aux={self.aux}, mode={self.mode})"

class ReactionsOperator(AbstractOperator):
    def __init__(self, mode: str | None, rxns: Iterator[AbstractReaction], reaction_solver):
        for rxn in rxns:
            if not isinstance(rxn, AbstractReaction):
                raise ValueError(f"rxns must be an iterator of AbstractReactions, got {rxn} of type {type(rxn)}")

        if reaction_solver == "RK4":
            reaction_solver_f = _RK4_step
        elif reaction_solver == "Euler":
            reaction_solver_f = _euler_step
        else:
            raise ValueError(f"Invalid reaction solver: {reaction_solver}")
            
        _rxns_to_dynamics_f = _rxns_to_dynamics(rxns)

        def rxn_update_f(key, state, non_state, dt):
            new_state = reaction_solver_f(state, non_state, _rxns_to_dynamics_f, dt)
            return key, new_state

        self._rxn_update_f= _rxns_to_update_f(rxns, reaction_solver)
        super().__init__(mode)
        # self.update_state_f = _rxns_to_update_f(rxns, reaction_solver)

    def update_state(self, key, state, non_state, dt):
        return self._rxn_update_f(key, state, non_state, dt)

    def __repr__(self):
        return f"{self.__class__.__name__}(rxns={self.rxns}, reaction_solver={self.reaction_solver})"

class FastReactionsOperator(AbstractOperator):
    def __init__(self, fast_rxns: Iterable[FastReaction]):
        for rxn in fast_rxns:
            if not isinstance(rxn, FastReaction):
                raise ValueError(f"fast_rxns must be an iterator of FastReactions, got {rxn} of type {type(rxn)}")

        self._fast_rxns_update_f = _fast_rxns_to_update_f(fast_rxns)
        super().__init__(None)

    def update_state(self, key, state, non_state, dt):
        return self._fast_rxns_update_f(key, state, non_state, dt)

    def __repr__(self):
        return f"{self.__class__.__name__}(fast_rxns={self.fast_rxns})"

class SpectralDiffusionOperator(AbstractOperator):
    def __init__(self, spatial_dims, dxs, mode):
        self.spatial_dims = spatial_dims
        self.diffusion_solver = diffusion_solver
        self.mode = mode

    def __call__(self, state, non_state, dt, key):
        return _diffusion_to_op(self.spatial_dims, self.diffusion_solver)(state, non_state, dt, key)

    def __repr__(self):
        return f"{self.__class__.__name__}(xs={self.xs}, dxs={self.dxs}, diffusion_solver={self.diffusion_solver})"

class ConvolutionalDiffusionOperator(AbstractOperator):
    def __init__(self, spatial_dims: int, diffusion_solver="convolutional"):
        self.spatial_dims = spatial_dims
        self.diffusion_solver = diffusion_solver

    def __call__(self, state, non_state, dt, key):
        return _convolutional_diffuse(self.spatial_dims, self.diffusion_solver)(state, non_state, dt, key)

    def __repr__(self):
        return f"{self.__class__.__name__}(xs={self.xs}, dxs={self.dxs}, diffusion_solver={self.diffusion_solver})"

# def _rxns_to_op(
#     rxns: list[AbstractReaction], 
#     spatial_dims: int = 0, 
#     reaction_solver="RK4"
# ):
#     dyn_f = _rxns_to_dynamics(rxns)

#     solver_f = None
#     if reaction_solver == "RK4":
#         solver_f = RK4_step
#     elif reaction_solver == "Euler":
#         solver_f = euler_step
#     else:
#         raise ValueError(f"Invalid reaction solver: {reaction_solver}")

#     res_f = 

#     if spatial_dims > 0:
#         vmap_solver_f = solver_f

#         for _ in range(spatial_dims):
#             vmap_solver_f = jax.vmap(vmap_solver_f, in_axes=(0, None, None, None))

#         def spatial_rxns_op(state, non_state, dt, key):
#             return fori_loop(
#                 0,
#                 spatial_dims,
#                 lambda i, x: (vmap_solver_f(x, non_state, dt, key), key),
#                 state,
#             )

#         return spatial_rxns_op

#     else:
#         def rxns_op(state, non_state, dt, key):
#             return solver_f(state, non_state, dyn_f, dt)

#         return rxns_op

# def _fast_rxns_to_op(state, non_state, dyn_f, dt):
#     pass


# def _diffusion_to_op(spatial_dim, diffusion_solver="spectral", boundary_conditions=None):
#     lap = _compute_lap_op(spatial_dim)

#     solver_f = None
#     if diffusion_solver == "spectral":

#         def spectral_diffusion_op(state, non_state, dt, key):
#             return (_spectral_diffuse(lap, state, non_state, dt), key)

#         return spectral_diffusion_op
#     elif diffusion_solver == "convolutional":

#         def convolutional_diffusion_op(state, non_state, dt, key):
#             return (_convolutional_diffuse(state, non_state, dt), key)

#         return convolutional_diffusion_op
#     else:
#         raise ValueError(f"Invalid diffusion solver: {diffusion_solver}")

# def _to_lie_trotter_ops(self, space, reaction_solver, diffusion_solver, boundary_conditions):
#     return [
#         _rxns_to_op(self.reactions, reaction_solver),
#         _diffusion_to_op(self.spatial_info, diffusion_solver, boundary_conditions),
#     ]
    
# def _to_strang_ops(self, space, reaction_solver, diffusion_solver, boundary_conditions):
#     return [
#         _rxns_to_op(self.reactions, reaction_solver),
#         _diffusion_to_op(self.spatial_info, diffusion_solver, boundary_conditions),
#         _rxns_to_op(self.reactions, reaction_solver)
#     ]

# def _to_well_mixed_ops(self, reaction_solver):
#     return [
#         _fast_rxns_to_op(self.)
#         _rxns_to_op(self.reactions, reaction_solver)
#     ]