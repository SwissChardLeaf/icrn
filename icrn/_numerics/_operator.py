from abc import ABC
from ..representation.reactions import AbstractReaction
from jax.lax import fori_loop

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

def _function_from_ops(ops: list[AbstractOperator]):
    return lambda state, non_state, dt, key: fori_loop(0, len(ops), lambda i, x: ops[i](x, non_state, dt, key), state)

class AbstractOperator(ABC):
    @abstractmethod
    def __call__(self, state, non_state, dt, key):
        pass

    @abstractmethod
    def __repr__(self):
        pass

class ReactionsOperator(AbstractOperator):
    def __init__(self, rxns: list[AbstractReaction], reaction_solver, mode):
        self.rxns = rxns
        self.reaction_solver = reaction_solver
        self.mode = mode
        self.op = _rxns_to_op(rxns, reaction_solver)

    def __call__(self, state, non_state, dt, key):
        return self.op(state, non_state, dt, key)

    def __repr__(self):
class FastReactionsOperator(AbstractOperator):
    def __init__(self, rxns: list[AbstractReaction], reaction_solver, mode):
        self.rxns = rxns
        self.reaction_solver = reaction_solver
        self.mode = mode
        self.op = _fast_rxns_to_op(rxns, reaction_solver)

    def __call__(self, state, non_state, dt, key):
        return _fast_rxns_to_op(self.rxns, self.reaction_solver)(state, non_state, dt, key)

    def __repr__(self):
        return f"{self.__class__.__name__}(rxns={self.rxns}, reaction_solver={self.reaction_solver})"

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

def _rxns_to_op(
    rxns: list[AbstractReaction], spatial_dims: int | None = None, reaction_solver="RK4"
):
    dyn_f = _rxns_to_dynamics(rxns)

    solver_f = None
    if reaction_solver == "RK4":
        solver_f = RK4_step
    elif reaction_solver == "Euler":
        solver_f = euler_step
    else:
        raise ValueError(f"Invalid reaction solver: {reaction_solver}")

    if spatial_dims:
        vmap_solver_f = solver_f

        for _ in range(spatial_dims):
            vmap_solver_f = jax.vmap(vmap_solver_f, in_axes=(0, None, None, None))

        def spatial_rxns_op(state, non_state, dt, key):
            return fori_loop(
                0,
                spatial_dims,
                lambda i, x: (vmap_solver_f(x, non_state, dt, key), key),
                state,
            )

        return spatial_rxns_op

    else:

        def rxns_op(state, non_state, dt, key):
            return solver_f(state, non_state, dyn_f, dt)

        return rxns_op

def _fast_rxns_to_op(state, non_state, dyn_f, dt):
    pass


def _diffusion_to_op(spatial_dim, diffusion_solver="spectral", boundary_conditions=None):
    lap = _compute_lap_op(spatial_dim)

    solver_f = None
    if diffusion_solver == "spectral":

        def spectral_diffusion_op(state, non_state, dt, key):
            return (_spectral_diffuse(lap, state, non_state, dt), key)

        return spectral_diffusion_op
    elif diffusion_solver == "convolutional":

        def convolutional_diffusion_op(state, non_state, dt, key):
            return (_convolutional_diffuse(state, non_state, dt), key)

        return convolutional_diffusion_op
    else:
        raise ValueError(f"Invalid diffusion solver: {diffusion_solver}")

def _to_lie_trotter_ops(self, space, reaction_solver, diffusion_solver, boundary_conditions):
    return [
        _rxns_to_op(self.reactions, reaction_solver),
        _diffusion_to_op(self.spatial_info, diffusion_solver, boundary_conditions),
    ]
    
def _to_strang_ops(self, space, reaction_solver, diffusion_solver, boundary_conditions):
    return [
        _rxns_to_op(self.reactions, reaction_solver),
        _diffusion_to_op(self.spatial_info, diffusion_solver, boundary_conditions),
        _rxns_to_op(self.reactions, reaction_solver)
    ]

def _to_well_mixed_ops(self, reaction_solver):
    return [
        _fast_rxns_to_op(self.)
        _rxns_to_op(self.reactions, reaction_solver)
    ]