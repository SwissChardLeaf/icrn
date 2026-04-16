# TODO
# use nested vmaps for reactions
from abc import ABC, abstractmethod
from ..representation.reactions import (
    AbstractNormalReaction,
    AbstractFastReaction,
)
from ._integrator import Integrator


class _BaseAction(ABC):
    @abstractmethod
    def update_concs(self, concs_data, rate_constant_data, diff_data, dt):
        pass


class _NormalReactionAction(_BaseAction):
    def __init__(
        self,
        normal_reactions: list[AbstractNormalReaction],
        integrator: Integrator,
    ) -> None:
        self._normal_reactions = normal_reactions
        self._integrator = integrator

        for rxn in normal_reactions:
            rxn.flux_expr()

    def update_concs(self, concs_data, rate_constant_data, diff_data, dt):
        pass


class _FastReactionAction(_BaseAction):
    def __init__(
        self,
        normal_reactions: list[AbstractNormalReaction],
    ) -> None:
        pass

    def update_concs(self, concs_data, rate_constant_data, diff_data, dt):
        pass


class _DiffusionAction(_BaseAction):
    def __init__(self, diffusion_method):
        pass

    def update_concs(self, concs_data, rate_constant_data, diff_data, dt):
        pass


def _build_forward_step():
    action_list = _build_action_list
    return _build_forward_step_from_actions(action_list)


def _build_action_list(icrn, exp_params):
    reaction_groups = icrn.reaction_groups(spatial_dim, spatial_rate_constant)

    integration_method = exp_params["integration_method"]
    diffusion_method = exp_params["diffusion_method"]

    # rxn_integrator = INT_METHOD_DICT[integration_method]

    # def wm_f(conc_data, rate_data, _, dt):
    #     for group in reaction_groups:
    #         conc_data = group(conc_data, rate_data, dt)
    #     return conc_data

    # res_f = wm_f

    # spatial_dim = exp_params["spatial_dim"]
    # batch = exp_params["batch"]

    # if spatial_dim is not None:
    #     if diffusion_method == "spectral":
    #         lap_op = _compute_lap_op(spatial_dim, dh=exp_params["dh"], dw=exp_params["dw"])

    #         def spectral_rd_f(conc_data, rate_data, diff_data, dt):
    #             conc_data = wm_f(conc_data, rate_data, diff_data, dt)
    #             return _spectral_diffuse(conc_data, diff_data, lap_op, dt)

    #         res_f = spectral_rd_f
    #     else:

    #         def conv_rd_f(conc_data, rate_data, diff_data, dt):
    #             conc_data = wm_f(conc_data, rate_data, diff_data, dt)
    #             return _conv_diffuse(conc_data, diff_data, dt)

    #         res_f = conv_rd_f

    # if batch:
    #     reaction_in_axes = (0, 0, 0, None)
    #     return vmap(res_f, in_axes=reaction_in_axes)
    # else:
    #     return res_f


def _build_forward_step_from_actions(actions: list[_BaseAction]):
    pass
