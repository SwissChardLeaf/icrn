class ReactionGroup:
    pass


class ICRN:
    """
    Class for representing an ICRN.
    """

    def __init__(self, reactions) -> None:
        # a list of lists
        self._reactions = reactions

        normal_reactions = []
        fast_reactions = []

        for rxn in reactions:
            if isinstance(rxn, FastReaction):
                fast_reactions.append(rxn)
            else:
                normal_reactions.append(rxn)

        self._normal_reactions = normal_reactions
        self._fast_reactions = fast_reactions

    @property
    def reactions(self):
        return self._reactions

    # @property
    # def normal_reactions(self):
    #     return self._normal_reactions

    # @property
    # def fast_reactions(self):
    #     return self._fast_reactions

    def __repr__(self):
        return repr(self.reactions) + repr(self.fast_reactions)

    def shapes(self):
        return {
            s: shape
            for reaction in self._reactions
            for s, shape in reaction.shapes().items()
        }

    def dynamics(self, spatial_dim, spatial_rate_constant):
        pass
        # def jittable_fast_dynamics(tensor_data):
        #     dynamics_dict = dict()

        #     for rxn in self.fast_reactions:
        #         for s, arr in rxn.flux(tensor_data).items():
        #             dynamics_dict[s] = dynamics_dict.get(s, 0) + arr

        #     return dynamics_dict

        # flux_list = [
        #     rxn.build_flux(spatial_dim, spatial_rate_constant)
        #     for rxn in self.normal_reactions
        # ]

        # def jittable_normal_dynamics(tensor_data):
        #     dynamics_dict = dict()

        #     for flux in flux_list:
        #         for s, arr in flux(tensor_data).items():
        #             dynamics_dict[s] = dynamics_dict.get(s, 0) + arr

        #     return dynamics_dict

        # return jittable_fast_dynamics, jittable_normal_dynamics

    def enumerate(self):
        return ICRN([enum_r for r in self.reactions for enum_r in r.enumerate()])
