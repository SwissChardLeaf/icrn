# reaction class
# fast reaction
# michaelis menten reaction
# ICRN class
# action group class

"""
This module contains the building blocks of ICRNs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable
from collections import defaultdict
from ..utils.dict_utils import dict_add, arr_mul

from .symbols import (
    Complex,
    Numeric,
    Species,
    TensorExpression,
    TensorFunction,
    TensorLiteral,
    TensorSymbol,
)

from .._numerics._mass_action import mass_action_flux_f
from .._numerics._fast_update import _fast_update_f


type TensorSymbolDict = dict[TensorSymbol]


@dataclass(frozen=True)
class AbstractReaction(ABC):
    reactants: Complex
    products: Complex
    aux: TensorExpression | None = None

    def __str__(self):
        return str(self.reactants) + " -> " + str(self.products) + " " + str(self.aux)

    def __repr__(self):
        return repr((self.reactants, self.products, self.aux))

    @abstractmethod
    def flux(self):
        pass


def rxns_to_dynamics_f(rxns: list[AbstractReaction]):
    flux_fs = list(map(lambda r: r.flux(), rxns))

    def dyn_f(state, non_state):
        dynamics = {k : 0 for k in state.keys()}
        for flux_f in flux_fs:
            val_dict = flux_f(state, non_state)

            for k, v in val_dict.items():
                dynamics[k] += v

        return dynamics

    return dyn_f


def _matching_shapes(all_species: set[Species]):
    shapes = dict()

    for s in all_species:
        base_s = s[()]
        shape_s = tuple(
            map(lambda x: x.index_set if x.index_set > 0 else None, s.index_symbols)
        )

        if base_s in shapes:
            existing_shape = shapes[base_s]
            if len(existing_shape) != len(shape_s):
                raise ValueError(
                    f"Shapes of {s} must be the same, got {existing_shape} and {shape_s}"
                )

            update_shape = []
            for i in range(len(existing_shape)):
                if existing_shape[i] and shape_s[i]:
                    if existing_shape[i] != shape_s[i]:
                        raise ValueError(
                            f"Shapes of {base_s} must be the same, got {existing_shape} and {shape_s}"
                        )
                elif not existing_shape[i] or not shape_s[i]:
                    update_shape.append(None)
                else:
                    update_shape.append(shape_s[i])

            shapes[base_s] = tuple(update_shape)
        else:
            shapes[base_s] = shape_s


class MassActionReaction(AbstractReaction):
    def __init__(
        self,
        reactants: Complex,
        products: Complex,
        rate_expr: TensorExpression,
    ):
        if not (isinstance(reactants, Complex | Species) or reactants == 0):
            raise ValueError(
                f"Reactants must be a Complex or Species or 0, got {type(reactants)}"
            )
        if isinstance(reactants, Species):
            reactants = Complex({reactants: 1})
        elif isinstance(reactants, int) and reactants == 0:
            reactants = Complex({})

        if not (isinstance(products, Complex | Species) or products == 0):
            raise ValueError(
                f"Products must be a Complex or Species, got {type(products)}"
            )
        if isinstance(products, Species):
            products = Complex({products: 1})
        elif isinstance(products, int) and products == 0:
            products = Complex({})

        if not isinstance(rate_expr, TensorExpression | Numeric) or isinstance(
            rate_expr, Species
        ):
            raise ValueError(
                f"Rate expression must be a TensorExpression (not Species)or Numeric, got {type(rate_expr)}"
            )
        if isinstance(rate_expr, Numeric):
            rate_expr = TensorLiteral(rate_expr)

        try:
            reactants_set = set(reactants.count_dict.keys())
            products_set = set(products.count_dict.keys())
            _matching_shapes(reactants_set | products_set)
        except ValueError:
            raise

        object.__setattr__(self, "reactants", reactants)
        object.__setattr__(self, "products", products)
        object.__setattr__(self, "aux", rate_expr)

    @property
    def rate_expr(self):
        return self.aux

    @rate_expr.setter
    def rate_expr(self, rate_expr):
        self.aux = rate_expr

    # should produce a function that takes in the state and rate_constant_data
    # tensors and results in the dictionary of fluxes.
    # the function is written in a way that is reminiscent of lazy evaluation
    # we want to control when the bulk of evaluation actually occurs.
    def flux(self):
        return mass_action_flux_f(self.reactants, self.products, self.aux)

    def shapes(self):
        pass

def fast_rxns_to_update_f(fast_rxns: list[FastReaction]):
    update_fs = [_fast_update_f(frxn.reactants, frxn.products) for frxn in fast_rxns]

    def _fast_updates_f(state):
        for f in update_fs:
            fast_update = f(state)
            for s, v in fast_update.items():
                state[s] += v
        return state

    return _fast_updates_f

@dataclass(frozen=True)
class FastReaction():
    reactants: Complex | Species
    products: Complex | Species | int
    

    def __post_init__(self):
        reactants = self.reactants
        products = self.products

        if not (isinstance(reactants, Complex | Species)):
            raise ValueError(
                f"Reactants must be a Complex or Species, got {type(reactants)}"
            )

        if isinstance(reactants, Species):
            reactants = Complex({reactants: 1})

        reactants_index_symbols = None
        for s in reactants.count_dict.keys():
            if reactants_index_symbols:
                if reactants_index_symbols != s.index_symbols:
                    raise ValueError(
                        f"The index symbols must be the same for all species in the reactants, got {reactants_index_symbols} and {s.index_symbols}"
                    )
            else:
                reactants_index_symbols = s.index_symbols

        if not (isinstance(products, Complex | Species) or products == 0):
            raise ValueError(
                f"Products must be a Complex or Species or 0, got {type(products)}"
            )
        if isinstance(products, Species):
            products = Complex({products: 1})
        elif isinstance(products, int) and products == 0:
            products = Complex({})

        for s in products.count_dict.keys():
            if set(s.index_symbols) - set(reactants_index_symbols) != set():
                raise ValueError(
                    f"The index symbols of the products must be a subset of the index symbols of the reactants, got {set(s.index_symbols)} and {index_symbols_set}"
                )

        object.__setattr__(self, "reactants", reactants)
        object.__setattr__(self, "products", products)


    # @rate_constant_expr.setter
    # def index_set(self, index_set):
    #     self.aux = index_set

    # def __init__(self, reactants, products, rate_constant_expr):
    #     pass

    # def flux_expr(self) -> dict[Species, Expr]:
    #     r_dict = reactants.count_dict
    #     p_dict = products.count_dict

    #     flux_expr = 1
    #     for sp, coeff in r_dict.items():
    #         flux_expr *= sp ** coeff

    #     all_species = r_dict.keys() | p_dict.keys()

    #     for sp in all_species:
    #         count_diff = p_dict[sp] - r_dict[sp]

    #         if count_diff == 0:
    #             continue
    #         else:


# @dataclass(frozen=True)
# class AbstractFastReaction(ABC):
#     @abstractmethod
#     def __str__(self):
#         pass

#     @abstractmethod
#     def __repr__(self):
#         pass

#     @abstractmethod
#     def update(self, data: DataDict) -> DataDict:
#         pass


# # class MichaelisMentenReaction(AbstractReaction):
# #     def __init__(self, substrate, enzyme, product, rate_constant, aux, name=None):
# #         pass
