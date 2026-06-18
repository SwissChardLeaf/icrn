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
from collections import defaultdict
from dataclasses import dataclass

from ._internal._fast_update import _fast_update_f
from ._internal._mass_action import (
    mass_action_flux_f,
    mass_action_flux_pairs_f,
    mass_action_flux_pd_f,
)
from ._internal._reaction_yaml import (
    ReactionSymbolTable,
    _complex_from_yaml,
    _complex_to_yaml,
    _rate_from_yaml,
    _rate_to_yaml,
    load_network_yaml,
    register_yaml_type,
    save_network_yaml,
)
from .symbols import (
    Complex,
    Numeric,
    Species,
    TensorExpression,
    TensorLiteral,
)


@dataclass(frozen=True)
class AbstractReaction(ABC):
    """Abstract base for reactions.

    Subclasses must implement `flux()`, which returns a callable mapping
    `(state, non_state)` to a per-species flux dictionary. Users looking for
    non mass action kinetics should extend this class.

    Parameters
    ----------
    reactants : Complex or Species of 0
        A multiset of indexed species.
    products : Complex
        A multiset of indexed species.
    aux : TensorExpression or None, optional
        Subclass-specific auxiliary data, e.g. the rate expression
        for [`MassActionReaction`][icrn.MassActionReaction].

    Attributes
    ----------
    See parameters.

    Notes
    -----
    User should extend this class to implement the `flux` and `flux_pd` methods.
    The class is inspired by lazy evaluation. The `flux` and `flux_pd` methods
    return callables instead of numerically evaluating values. This is useful
    because the `flux` and `flux_pd` methods are expensive to compile and the
    `flux` and `flux_pd` will be called repeatedly during integration.

    See Also
    --------
    [`MassActionReaction`][icrn.MassActionReaction] : Standard mass-action
        reaction.
    [`FastReaction`][icrn.FastReaction] : Limiting-reagent reaction.
    """

    reactants: Complex
    products: Complex
    aux: TensorExpression | None = None

    def __str__(self):
        return (
            str(self.reactants)
            + " -> "
            + str(self.products)
            + " "
            + str(self.aux)
        )

    def __repr__(self):
        return repr((self.reactants, self.products, self.aux))

    @abstractmethod
    def flux(self):
        """Return a callable that evaluates this reaction's net flux.

        The returned function is compiled from the reaction definition and
        may be called repeatedly during integration. Numerical evaluation is
        deferred until the callable is invoked.

        Returns
        -------
        callable
            A function ``f(state, non_state)`` returning a dictionary of
            per-species net flux contributions (production minus
            destruction).

        See Also
        --------
        flux_pd : Production/destruction split for Patankar integrators.
        """
        pass

    @abstractmethod
    def flux_pd(self):
        """Return a callable that evaluates production and destruction rates.

        The returned function splits each species' net flux into non-negative
        production and destruction terms. This split is used by Patankar-type
        integrators that treat destruction implicitly.

        Returns
        -------
        callable
            A function ``f(state, non_state)`` returning
            ``(production, destruction)``, each a dictionary mapping base
            species to non-negative rates.

        See Also
        --------
        flux : Net per-species flux for standard integrators.
        """
        pass

    @abstractmethod
    def flux_pairs(self, split="uniform"):
        """Return a callable that evaluates production and destruction pairs.

        The returned function splits each species' net flux into non-negative
        production and destruction terms, and attributes the production to the
        reactant species that caused it. This split is used by the modified
        Patankar method.

        Parameters
        ----------
        split : {"uniform", "stoichiometry"}, optional
            Strategy used to distribute each product's production across the
            reactants. See
            [`_split_weights`][icrn._internal._mass_action._split_weights].
            Defaults to ``"uniform"``.

        Returns
        -------
        callable
            A function ``f(state, non_state)`` returning a tuple
            ``(destruction, pairs, explicit)`` where
            - ``destruction`` maps a base species to its destruction
              rate (used to build the matrix diagonal),
            - ``pairs`` maps an ordered ``(product, reactant)`` pair of base
              species to the production of the product attributed to that
              reactant (used to build the off-diagonal entries),
            - ``explicit`` maps a base species to production with no reactant
              source (a sourceless influx), to be added explicitly.
        """
        pass

    def to_dict(self) -> dict:
        """Return a YAML-serializable description of this reaction.

        Subclasses that support YAML I/O should override this method and
        register with `register_yaml_type`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support YAML serialization."
        )

    @classmethod
    def from_dict(cls, entry: dict, symbols: ReactionSymbolTable):
        """Reconstruct a reaction from a YAML entry and symbol table."""
        raise NotImplementedError(
            f"{cls.__name__} does not support YAML deserialization."
        )

    @classmethod
    def register_yaml_type(cls, type_name: str, reaction_cls: type):
        """Register a reaction class for use with `from_dict` / `load_yaml`."""
        register_yaml_type(type_name, reaction_cls)

    @classmethod
    def from_dict_entry(cls, entry: dict, symbols: ReactionSymbolTable):
        """Dispatch ``entry['type']`` to a registered reaction class."""
        from ._internal._reaction_yaml import dispatch_from_dict

        return dispatch_from_dict(entry, symbols)

    @staticmethod
    def save_yaml(path, rxns):
        """Write a list of reactions to a YAML network file."""
        save_network_yaml(path, rxns)

    @classmethod
    def load_yaml(cls, path):
        """Load a reaction network from a YAML file.

        Returns
        -------
        reactions : list of AbstractReaction or FastReaction
        symbols : ReactionSymbolTable
        """
        return load_network_yaml(path)


def rxns_to_dynamics_f(rxns: list[AbstractReaction]):
    flux_fs = list(map(lambda r: r.flux(), rxns))

    def dyn_f(state, non_state):
        dynamics = {k: 0 for k in state.keys()}
        for flux_f in flux_fs:
            val_dict = flux_f(state, non_state)

            for k, v in val_dict.items():
                dynamics[k] += v

        return dynamics

    return dyn_f


def rxns_to_pd_dynamics_f(rxns: list[AbstractReaction]):
    flux_pd_fs = list(map(lambda r: r.flux_pd(), rxns))

    def pd_f(state, non_state):
        production = {k: 0 for k in state.keys()}
        destruction = {k: 0 for k in state.keys()}
        for flux_pd_f in flux_pd_fs:
            prod_dict, dest_dict = flux_pd_f(state, non_state)

            for k, v in prod_dict.items():
                production[k] += v
            for k, v in dest_dict.items():
                destruction[k] += v

        return production, destruction

    return pd_f


def rxns_to_mpe_dynamics_f(
    rxns: list[AbstractReaction],
    split: str = "uniform",
):
    """Aggregate the modified-Patankar terms of a reaction network.

    Combines the per-reaction
    [`flux_pairs`][icrn.MassActionReaction.flux_pairs] callables into a
    single function that returns the lumped destruction, the source-attributed
    production pairs, and the sourceless (explicit) production for the whole
    network. This is the dynamics function consumed by
    [`_mpe_step`][icrn._internal._reaction_numerics._mpe_step].

    Parameters
    ----------
    rxns : list of AbstractReaction
        Reactions making up the network.
    split : {"uniform", "stoichiometry"}, optional
        Strategy used to distribute each product's production across the
        reactants of the producing reaction. Defaults to ``"uniform"``.

    Returns
    -------
    callable
        A function ``mpe_f(state, non_state)`` returning a tuple
        ``(destruction, pairs, explicit)`` accumulated over all reactions,
        with ``destruction`` and ``explicit`` defined for every species in
        ``state``.

    See Also
    --------
    rxns_to_pd_dynamics_f : Lumped production/destruction without source
        attribution.
    """
    pair_fs = list(map(lambda r: r.flux_pairs(split), rxns))

    def mpe_f(state, non_state):
        destruction = {k: 0.0 for k in state.keys()}
        explicit = {k: 0.0 for k in state.keys()}
        pairs = defaultdict(lambda: 0.0)
        for pair_f in pair_fs:
            dest_dict, pair_dict, explicit_dict = pair_f(state, non_state)

            for k, v in dest_dict.items():
                destruction[k] += v
            for k, v in explicit_dict.items():
                explicit[k] += v
            for key, v in pair_dict.items():
                pairs[key] += v

        return destruction, dict(pairs), explicit

    return mpe_f


def _matching_shapes(all_species: set[Species]):
    shapes = dict()

    for s in all_species:
        base_s = s[()]
        shape_s = tuple(
            map(
                lambda x: x.index_set if x.index_set > 0 else None,
                s.index_symbols,
            )
        )

        if base_s in shapes:
            existing_shape = shapes[base_s]
            if len(existing_shape) != len(shape_s):
                raise ValueError(
                    f"Shapes of {s} must be the same, got {existing_shape} and {
                        shape_s
                    }"
                )

            update_shape = []
            for i in range(len(existing_shape)):
                if existing_shape[i] and shape_s[i]:
                    if existing_shape[i] != shape_s[i]:
                        raise ValueError(
                            f"Shapes of {base_s} must be the same, got "
                            f"{existing_shape} and {shape_s}"
                        )
                elif not existing_shape[i] or not shape_s[i]:
                    update_shape.append(None)
                else:
                    update_shape.append(shape_s[i])

            shapes[base_s] = tuple(update_shape)
        else:
            shapes[base_s] = shape_s


class MassActionReaction(AbstractReaction):
    r"""A mass-action reaction with a (possibly indexed) rate expression.

    The instantaneous flux of the reaction at species concentrations $x$
    is `rate_expr * prod(x[s] ** count for s, count in reactants)`,
    contributing $+(c_\text{product} - c_\text{reactant}) \cdot \text{flux}$
    to each species (with $c$ the stoichiometric coefficient).

    Parameters
    ----------
    reactants : Complex or Species or 0
        A multiset of indexed species.
    products : Complex or Species or 0
        A multiset of indexed species.
    rate_expr : TensorExpression or Numeric
        scalar rate (wrapped automatically in a
        [`TensorLiteral`][icrn.TensorLiteral]) or an indexed
        [`RateConstant`][icrn.RateConstant] /
        [`TensorExpression`][icrn.TensorExpression].

    Attributes
    ----------
    reactants : Complex
        Input reactants are turned into a Complex object.
    products : Complex
        Input products are turned into a Complex object.
    rate_expr : TensorExpression
        Alias for `aux`.
    """

    def __init__(
        self,
        reactants: Complex,
        products: Complex,
        rate_expr: TensorExpression,
    ):
        if not (isinstance(reactants, Complex | Species) or reactants == 0):
            raise ValueError(
                f"Reactants must be a Complex or Species or 0, got {
                    type(reactants)
                }"
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
                f"Rate expression must be a TensorExpression (not Species)"
                f"or Numeric, got {type(rate_expr)}"
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

    def flux_pd(self):
        return mass_action_flux_pd_f(self.reactants, self.products, self.aux)

    def flux_pairs(self, split="uniform"):
        return mass_action_flux_pairs_f(
            self.reactants,
            self.products,
            self.aux,
            split=split,
        )

    def to_dict(self) -> dict:
        return {
            "type": "mass_action",
            "reactants": _complex_to_yaml(self.reactants),
            "products": _complex_to_yaml(self.products),
            "rate": _rate_to_yaml(self.aux),
        }

    @classmethod
    def from_dict(cls, entry: dict, symbols: ReactionSymbolTable):
        if entry["type"] != "mass_action":
            raise ValueError(
                f"Expected type 'mass_action', got {entry['type']!r}"
            )
        return cls(
            _complex_from_yaml(entry.get("reactants", []), symbols),
            _complex_from_yaml(entry.get("products", []), symbols),
            _rate_from_yaml(entry["rate"], symbols),
        )

    def shapes(self):
        pass


AbstractReaction.register_yaml_type("mass_action", MassActionReaction)


def fast_rxns_to_update_f(fast_rxns: list[FastReaction]):
    update_fs = [
        _fast_update_f(frxn.reactants, frxn.products) for frxn in fast_rxns
    ]

    def _fast_updates_f(state):
        for f in update_fs:
            fast_update = f(state)
            for s, v in fast_update.items():
                state[s] += v
        return state

    return _fast_updates_f


@dataclass(frozen=True)
class FastReaction:
    """An infinitely fast reaction.

    At every integration step, reactions fire until the reactants are fully
    consumed. They are useful for modelling reactions that are on a timescale
    much faster than the dt. Currently, a concrete species can be a reactant in
    at most one fast reaction.
    Parameters
    ----------
    reactants : Complex or Species
        All reactants must be indexed by the same tuples.
    products : Complex or Species or 0
        Each product's index symbols must be a subset of the reactants' index
        symbols. `0` denotes a sink (no products).

    Attributes
    ----------
    reactants : Complex
        Input reactants are turned into a Complex object.
    products : Complex
        Input products are turned into a Complex object.
    """

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
                        f"The index symbols must be the same for all "
                        f"species in the reactants, got "
                        f"{reactants_index_symbols} and {s.index_symbols}"
                    )
            else:
                reactants_index_symbols = s.index_symbols

        if not (isinstance(products, Complex | Species) or products == 0):
            raise ValueError(
                f"Products must be a Complex or Species or 0, got {
                    type(products)
                }"
            )
        if isinstance(products, Species):
            products = Complex({products: 1})
        elif isinstance(products, int) and products == 0:
            products = Complex({})

        for s in products.count_dict.keys():
            if set(s.index_symbols) - set(reactants_index_symbols) != set():
                raise ValueError(
                    "The index symbols of the products must be a subset "
                    "of the index symbols of the reactants, got "
                    f"{set(s.index_symbols)} and {set(reactants_index_symbols)}"
                )

        object.__setattr__(self, "reactants", reactants)
        object.__setattr__(self, "products", products)

    def to_dict(self) -> dict:
        return {
            "type": "fast",
            "reactants": _complex_to_yaml(self.reactants),
            "products": _complex_to_yaml(self.products),
        }

    @classmethod
    def from_dict(cls, entry: dict, symbols: ReactionSymbolTable):
        if entry["type"] != "fast":
            raise ValueError(f"Expected type 'fast', got {entry['type']!r}")
        products_entries = entry.get("products", [])
        if products_entries:
            products = _complex_from_yaml(products_entries, symbols)
        else:
            products = 0
        return cls(
            _complex_from_yaml(entry.get("reactants", []), symbols),
            products,
        )


register_yaml_type("fast", FastReaction)
