"""YAML serialization helpers for reaction networks.

This module converts lists of reaction objects to and from YAML documents.
Reaction classes register a string ``type`` tag; loading dispatches each
YAML entry to the matching ``from_dict`` implementation.

Network files have two top-level sections. The ``symbols`` block declares
shared names (species, rate constants, index symbols). A
``ReactionSymbolTable`` maps those labels back to icrn symbol objects on
load and is built automatically from a reaction list when saving. The
``reactions`` block lists entries that reference symbols by name and are
routed by ``type`` to the appropriate reaction class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from ..symbols import (
    Complex,
    IndexSymbol,
    RateConstant,
    Species,
    TensorExpression,
    TensorFunction,
    TensorLiteral,
    many_index_symbols,
    many_rate_constants,
    many_species,
)

_YAML_REACTION_TYPES: dict[str, Any] = {}


def register_yaml_type(type_name: str, reaction_cls: Any) -> None:
    """Register a reaction class for YAML loading.

    Parameters
    ----------
    type_name : str
        Value of the ``type`` field in a saved reaction entry, e.g.
        ``"mass_action"`` or ``"fast"``.
    reaction_cls : type
        Class that implements ``from_dict(entry, symbols)`` for that
        ``type_name``.

    See Also
    --------
    dispatch_from_dict : Uses the registry when loading a network.
    """
    _YAML_REACTION_TYPES[type_name] = reaction_cls


def dispatch_from_dict(entry: dict, symbols: ReactionSymbolTable):
    """Construct one reaction from a YAML entry.

    Looks up ``entry["type"]`` in the reaction-type registry and delegates
    to that class's ``from_dict`` method.

    Parameters
    ----------
    entry : dict
        One element of the ``reactions`` list in a network document. Must
        contain a ``type`` key.
    symbols : ReactionSymbolTable
        Symbol table built from the document ``symbols`` section.

    Returns
    -------
    reaction
        Instance of the registered reaction class for ``entry["type"]``.

    Raises
    ------
    ValueError
        If ``entry["type"]`` is not registered.
    KeyError
        If ``entry`` does not contain ``"type"``.
    """
    type_name = entry["type"]
    if type_name not in _YAML_REACTION_TYPES:
        raise ValueError(
            f"Unknown reaction type {type_name!r}. "
            f"Known types: {sorted(_YAML_REACTION_TYPES)}"
        )
    return _YAML_REACTION_TYPES[type_name].from_dict(entry, symbols)


def _species_label(species: Species) -> str:
    return species[()].label


def _index_names(species: Species) -> list[str]:
    return [idx.label for idx in species.index_symbols]


def _species_entry(species: Species, count: int) -> dict:
    entry = {"species": _species_label(species), "count": count}
    if species.index_symbols:
        entry["indices"] = _index_names(species)
    return entry


def _complex_to_yaml(complex_obj: Complex) -> list[dict]:
    return [
        _species_entry(species, count)
        for species, count in complex_obj.count_dict.items()
    ]


def _rate_to_yaml(rate_expr: TensorExpression) -> dict | float | str:
    if isinstance(rate_expr, TensorLiteral):
        return float(rate_expr.numeric_value)
    if isinstance(rate_expr, RateConstant):
        entry: dict = {"rate_constant": rate_expr.label}
        if rate_expr.index_symbols:
            entry["indices"] = [idx.label for idx in rate_expr.index_symbols]
        return entry
    raise NotImplementedError(
        f"Cannot serialize rate expression of type {type(rate_expr).__name__}"
    )


def _resolve_species(entry: dict, symbols: ReactionSymbolTable) -> Species:
    label = entry["species"]
    indices = entry.get("indices", [])
    species = symbols.species[label]
    if not indices:
        return species
    index_objs = tuple(symbols.index_symbols[name] for name in indices)
    if len(index_objs) == 1:
        return species[index_objs[0]]
    return species[index_objs]


def _complex_from_yaml(
    entries: list[dict], symbols: ReactionSymbolTable
) -> Complex:
    complex_obj = Complex({})
    for entry in entries:
        species = _resolve_species(entry, symbols)
        count = entry.get("count", 1)
        complex_obj = complex_obj.add_species(species, count)
    return complex_obj


def _rate_from_yaml(
    entry: dict | float | str | int, symbols: ReactionSymbolTable
):
    if isinstance(entry, int | float):
        return TensorLiteral(entry)
    if isinstance(entry, str):
        return symbols.rate_constants[entry]
    if isinstance(entry, dict):
        if "literal" in entry:
            return TensorLiteral(entry["literal"])
        label = entry["rate_constant"]
        rate = symbols.rate_constants[label]
        indices = entry.get("indices", [])
        if not indices:
            return rate
        index_objs = tuple(symbols.index_symbols[name] for name in indices)
        if len(index_objs) == 1:
            return rate[index_objs[0]]
        return rate[index_objs]
    raise ValueError(f"Unsupported rate entry: {entry!r}")


@dataclass
class ReactionSymbolTable:
    """Named symbols shared across a reaction network YAML file.

    Maps string labels from the YAML ``symbols`` section to icrn symbol
    objects used when reconstructing reactions.

    Attributes
    ----------
    species : dict[str, Species]
        Base species keyed by label.
    rate_constants : dict[str, RateConstant]
        Base rate constants keyed by label.
    index_symbols : dict[str, IndexSymbol]
        Index symbols keyed by label.
    """

    species: dict[str, Species] = field(default_factory=dict)
    rate_constants: dict[str, RateConstant] = field(default_factory=dict)
    index_symbols: dict[str, IndexSymbol] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize the symbol table to a YAML-compatible dict.

        Returns
        -------
        dict
            Mapping with keys ``species``, ``rate_constants``, and
            ``index_symbols``.
        """
        index_entries = []
        for name, idx in sorted(self.index_symbols.items()):
            index_entries.append({"names": [name], "size": idx.index_set})

        return {
            "species": sorted(self.species),
            "rate_constants": sorted(self.rate_constants),
            "index_symbols": index_entries,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ReactionSymbolTable:
        """Build a symbol table from a YAML ``symbols`` section.

        Parameters
        ----------
        data : dict
            Parsed ``symbols`` block with optional keys ``species``,
            ``rate_constants``, and ``index_symbols``.

        Returns
        -------
        ReactionSymbolTable
            Populated symbol table ready for `dispatch_from_dict`.
        """
        table = cls()
        for name in data.get("species", []):
            table.species[name] = many_species(name)
        for name in data.get("rate_constants", []):
            table.rate_constants[name] = many_rate_constants(name)
        for entry in data.get("index_symbols", []):
            names = entry["names"]
            size = entry["size"]
            created = many_index_symbols(", ".join(names), size)
            if isinstance(created, tuple):
                for sym in created:
                    table.index_symbols[sym.label] = sym
            else:
                table.index_symbols[created.label] = created
        return table


def _iter_side_species(side) -> Iterable[tuple[Species, int]]:
    if isinstance(side, Complex):
        yield from side.count_dict.items()
    elif isinstance(side, Species):
        yield side, 1


def collect_symbol_table(rxns: Iterable) -> ReactionSymbolTable:
    """Collect all symbols referenced by a list of reactions.

    Walks reactants, products, rate expressions, and optional
    ``collect_yaml_symbols`` hooks to build the ``symbols`` section for
    saving a network.

    Parameters
    ----------
    rxns : iterable
        Reaction objects that implement ``to_dict()``.

    Returns
    -------
    ReactionSymbolTable
        Deduped species, rate constants, and index symbols used by ``rxns``.
    """
    table = ReactionSymbolTable()

    def add_species(species: Species):
        table.species[_species_label(species)] = species[()]
        for idx in species.index_symbols:
            table.index_symbols[idx.label] = idx

    def add_rate(rate_expr: TensorExpression | None):
        if rate_expr is None:
            return
        if isinstance(rate_expr, RateConstant):
            table.rate_constants[rate_expr.label] = rate_expr[()]
            for idx in rate_expr.index_symbols:
                table.index_symbols[idx.label] = idx
        elif isinstance(rate_expr, TensorFunction):
            for arg in rate_expr.args:
                add_rate(arg)
        elif isinstance(rate_expr, Species):
            add_species(rate_expr)

    for rxn in rxns:
        for species, _count in _iter_side_species(rxn.reactants):
            add_species(species)
        for species, _count in _iter_side_species(rxn.products):
            add_species(species)
        add_rate(getattr(rxn, "aux", None))
        if hasattr(rxn, "collect_yaml_symbols"):
            rxn.collect_yaml_symbols(table)

    return table


def network_to_document(rxns: Iterable) -> dict:
    """Convert a reaction list to a YAML-serializable network document.

    Parameters
    ----------
    rxns : iterable
        Reaction objects that implement ``to_dict()``.

    Returns
    -------
    dict
        Document with ``symbols`` and ``reactions`` keys.
    """
    rxn_list = list(rxns)
    symbols = collect_symbol_table(rxn_list)
    return {
        "symbols": symbols.to_dict(),
        "reactions": [rxn.to_dict() for rxn in rxn_list],
    }


def network_from_document(document: dict):
    """Load reactions from a parsed network document.

    Parameters
    ----------
    document : dict
        Parsed YAML with ``symbols`` and ``reactions`` keys.

    Returns
    -------
    rxns : list
        Reconstructed reaction objects.
    symbols : ReactionSymbolTable
        Symbol table used during deserialization.
    """
    symbols = ReactionSymbolTable.from_dict(document["symbols"])
    rxns = [
        dispatch_from_dict(entry, symbols) for entry in document["reactions"]
    ]
    return rxns, symbols


def save_network_yaml(path: str | Path, rxns: Iterable) -> None:
    """Write a reaction network to a YAML file.

    Parameters
    ----------
    path : str or Path
        Output file path.
    rxns : iterable
        Reaction objects that implement ``to_dict()``.

    See Also
    --------
    load_network_yaml : Inverse operation.
    network_to_document : Build the document without writing a file.
    """
    document = network_to_document(rxns)
    Path(path).write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )


def load_network_yaml(path: str | Path):
    """Load a reaction network from a YAML file.

    Parameters
    ----------
    path : str or Path
        Input file path.

    Returns
    -------
    rxns : list
        Reconstructed reaction objects.
    symbols : ReactionSymbolTable
        Symbol table from the file.

    See Also
    --------
    save_network_yaml : Inverse operation.
    network_from_document : Parse an in-memory document.
    """
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return network_from_document(document)
