"""
This module contains the building blocks of ICRNs.
"""

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field, replace
import jax.numpy as jnp
from typing import Self
from .tensor_symbol import AbstractChemicalSymbol, Function
from ._abstract_symbol import OrderedSymbol


class RepresentationError(Exception):
    """
    The exception class for invalid ICRNs.
    """

    pass


@dataclass(frozen=True)
class IndexSymbol(OrderedSymbol):
    """
    Parameters
    ----------
    label : str
        The name of the index symbol.
    index_set: int
        The index set is defined to be [0, `index_set`).
    """

    label: str
    index_set: int
    """
    Class for representing index symbols.
    """

    # def __init__(self, label, index_set) -> None:
    #     """
    #     Parameters
    #     ----------
    #     label : str
    #         The name of the index symbol.
    #     index_set: int
    #         The index set is defined to be [0, `index_set`).
    #     """
    #     self._label = label
    #     self._index_set = index_set

    # @property
    # def label(self):
    #     return self._label

    # @property
    # def index_set(self):
    #     return self._index_set

    def __eq__(self, other):
        return self.label == other.label and self.index_set == other.index_set

    def __lt__(self, other):
        return self.label < other.label

    def __gt__(self, other):
        return not self < other

    def __le__(self, other):
        return self < other or self.label == other.label

    def __ge__(self, other):
        return not self <= other

    def __str__(self):
        return self.label

    def __repr__(self):
        return self.label + ":0..." + str(self.index_set - 1)

    # def __hash__(self):
    #     return hash((self._label, self.index_set))


def check_tuple_all_index_symbols(tup):
    if isinstance(tup, tuple):
        for e in tup:
            if not isinstance(e, IndexSymbol):
                return False
        return True
    elif isinstance(tup, IndexSymbol):
        return True
    else:
        return False


def check_tuple_no_index_symbol(tup):
    if isinstance(tup, IndexSymbol):
        return False
    elif isinstance(tup, tuple):
        for e in tup:
            if isinstance(e, IndexSymbol):
                return False
        return True
    else:
        return True


type SingleIndexing = IndexSymbol | int
type Indexing = None | SingleIndexing | tuple[SingleIndexing]


class IndexingType(Enum):
    BASE = 0
    INDEXED = 1
    CONCRETE = 2


def _find_indexing_type(indexing):
    if indexing is None:
        return IndexingType.BASE
    else:
        if isinstance(indexing, tuple) and len(indexing) > 0:
            if isinstance(indexing[0], IndexSymbol):
                for element in indexing:
                    if not isinstance(element, IndexSymbol):
                        raise TypeError

                return IndexingType.INDEXED
            elif isinstance(indexing[0], int):
                for element in indexing:
                    if not isinstance(element, int):
                        raise TypeError

                return IndexingType.CONCRETE
            else:
                raise TypeError
        else:
            raise TypeError


type DataDict = dict[AbstractChemicalSymbol, jnp.ndarray]


@dataclass(frozen=True)
class AbstractChemicalSymbol(Expr):
    label: str
    indexing: None | tuple = None
    indexing_type: IndexingType = field(init=False)

    def __init__(self, label, indexing=None):
        object.__setattr__(self, "label", label)

        if not isinstance(indexing, tuple) and indexing is not None:
            indexing = tuple([indexing])

        object.__setattr__(self, "indexing", indexing)

        object.__setattr__(
            self, "indexing_type", _find_indexing_type(self.indexing)
        )

    def __lt__(self, other):
        return self.label < other.label

    def __gt__(self, other):
        return not self < other

    def __le__(self, other):
        return self < other or self.label == other.label

    def __ge__(self, other):
        return not self <= other

    def __getitem__(self, index_symbols):
        return self.__class__(self.label, index_symbols)

    def eval(self, arr_dict):
        if self.indexing_type == IndexingType.CONCRETE:
            return arr_dict[self][self.indexing]
        else:
            return arr_dict[self]

    def __eq__(self, other):
        return (
            self.label == other.label
            and self.indexing_type == other.indexing_type
        )

    def __hash__(self):
        return hash((self.label, self.indexing, self.indexing_type))

    def __repr__(self):
        if self.indexing_type == IndexingType.BASE:
            return self.label
        else:
            return self.label + "[" + repr(self.indexing) + "]"

    def __str__(self):
        if self.indexing_type == IndexingType.BASE:
            return self.label
        else:
            return self.label + "[" + ",".join(map(str, self.indexing)) + "]"


class Species(AbstractChemicalSymbol):
    def __add__(self, other):
        if isinstance(other, Species):
            new_complex = Complex({})

            new_complex.add_species(self)
            new_complex.add_species(other)
            return new_complex
        elif isinstance(other, Complex):
            other.add_species(self)
            return other
        else:
            raise NotImplementedError

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, int):
            new_complex = Complex({})
            new_complex.add_species(self, other)
            return new_complex
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        return self.__mul__(other)


class IndexedSpecies:
    pass


class ConcreteSpecies:
    pass


@dataclass(frozen=True)
class Complex:
    count_dict: dict
    # def __init__(self, count_dict={}):
    #     self._count_dict = count_dict

    # @property
    # def count_dict(self):
    #     return self._count_dict

    # def __eq__(self, other):
    #     if isinstance(other, Complex):
    #         return self.count_dict == other.count_dict
    #     else:
    #         raise NotImplementedError

    def add_species(self, s, count=1):
        new_count_dict = self.count_dict
        new_count_dict[s] = new_count_dict.get(s, 0) + count
        return replace(self, count_dict=new_count_dict)

    def __add__(self, other: Self | Species):
        if isinstance(other, Complex):
            for species, count in other.count_dict.items():
                self.add_species(species, count)
            return self
        elif isinstance(other, Species):
            return other.__add__(self)
        else:
            raise NotImplementedError

    def __str__(self):
        res_str = [str(c) + str(s) for s, c in self.count_dict.items()]
        return " + ".join(res_str)

    def __repr__(self):
        return repr(self.count_dict)


# @dataclass(frozen=True)
# class RateConstantExpr(TensorBacked):
#     pass
#     # expr : RateConstantFunction | RateConstant
type RateExpr = RateConstant | RateConstantFunction | int | float


@dataclass(frozen=True)
class RateConstantFunction(Function):
    fn: jnp.ufunc
    args: list[RateExpr]

    def __init__(self, fn, args) -> None:
        object.__setattr__(self, "fn", fn)

        if not isinstance(args, tuple):
            args = tuple([args])

        object.__setattr__(self, "args", args)

    def eval(self, arr_dict: DataDict) -> jnp.ndarray:
        def eval_arg(x):
            if isinstance(x, int | float):
                return x
            else:
                x.eval(arr_dict)

        data_input = map(eval_arg, self.args)
        return self.fn(*data_input)

    def __str__(self) -> str:
        fn_name = self.fn.__class__.__name__
        return fn_name + "(" + ",".join(map(str, self.args)) + ")"

    def __repr__(self) -> str:
        fn_name = self.fn.__class__.__name__
        return fn_name + "(" + ",".join(map(repr, self.args)) + ")"


class RateConstant(AbstractChemicalSymbol):
    def __add__(self, other: RateExpr) -> RateConstantFunction:
        if isinstance(other, RateConstant | RateConstantFunction | int | float):
            return RateConstantFunction(jnp.add, (self, other))
        else:
            raise TypeError

    def __radd__(self, other: RateExpr) -> RateConstantFunction:
        if isinstance(other, RateConstant | RateConstantFunction | int | float):
            return RateConstantFunction(jnp.add, (other, self))
        else:
            raise TypeError

    def __mul__(self, other: RateExpr) -> RateConstantFunction:
        if isinstance(other, RateConstant | RateConstantFunction | int | float):
            return RateConstantFunction(jnp.multiply, (self, other))
        else:
            raise TypeError

    def __rmul__(self, other: RateExpr) -> RateConstantFunction:
        if isinstance(other, RateConstant | RateConstantFunction | int | float):
            return RateConstantFunction(jnp.multiply, (other, self))
        else:
            raise TypeError

    def __sub__(self, other: RateExpr) -> RateConstantFunction:
        if isinstance(other, RateConstant | RateConstantFunction | int | float):
            return RateConstantFunction(jnp.subtract, (self, other))
        else:
            raise TypeError

    def __rsub__(self, other: RateExpr) -> RateConstantFunction:
        if isinstance(other, RateConstant | RateConstantFunction | int | float):
            return RateConstantFunction(jnp.subtract, (other, self))
        else:
            raise TypeError

    def __div__(self, other: RateExpr) -> RateConstantFunction:
        if isinstance(other, RateConstant | RateConstantFunction | int | float):
            return RateConstantFunction(jnp.true_divide, (self, other))
        else:
            raise TypeError

    def __rdiv__(self, other: RateExpr) -> RateConstantFunction:
        if isinstance(other, RateConstant | RateConstantFunction | int | float):
            return RateConstantFunction(jnp.true_divide, (other, self))
        else:
            raise TypeError

    def __neg__(self) -> RateConstantFunction:
        return RateConstantFunction(jnp.negative, self)


def _parse_names(names: str) -> list[str]:
    names_list = names.split(",")
    return list(map(lambda x: x.strip(), names_list))


def many_index_symbols(
    names: str, index_set: int
) -> IndexSymbol | list[IndexSymbol]:
    """
    Instanitate multiple index symbols with the same index sets at once.

    Parameters
    ----------
    names : str
        A single string with index symbol comma-separated names.
    index_set : int
        The upper limit of index sets for each index symbol.

    Returns
    -------
    tuple of IndexSymbols
        Order of IndexSymbols is based on their order in `names`.
    """
    name_list = _parse_names(names)
    if len(name_list) == 1:
        return IndexSymbol(name_list[0], index_set)
    else:
        return list(map(lambda name: IndexSymbol(name, index_set), name_list))


def many_species(names: str) -> Species | list[Species]:
    """
    Instanitate multiple Species at once.

    Parameters
    ----------
    names : str
        Single string with Species names comma-separated or space-separated.

    Returns
    -------
    tuple of Species
        Order of Species is based on their order in `names`.
    """
    name_list = _parse_names(names)

    if len(name_list) == 1:
        return Species(name_list[0])
    else:
        return list(map(lambda name: Species(name), name_list))


def many_rate_constants(names: str) -> RateConstant | list[RateConstant]:
    """
    Instanitate multiple RateConstants at once.

    Parameters
    ----------
    names : str
        Single string with RateConstant names comma-separated or
        space-separated.

    Returns
    -------
    tuple of RateConstants
        Order of RateConstants is based on their order in `names`.
    """
    name_list = _parse_names(names)
    if len(name_list) == 1:
        return RateConstant(name_list[0])
    else:
        return list(map(lambda name: RateConstant(name), name_list))
