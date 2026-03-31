from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Sequence
import jax.numpy as jnp
from jax.typing import ArrayLike

### symbols need to be hashable to use in dict, they also need to be ordered to be compatible with jax compilation


@dataclass(frozen=True)
class OrderedHashableSymbol(ABC):
    label: str
    aux: Any

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return (self.label, self.aux) == (other.label, other.aux)
        else:
            raise TypeError

    def __lt__(self, other):
        if isinstance(other, type(self)):
            return (self.label, self.aux) < (other.label, other.aux)
        else:
            raise TypeError

    def __gt__(self, other):
        return not self < other

    def __le__(self, other):
        return self < other or self == other

    def __ge__(self, other):
        return self > other or self == other

Numeric = ArrayLike

def _is_tensor_literal(x):
    return isinstance(x, Numeric)


def _index_symbols_for_arg(arg):
    if isinstance(arg, TensorExpression):
        return arg.index_symbols
    if _is_tensor_literal(arg):
        return None
    raise TypeError(f"Unsupported argument type {type(arg)}")

def _index_symbols_empty(s):
    return s is None or s == ()


def _unify_index_symbols(args):
    """Unify index symbols; None and () both mean 'no indices' and are compatible."""
    sym = None
    for arg in args:
        if isinstance(arg, TensorExpression):
            s = arg.index_symbols
            if _index_symbols_empty(s):
                continue
            if sym is None:
                sym = s
            elif s != sym:
                raise ValueError(
                    f"All tensor arguments must share index symbols, got {s} and {sym}"
                )
    return sym if sym is not None else ()


def _binary_op_helper(fn, a, b):
    if isinstance(a, Species) or isinstance(b, Species):
        raise NotImplementedError
    elif isinstance(a, TensorExpression | Numeric) or isinstance(b, TensorExpression | Numeric):
        if isinstance(a, Numeric):
            a = TensorLiteral(a)
        if isinstance(b, Numeric):
            b = TensorLiteral(b)

        if a.index_symbols != b.index_symbols:
            raise ValueError(f"Index symbols must be the same, got {a.index_symbols} and {b.index_symbols}")

        return TensorFunction(fn, (a, b))
    else:
        raise NotImplementedError

def _check_shape(res: ArrayLike, index_symbols):
    if not isinstance(res, ArrayLike):
        raise ValueError(f"Result must be an array-like object, got {type(res)}")
    if not isinstance(index_symbols, tuple):
        raise ValueError(f"Index symbols must be a tuple, got {type(index_symbols)}")
    if not all(isinstance(i, IndexSymbol) for i in index_symbols):
        raise ValueError(f"Index symbols must be a tuple of IndexSymbols, got {type(index_symbols)}")

    res_shape = res.shape

    if len(res_shape) != len(index_symbols):
        return False
    
    for i in range(len(res_shape)):
        if index_symbols[i].index_set > 0 and res_shape[i] != index_symbols[i].index_set:
            return False

    return True

class TensorExpression(ABC):
    def eval_with_check(self, data):
        res = self.eval(data)
        if _check_shape(res, self.index_symbols):
            return res
        else:
            raise ValueError(f"Shape of {res} does not match index symbols {self.index_symbols}")

    @abstractmethod
    def eval(self, data):
        pass

    @property
    @abstractmethod
    def index_symbols(self):
        pass

    def __add__(self, other):
        return _binary_op_helper(jnp.add, self, other)

    def __radd__(self, other):
        return _binary_op_helper(jnp.add, other, self)

    def __mul__(self, other):
        return _binary_op_helper(jnp.multiply, self, other)

    def __rmul__(self, other):
        return _binary_op_helper(jnp.multiply, other, self)

    def __sub__(self, other):
        return _binary_op_helper(jnp.subtract, self, other)

    def __rsub__(self, other):
        return _binary_op_helper(jnp.subtract, other, self)

    def __truediv__(self, other):
        return _binary_op_helper(jnp.true_divide, self, other)

    def __rtruediv__(self, other):
        return _binary_op_helper(jnp.true_divide, other, self)

    def __neg__(self):
        return TensorFunction(jnp.negative, (self,))

# class IndexExpression(ABC):
#     def __init__(self, op, args: tuple[IndexExpression]):
#         self._op = op
#         self._args = args

#     def __add__(self, other):
#         return IndexExpression(operator.add, (self, other))

#     def __radd__(self, other):
#         return IndexExpression(operator.add, (self, other))

#     def __mul__(self, other):
#         return IndexExpression(operator.mul, (self, other))

#     def __rmul__(self, other):
#         return IndexExpression(operator.mul, (self, other))

#     def __sub__(self, other):
#         return IndexExpression(operator.sub, (self, other))

#     def __rsub__(self, other):
#         return IndexExpression(operator.sub, (other, self))

#     def __neg__(self):
#         return IndexExpression(operator.neg, (self,))

#     def __repr__(self):
#         return super().__repr__()

#     def __str__(self):
#         return super().__str__()

#     def map_to_jnp_index(self, idx):
#         pass


class IndexSymbol(OrderedHashableSymbol):
    def __init__(self, label: str, index_set: int=0):
        if not isinstance(label, str):
            raise ValueError(f"Label must be a string, got {type(label)}")
        if index_set:
            if not isinstance(index_set, int):
                raise ValueError(f"Index set must be an integer, got {type(index_set)}")
            if index_set < 0:
                raise ValueError(f"Index set must be greater than or equal to 0, got {index_set}")
    
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "aux", index_set)

    @property
    def index_set(self):
        return self.aux

    @index_set.setter
    def index_set(self, index_set):
        self.aux = index_set

    # @index_set.setter
    # def index_set(self, index_set):
    #     self.aux = index_set

    # def __add__(self, other):
    #     if isinstance(other, type(self)):
    #         new_complex = Complex({})

    #         new_complex.add_species(self)
    #         new_complex.add_species(other)
    #         return new_complex
    #     elif isinstance(other, int):
    #         other.add_species(self)
    #         return other
    #     else:
    #         raise NotImplementedError

    # def __radd__(self, other):
    #     return self.__add__(other)

    # def __mul__(self, other):
    #     if isinstance(other, int):
    #         pass
    #     else:
    #         raise NotImplementedError

    # def __rmul__(self, other):
    #     return self.__mul__(other)

    def __str__(self):
        return self.label

    def __repr__(self):
        if self.index_set > 0:
            return self.label + ":0..." + str(self.index_set - 1)
        else:
            return self.label + ":0"

class TensorSymbol(OrderedHashableSymbol, TensorExpression):
    def __init__(self, label: str, indexing: tuple[IndexSymbol, ...] = ()):
        if not isinstance(label, str):
            raise ValueError(f"Label must be a string, got {type(label)}")
        object.__setattr__(self, "label", label)

        if not isinstance(indexing, tuple):
            raise ValueError(f"Indexing must be a tuple, got {type(indexing)}")
        # if isinstance(indexing, IndexSymbol):
        #     indexing = (indexing,)
        if not all(isinstance(i, IndexSymbol) for i in indexing):
            raise ValueError(f"Indexing must be a tuple of IndexSymbols, got {type(indexing)}")

        object.__setattr__(self, "aux", indexing)

    @property
    def index_symbols(self):
        return self.aux

    def eval(self, data):
        return data[self[()]]

    def __str__(self):
        if not self.aux:
            return self.label
        else:
            return self.label + "[" + ",".join(map(str, self.aux)) + "]"

    def __repr__(self):
        if not self.aux:
            return self.label
        else:
            return self.label + "[" + repr(self.aux) + "]"

    def __getitem__(self, index_symbols):
        if not isinstance(index_symbols, tuple):
            index_symbols = (index_symbols,)
        if not isinstance(index_symbols, tuple) or not all(isinstance(i, IndexSymbol) for i in index_symbols):
            raise ValueError(f"Index symbols must be a tuple of IndexSymbols, got {type(index_symbols)}")

        return self.__class__(self.label, index_symbols)

class Species(TensorSymbol):
    def __add__(self, other):
        if isinstance(other, Species):
            new_complex = Complex({self: 1})
            return new_complex.add_species(other)
        elif isinstance(other, Complex):
            return other.add_species(self)
        else:
            raise NotImplementedError

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, int):
            return Complex({self: other})
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        return self.__mul__(other)

@dataclass(frozen=True)
class Complex:
    count_dict: dict

    def __post_init__(self):
        if not isinstance(self.count_dict, dict):
            raise ValueError(f"count_dict must be a dict, got {type(self.count_dict)}")
        for s, c in self.count_dict.items():
            if not isinstance(s, Species):
                raise ValueError(f"count_dict keys must be Species, got {type(s)}")
            if type(c) is not int:
                raise ValueError(f"count_dict values must be int, got {type(c)}")
            if c <= 0:
                raise ValueError(f"count_dict values must be positive, got {c}")

    def add_species(self, s, count=1):
        if not isinstance(s, Species):
            raise ValueError(f"Species must be a Species, got {type(s)}")
        if not isinstance(count, int):
            raise ValueError(f"Count must be an integer, got {type(count)}")
        if count <= 0:
            raise ValueError(f"Count must be greater than 0, got {count}")
            
        new_count_dict = self.count_dict.copy()
        new_count_dict[s] = new_count_dict.get(s, 0) + count
        return Complex(new_count_dict)

    def __add__(self, other: Complex | Species):
        if isinstance(other, Complex):
            new_complex = self
            for species, count in other.count_dict.items():
                new_complex = new_complex.add_species(species, count)
            return new_complex
        elif isinstance(other, Species):
            return self.add_species(other)
        else:
            raise NotImplementedError

    def __radd__(self, other):
        return self.__add__(other)

    def __eq__(self, other):
        if isinstance(other, Complex):
            return self.count_dict == other.count_dict
        elif isinstance(other, Species):
            return self.count_dict == {other: 1}
        else:
            raise NotImplementedError

    def __str__(self):
        res_str = [str(c) +"*"+ str(s) if c > 1 else str(s) for s, c in self.count_dict.items()]
        return " + ".join(res_str)

    def __repr__(self):
        return repr(self.count_dict)

class RateConstant(TensorSymbol):
    pass

@dataclass(frozen=True)
class TensorLiteral(TensorExpression):
    numeric_value: Numeric

    def __init__(self, numeric_value):
        if not isinstance(numeric_value, Numeric):
            raise ValueError(f"Numeric value must be a Numeric, got {type(numeric_value)}")

        val_as_array = jnp.array(numeric_value).astype(float)
        if val_as_array.ndim > 0:
            raise ValueError(f"Numeric value must be a scalar, got {val_as_array.shape}")

        object.__setattr__(self, "numeric_value", val_as_array)

    def eval(self, data):
        return self.numeric_value

    @property
    def index_symbols(self):
        return ()

    def __str__(self):
        return str(self.numeric_value)

    def __repr__(self):
        return repr(self.numeric_value)

@dataclass(frozen=True)
class TensorFunction(TensorExpression):
    fn: Callable
    args: tuple

    def __init__(self, fn, args):
        if not isinstance(fn, Callable):
            raise ValueError(f"Function must be callable, got {type(fn)}")

        if not isinstance(args, tuple):
            args = (args,)

        if isinstance(fn, jnp.ufunc):
            if fn.nin != len(args):
                raise ValueError(f"Function must have {fn.nin} arguments, got {len(args)}")
        for arg in args:
            _index_symbols_for_arg(arg)
        _unify_index_symbols(args)

        object.__setattr__(self, "fn", fn)
        object.__setattr__(self, "args", args)

    def eval(self, data):
        return self.fn(*(x.eval(data) for x in self.args))

    def __str__(self):
        return self.fn.__name__ + "(" + ",".join(map(str, self.args)) + ")"

    def __repr__(self):
        return self.fn.__name__ + "(" + ",".join(map(repr, self.args)) + ")"

    @property
    def index_symbols(self):
        return _unify_index_symbols(self.args)
    

def _parse_names(names: str) -> list[str]:
    names_list = names.split(",")
    return list(map(lambda x: x.strip(), names_list))

def many_index_symbols(names: str, index_set: int=0) -> IndexSymbol | list[IndexSymbol]:
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
    try:
        name_list = _parse_names(names)
    except AttributeError:
        raise ValueError(f"Invalid index symbol names: {names}")

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
    try:
        name_list = _parse_names(names)
    except AttributeError:
        raise ValueError(f"Invalid species names: {names}")

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
    try:
        name_list = _parse_names(names)
    except AttributeError:
        raise ValueError(f"Invalid species names: {names}")

    if len(name_list) == 1:
        return RateConstant(name_list[0])
    else:
        return list(map(lambda name: RateConstant(name), name_list))
