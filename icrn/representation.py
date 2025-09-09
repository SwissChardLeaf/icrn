"""
This module contains the building blocks of ICRNs.
"""
from abc import ABC, abstractmethod
from itertools import product
import jax.numpy as jnp
import jax

class RepresentationError(Exception):
    """
    The exception class for invalid ICRNs.
    """
    pass

class IndexSymbol():
    """
    Class for representing index symbols.
    """
    def __init__(self, label, index_set) -> None:
        '''
        Parameters
        ----------
        label : str
            The name of the index symbol.
        index_set: int
            The index set is defined to be [0, `index_set`).
        '''
        self._label = label
        self._index_set = index_set

    @property
    def label(self):
        return self._label
    
    @property
    def index_set(self):
        return self._index_set
    
    def __eq__(self, other):
        return self._label == other.label and self.index_set == other.index_set
    
    def __lt__(self, other):
        return self.label < other.label

    def __gt__(self, other):
        return not self < other

    def __le__(self, other):
        return self < other or self.label == other.label

    def __ge__(self, other):
        return not self <= other
    
    def __repr__(self):
        return self.label
    
    def __str__(self):
        return self.label
    
    def __hash__(self):
        return hash((self._label, self.index_set))
    
class IndexedObject(ABC):
    """
    Class for representing components of an indexed expression. All classes
    that have indexing should subclass this class.
    """
    
    @abstractmethod
    def get_index_symbols_set(self, spatial_dim=None):
        """
        Method for retrieving the IndexSymbols.

        Returns
        -------
        list of IndexSymbols
            A list of the Species in a complex, sorted.
        """
        pass
    
    @abstractmethod
    def index_symbols_replace(self, values):
        """
        Method for replacing the IndexSymbols with a combination of values
        from their index sets.
        
        Parameters
        ----------
        values: tuple of ints
            An IndexSymbol will be replaced by the component of `values` that 
            corresponds to the component of the IndexSymbol in a sorted list
            of indreturned by index_symbols().

        Returns
        -------
        IndexedObject
            An IndexedObject with index symbols replaced by values.
        """
        pass

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

class BaseObject(ABC):
    def __init__(self, label):
        self._label = label

    @property
    def label(self):
        return self._label

    def __lt__(self, other):
        return self.label < other.label

    def __gt__(self, other):
        return not self < other

    def __le__(self, other):
        return self < other or self.label == other.label

    def __ge__(self, other):
        return not self <= other
    
    @abstractmethod
    def __getitem__(self, index_symbols):
        pass

    def __eq__(self, other):
        return self.label == other.label
    
    def __hash__(self):
        return hash(self.label)
    
    def __repr__(self):
        return self.label
    
    def __str__(self):
        return self.label
    
class IndexedBase(IndexedObject):
    def __init__(self, base, index_symbols):
        self._base = base
        self._index_symbols = index_symbols

    @property
    def base(self):
        return self._base
    
    @property
    def index_symbols(self):
        return self._index_symbols
    
    def get_index_symbols_set(self, spatial_dim=None):
        idx_set = set(self.index_symbols)
        if spatial_dim is not None:
            idx_set.add(IndexSymbol("H", spatial_dim[0]))
            idx_set.add(IndexSymbol("W", spatial_dim[1]))
        return idx_set
    
    def __eq__(self, other):
        return self.base == other.base and self.index_symbols == other.index_symbols
    
    def __hash__(self):
        return hash((self.base, self.index_symbols))
    
    def __str__(self):
        return repr(self)
    
    def __repr__(self):
        return str(self.base) + "[" + ",".join(map(str, self.index_symbols)) + "]"
    
class ConcreteObject(ABC):
    def __init__(self, base, index_symbols):
        self._base = base
        self._index_symbols = index_symbols
            
    @property
    def base(self):
        return self._base
    
    @property
    def index_symbols(self):
        return self._index_symbols
    
    def get_index_symbols_set(self):
        return set(self.index_symbols)
    
    def __eq__(self, other):
        return self.base == other.base and self.index_symbols == other.index_symbols
    
    def __hash__(self):
        return hash((self.base, self.index_symbols))
    
    def __str__(self):
        return repr(self)
    
    def __repr__(self):
        return str(self.base) + "[" + ",".join(map(str, self.index_symbols)) + "]"

class Reactant(ABC):
    def __add__(self, other):
        if isinstance(other, Reactant):
            new_complex = Complex({})

            new_complex.add_reactant(self)
            new_complex.add_reactant(other)
            return new_complex
        elif isinstance(other, Complex):
            other.add_reactant(self)
            return other
        else:
            raise NotImplementedError

    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        if isinstance(other, int):
            new_complex = Complex({})
            new_complex.add_reactant(self, other)
            return new_complex
        else:
            raise NotImplementedError
        
    def __rmul__(self, other):
        return self.__mul__(other)
    
    @abstractmethod
    def eval(self, tensor_data):
        pass

    @abstractmethod
    def einsum_index_symbol_string(self, spatial_dim, spatial_rate_constant):
        pass
    
class RateConstantExpr(ABC):
    @abstractmethod
    def eval(self, tensor_data):
        pass

    @abstractmethod
    def einsum_index_symbol_string(self):
        pass

    def __add__(self, other):
        return RateConstantFunction(jnp.add, (self, other))

    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        return RateConstantFunction(jnp.subtract, (self, other))
    
    def __rsub__(self, other):
        return RateConstantFunction(jnp.subtract, (other, self))

    def __mul__(self, other):
        return RateConstantFunction(jnp.multiply, (other, self))
        
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        return RateConstantFunction(jnp.true_divide, (self, other))
    
    def __rtruediv__(self, other):
        return RateConstantFunction(jnp.true_divide, (other, self))
    
    def __neg__(self):
        return RateConstantFunction(jnp.negative, self)

    
class Species(BaseObject, Reactant):
    """
    Class for representing base species in the ICRN.
    """
    def __getitem__(self, x):
        # print(x, type(x))

        tup_x = x
        if not isinstance(x, tuple):
            tup_x = (x,)

        if check_tuple_all_index_symbols(tup_x):
            return IndexedSpecies(self, tup_x)
        elif check_tuple_no_index_symbol(tup_x):
            return ConcreteSpecies(self, tup_x)
        else:
            raise RepresentationError
        
    def eval(self, tensor_data):
        return tensor_data[self]
    
    def einsum_index_symbol_string(self, spatial_dim, _):
        if spatial_dim:
            return "HW"
        else:
            return ""

class IndexedSpecies(IndexedBase, Reactant):
    def index_symbols_replace(self, values_dict):
        new_index_symbols = tuple(map(lambda x: values_dict[x], self.index_symbols))
        return ConcreteSpecies(self.base, new_index_symbols)
    
    def eval(self, tensor_data):
        return tensor_data[self.base]
    
    def einsum_index_symbol_string(self, spatial_dim, _):
        if spatial_dim:
            return "HW" + "".join(map(lambda x : str(x), self.index_symbols))
        else:
            return "".join(map(str, self.index_symbols))

class ConcreteSpecies(ConcreteObject, Reactant):
    def eval(self, tensor_data):
        return (tensor_data[self.base])[self.index_symbols]
    
    def einsum_index_symbol_string(self, *_):
        return ""

class RateConstant(BaseObject, RateConstantExpr):
    def __getitem__(self, x):
        tup_x = x
        if not isinstance(x, tuple):
            tup_x = (x,)

        if check_tuple_all_index_symbols(tup_x):
            return IndexedRateConstant(self, tup_x)
        elif check_tuple_no_index_symbol(tup_x):
            return ConcreteRateConstant(self, tup_x)
        else:
            raise RepresentationError
        
    def eval(self, tensor_data):
        return tensor_data[self]
    
    def einsum_index_symbol_string(self, _, spatial_rate_constant):
        if spatial_rate_constant:
            return "HW"
        else:    
            return ""

class IndexedRateConstant(IndexedBase, RateConstantExpr):
    def index_symbols_replace(self, values_dict):
        new_index_symbols = tuple(map(lambda x: values_dict[x], self.index_symbols))
        return ConcreteRateConstant(self.base, new_index_symbols)
    
    def eval(self, tensor_data):
        return tensor_data[self.base]
    
    def einsum_index_symbol_string(self, _, spatial_rate_constant):
        if spatial_rate_constant:
            return "HW" + "".join(map(lambda x : str(x), self.index_symbols))
        else:
            return "".join(map(str, self.index_symbols))

class ConcreteRateConstant(ConcreteObject, RateConstantExpr):
    def eval(self, tensor_data):
        return (tensor_data[self.base])[self.index_symbols]
    
    def einsum_index_symbol_string(self, *_):
        return ""

def _parse_names(names):
    names_list = names.split(",")
    return list(map(lambda x: x.strip(), names_list))

def many_index_symbols(names, index_set):
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
        return tuple(map(lambda name: IndexSymbol(name, index_set), name_list))

def many_species(names):
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
        return tuple(map(lambda name: Species(name), name_list))

def many_rate_constants(names):
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
        return tuple(map(lambda name: RateConstant(name), name_list))

class Complex(IndexedObject):
    def __init__(self, count_dict={}):
        self._count_dict = count_dict

    @property
    def count_dict(self):
        return self._count_dict
    
    def __eq__(self, other):
        if isinstance(other, Complex):
            return self.count_dict == other.count_dict
        else:
            raise NotImplementedError

    def add_reactant(self, s, count=1):
        self._count_dict[s] = self._count_dict.get(s, 0) + count

    def __add__(self, other):
        if isinstance(other, Complex):
            for species, count in other._count_dict.items():
                self.add_reactant(species, count)
            return self
        elif isinstance(other, Reactant):
            return other.__add__(self)
        else:
            raise NotImplementedError
        
    def get_index_symbols_set(self, spatial_dim=None):
        syms = self.count_dict.keys()

        index_symbols_set = set()

        for sym in syms:
            if isinstance(sym, IndexedSpecies):
                for i in sym.get_index_symbols_set(spatial_dim):
                    index_symbols_set.add(i)
            else:
                if spatial_dim is not None:
                    index_symbols_set.add(IndexSymbol("H", spatial_dim[0]))
                    index_symbols_set.add(IndexSymbol("W", spatial_dim[1]))

        return index_symbols_set

    def index_symbols_replace(self, values_dict):
        new_complex = Complex({})
        for sym, count in self.count_dict.items():
            if isinstance(sym, IndexedSpecies):
                new_complex.add_reactant(sym.index_symbols_replace(values_dict), count)
            else:
                new_complex.add_reactant(sym, count)
        return new_complex
    
    def __str__(self):
        return repr(self)

    def __repr__(self):
        return repr(self.count_dict)
        

class NumericRateConstant(RateConstantExpr):
    def __init__(self, num):
        self._num = num

    @property
    def num(self):
        return self._num
    
    def eval(self, _):
        return self.num

    def einsum_index_symbol_string(self, *_):
        return ""
    
    def __str__(self):
        return repr(self)
    
    def __repr__(self):
        return repr(self.num)
    
    def __eq__(self, other):
        if isinstance(other, NumericRateConstant):
            return self.num == other.num
        else:
            return False


class RateConstantFunction(IndexedObject, RateConstantExpr):
    def __init__(self, fn, args):
        # args is a RateConstant, IndexedRateConstant, or ConcreteRateConstant, or another RateConstantFunction
        self._fn = fn
        self._args = args
        
        # currently assume that all the args have the same indexings
        self._index_symbols = ()
        if isinstance(args, tuple):
            if isinstance(args[0], IndexedRateConstant):
                self._index_symbols = args[0].index_symbols
            
        elif isinstance(args, IndexedRateConstant):
            self._index_symbols = args.index_symbols

    @property
    def fn(self):
        return self._fn
    
    @property
    def args(self):
        return self._args

    @property
    def index_symbols(self):
        return self._index_symbols

    def get_index_symbols_set(self, spatial_dim=None):
        idx_set = set(self.index_symbols)
        if isinstance(self.args, tuple):
            if isinstance(self.args[0], IndexedRateConstant):
                return self.args[0].get_index_symbols_set(spatial_dim)
            
        elif isinstance(self.args, IndexedObject):
            return self.args.get_index_symbols_set(spatial_dim)

        if spatial_dim is not None:
            idx_set.add(IndexSymbol("H", spatial_dim[0]))
            idx_set.add(IndexSymbol("W", spatial_dim[1]))
        return idx_set
    
    def index_symbols_replace(self, values_dict):
        if isinstance(self.args, tuple):
            def replace_helper(x):
                if isinstance(x, IndexedObject):
                    x.index_symbols_replace(values_dict)
                else:
                    return x
            return RateConstantFunction(self.fn, tuple(map(replace_helper, self.args)))
        elif isinstance(self.args, IndexedRateConstant):
            return RateConstantFunction(self.fn, self.args.index_symbols_replace(values_dict))
        elif isinstance(self.args, RateConstantFunction):
            return RateConstantFunction(self.fn, self.args.index_symbols_replace(values_dict))
        else:
            return RateConstantFunction(self.fn, self.args)
    
    def eval(self, tensor_data):
        if isinstance(self.args, tuple):
            data_input = map(lambda x : x.eval(tensor_data), self.args)
            return self.fn(*data_input)
        else:
            return self.fn(self.args.eval(tensor_data))

    def einsum_index_symbol_string(self, spatial_dim, spatial_rate_constant):
        if isinstance(self.args, tuple):
            return self.args[0].einsum_index_symbol_string(spatial_dim, spatial_rate_constant)
        else:
            return self.args.einsum_index_symbol_string(spatial_dim, spatial_rate_constant)
    
    def __str__(self):
        return repr(self)
    
    def __repr__(self):
        return self.fn.__name__ + "(" + repr(self.args) + ")"
    
    def __eq__(self, other):
        return self.fn == other.fn and self.args == other.args


def relu(x): return RateConstantFunction(jax.nn.relu, x)

class EinsumOnes():
    def __init__(self, index_symbol):
        self._index_symbol = index_symbol

        shape = index_symbol.index_set
        self._jnp_ones = jnp.ones(shape)

    @property
    def index_symbol(self):
        return self._index_symbol
    
    @property
    def jnp_ones(self):
        return self._jnp_ones
    
    def einsum_index_symbol_string(self, *_):
        return str(self.index_symbol)
    
    def eval(self, _):
        return self.jnp_ones
    
    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "jnp_ones_" + str(self.index_symbol)

def mass_action(reactants, products, aux, spatial_dim, spatial_rate_constant):
    r_dict = reactants.count_dict
    r_idx = reactants.get_index_symbols_set(spatial_dim)

    p_dict = products.count_dict
    p_idx = products.get_index_symbols_set(spatial_dim)

    species_set = set(r_dict.keys()) | set(p_dict.keys())

    tensor_list = list(r_dict.items())
    tensor_list.append((aux, 1))

    non_p_idx = r_idx

    if isinstance(aux, IndexedObject):
        for idx in aux.get_index_symbols_set():
            non_p_idx.add(idx)

    for idx in p_idx.difference(non_p_idx):
        tensor_list.append((EinsumOnes(idx), 1))

    spatial_dim_bool = spatial_dim is not None
    base_str_list = list(map(lambda x : x[0].einsum_index_symbol_string(spatial_dim_bool, spatial_rate_constant), tensor_list))

    base_str = ",".join(base_str_list)

    base_str = ",".join(base_str_list)

    einsum_dict = dict()

    for s in species_set:
        diff = p_dict.get(s, 0) - r_dict.get(s, 0)

        if not diff == 0:
            key = s
            
            if isinstance(s, IndexedSpecies):
                key = s.base

            einsum_dict[s] = (diff, key, base_str + "->" + s.einsum_index_symbol_string(spatial_dim_bool, spatial_rate_constant))

    def get_tensor_vals(tensor_data):
        def apply_pow(item):
            tensor, count = item
            val = tensor.eval(tensor_data)
            
            if count > 1:
                return jnp.pow(val, count)
            else:
                return val

        val_list = list(map(apply_pow, tensor_list))

        return val_list
    
    def dynamics_func(tensor_data):
        dynamics_dict = dict()

        vals = get_tensor_vals(tensor_data)

        for s, (diff, key, einsum_str) in einsum_dict.items():

            dynamics_dict[key] = dynamics_dict.get(key, 0) + diff * jnp.einsum(einsum_str, *vals)
        return dynamics_dict
    
    return dynamics_func

def get_aux_shape(aux):
    if isinstance(aux, RateConstant):
        return {aux : ()}
    elif isinstance(aux, IndexedRateConstant):
        return {aux.base : tuple(map(lambda x : x.index_set, aux.index_symbols))}
    elif isinstance(aux, NumericRateConstant):
        return None
    elif isinstance(aux, ConcreteRateConstant):
        return None
    else:
        if isinstance(aux.args, tuple):
            return {k : v for shape_dict in map(get_aux_shape, aux.args) for k, v in shape_dict.items()}
        else:
            return get_aux_shape(aux.args)

class BulkReaction(IndexedObject):
    def __init__(self, reactants, products, aux, rule=mass_action, name=None):
        if isinstance(reactants, int) and reactants == 0:
            self._reactants = Complex({})
        elif not isinstance(reactants, Complex):
            self._reactants = Complex({reactants : 1})
        else:
            self._reactants = reactants

        if isinstance(products, int) and products == 0:
            self._products = Complex({})
        elif not isinstance(products, Complex):
            self._products = Complex({products : 1})
        else:
            self._products = products

        if isinstance(aux, RateConstantExpr):
            self._aux = aux
        else:
            self._aux = NumericRateConstant(aux)


        self._rule = rule
        self._name = name


    @property
    def name(self):
        return self._name

    @property
    def reactants(self):
        return self._reactants

    @property
    def products(self):
        return self._products

    @property
    def aux(self):
        return self._aux
    
    @property
    def rule(self):
        return self._rule

    def __eq__(self, other):
        return self.name == other.name and \
               self.reactants == other.reactants and \
               self.products == other.products and \
               self.aux == other.aux and \
               self.rule == other.rule
    
    def shapes(self):
        r_dict = self.reactants.count_dict
        p_dict = self.products.count_dict

        shapes_dict = dict()

        for s in r_dict.keys() | p_dict.keys():
            shape = ()
            key = s

            if isinstance(s, IndexedSpecies):
                shape = tuple(map(lambda x : x.index_set, s.index_symbols))
                key = s.base

            if isinstance(s, ConcreteSpecies):
                continue

            if key in shapes_dict:
                if shapes_dict[key] != shape:
                    raise RepresentationError
            else:
                shapes_dict[key] = shape

        aux_shape_dict = get_aux_shape(self.aux)
        if aux_shape_dict is not None:
            shapes_dict = shapes_dict | aux_shape_dict

        return shapes_dict
        
    def get_index_symbols_set(self):
        return_set = self.reactants.get_index_symbols_set() | self.products.get_index_symbols_set()
        if isinstance(self.aux, IndexedRateConstant) or isinstance(self.aux, RateConstantFunction):
            return_set = return_set | self.aux.get_index_symbols_set()

        return return_set

    def index_symbols_replace(self, values):
        new_aux = self.aux

        if isinstance(self.aux, IndexedRateConstant) or isinstance(self.aux, RateConstantFunction):
            new_aux = self.aux.index_symbols_replace(values)

        
        return BulkReaction(
            self.reactants.index_symbols_replace(values),
            self.products.index_symbols_replace(values),
            new_aux,
            self.rule,
            self.name
        )

    def build_flux(self, spatial_dim, spatial_rate_constant):
        return self.rule(self.reactants, self.products, self.aux, spatial_dim, spatial_rate_constant)

    def enumerate(self):
        idx_syms_list = list(self.get_index_symbols_set())
        idx_syms_list.sort()
        combos = product(*list(map(lambda x : range(x.index_set), idx_syms_list)))

        return [self.index_symbols_replace(dict(zip(idx_syms_list, combo))) for combo in combos]
        

MassActionReaction = BulkReaction

class FastReaction(IndexedObject):
    def __init__(self, reactants, products, name=None):
        if isinstance(reactants, int) and reactants == 0:
            self._reactants = Complex({})
        elif not isinstance(reactants, Complex):
            self._reactants = Complex({reactants : 1})
        else:
            self._reactants = reactants

        if isinstance(products, int) and products == 0:
            self._products = Complex({})
        elif not isinstance(products, Complex):
            self._products = Complex({products : 1})
        else:
            self._products = products

        self._name = name


        r_dict = self.reactants.count_dict
        p_dict = self.products.count_dict

        species_set = set(r_dict.keys()) | set(p_dict.keys())

        def f(tensor_data):
            delta_unit = jnp.inf

            for (s, c) in r_dict.items():
                delta_unit = jnp.minimum(delta_unit, s.eval(tensor_data) / c)

            return_dict = dict()

            for s in species_set:
                diff = p_dict.get(s, 0) - r_dict.get(s, 0)

                if not diff == 0:
                    key = s
                    if isinstance(s, IndexedSpecies):
                        key = s.base

                    return_dict[key] = return_dict.get(key, 0) + diff * delta_unit
            return return_dict
        self.dynamics_func = f
    @property
    def name(self):
        return self._name

    @property
    def reactants(self):
        return self._reactants

    @property
    def products(self):
        return self._products
    
    def __eq__(self, other):
        return self.name == other.name and \
               self.reactants == other.reactants and \
               self.products == other.products
    
    def shapes(self):
        r_dict = self.reactants.count_dict
        p_dict = self.products.count_dict

        shapes_dict = dict()

        for s in r_dict.keys() | p_dict.keys():
            shape = ()
            key = s

            if isinstance(s, IndexedSpecies):
                shape = tuple(map(lambda x : x.index_set, s.index_symbols))
                key = s.base

            if isinstance(s, ConcreteSpecies):
                continue

            if key in shapes_dict:
                if shapes_dict[key] != shape:
                    raise RepresentationError
            else:
                shapes_dict[key] = shape

        return shapes_dict
    
    def get_index_symbols_set(self):
        return_set = self.reactants.get_index_symbols_set() | self.products.get_index_symbols_set()
        return return_set

    def index_symbols_replace(self, values):
        return FastReaction(
            self.reactants.index_symbols_replace(values),
            self.products.index_symbols_replace(values),
            self.name
        )
    
    def flux(self, tensor_data):
        return self.dynamics_func(tensor_data)

    def enumerate(self):
        idx_syms_list = list(self.get_index_symbols_set())
        idx_syms_list.sort()

        combos = product(*list(map(lambda x : range(x.index_set), idx_syms_list)))

        return [self.index_symbols_replace(dict(zip(idx_syms_list, combo))) for combo in combos]


class ICRN():
    """
    Class for representing an ICRN.
    """
    def __init__(self, reactions) -> None:
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
    
    @property
    def normal_reactions(self):
        return self._normal_reactions
    
    @property
    def fast_reactions(self):
        return self._fast_reactions
    
    def __repr__(self):
        return repr(self.reactions) + repr(self.fast_reactions)
    
    def shapes(self):
        return {s : shape for reaction in self._reactions for s, shape in reaction.shapes().items()}

    def dynamics(self, spatial_dim, spatial_rate_constant):
        def jittable_fast_dynamics(tensor_data):
            dynamics_dict = dict()

            for rxn in self.fast_reactions:
                for s, arr in rxn.flux(tensor_data).items():
                    dynamics_dict[s] = dynamics_dict.get(s, 0) + arr

            return dynamics_dict


        flux_list = [rxn.build_flux(spatial_dim, spatial_rate_constant) for rxn in self.normal_reactions]

        def jittable_normal_dynamics(tensor_data):
            dynamics_dict = dict()

            for flux in flux_list:
                for s, arr in flux(tensor_data).items():
                    dynamics_dict[s] = dynamics_dict.get(s, 0) + arr

            return dynamics_dict

        return jittable_fast_dynamics, jittable_normal_dynamics
    
    def enumerate(self):
        return ICRN([enum_r for r in self.reactions for enum_r in r.enumerate()])