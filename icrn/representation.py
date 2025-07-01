"""
representation.py
Module for building ICRNs.
"""
from abc import ABC, abstractmethod
from sympy import Idx, IndexedBase, Indexed, oo, S, Pow, Mul, Min, Add, Function, Atom, Tuple, symbols, Symbol, sympify, Number, Expr

from .compiler import toeinsum, unify, standardize, lambdify_expr, tomin, get_bases
from .dict_utils import sjdict_builder
from itertools import product

_SUBINDEX = 0x03B1
zero = S.Zero
fast = oo

class RepresentationError(Exception):
    pass

class IndexSymbol(Idx):
    def __init__(self, label, range_val) -> None:
        super().__init__()
        self.range = range_val

    def __lt__(self, other):
        return str(self) < str(other)

    def __gt__(self, other):
        return not self < other

    def __le__(self, other):
        return self < other or self == other

    def __ge__(self, other):
        return not self <= other

def many_index_symbols(names, range_val):
    return symbols(names, cls= lambda x : IndexSymbol(x, range_val))

class ICRNBase(ABC):
    def __lt__(self, other):
        return str(self) < str(other)

    def __gt__(self, other):
        return not self < other

    def __le__(self, other):
        return self < other or self == other

    def __ge__(self, other):
        return not self <= other
        

class Species(ICRNBase, IndexedBase):
    pass

def many_species(names):
    return symbols(names, cls=Species)

class RateConstant(ICRNBase, IndexedBase):
    pass

def many_rate_constants(names):
    return symbols(names, cls=RateConstant)

def get_base_from_sym(sym):
    if isinstance(sym, Indexed):
        return sym.base
    elif isinstance(sym, IndexedBase):
        return sym
    else:
        return None

def extract_bases(exp):
    if isinstance(exp, Number):
        return set()
    elif isinstance(exp, ICRNBase):
        return {exp}
    elif isinstance(exp, Indexed):
        return {exp.base}
    elif isinstance(exp, Expr):
        return {sym for arg in exp.args for sym in extract_bases(arg)}
    else:
        return set()

def extract_index_symbols(exp):
    return {sym for sym in exp.free_symbols if isinstance(sym, IndexSymbol)}

def extract_indexed_bases(exp):
    return {sym for sym in exp.free_symbols if isinstance(sym, Indexed)}

class Reaction(ABC):
    def __init__(self, reactants, products, aux=None, name=None, fast=False) -> None:
        self._name = name
        self._reactants = sympify(reactants)
        self._products = sympify(products)
        self._aux = sympify(aux)
        self._fast = fast

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
    def fast(self):
        return self._fast

    @abstractmethod
    def flux(self):
        pass

    @abstractmethod
    def bases(self):
        pass

    @abstractmethod
    def indexed(self):
        pass

    def shapes(self):
        i_bases = self.indexed()
        i_bases_shapes = {s.base : s.shape for s in i_bases}
        
        bases = self.bases()
        bases_shapes = {s : () for s in bases}

        return bases_shapes | i_bases_shapes

    def index_symbols(self):
        # returns index symbols in alphabetical order
        idx_syms_set = {idx_sym for syms in [self.reactants, self.products, self.aux] for idx_sym in extract_index_symbols(syms)}
        idx_syms_list = list(idx_syms_set)
        return sorted(idx_syms_list)
    
    def index_symbols_replace(self, idx_vals_tup):
        idx_syms_list = self.index_symbols()
        idx_vals_list = list(idx_vals_tup)

        new_reactants = self.reactants
        new_products = self.products
        new_aux = self.aux

        for idx_sym, idx_val in zip(idx_syms_list, idx_vals_list):
            new_reactants = new_reactants.replace(idx_sym, idx_val)
            new_products = new_products.replace(idx_sym, idx_val)
            new_aux = new_aux.replace(idx_sym, idx_val)

        new_reaction = self.__new__(type(self))
        new_reaction.__init__(new_reactants, new_products, new_aux)

        return new_reaction

    def __call__(self, *index_vals):
        return self.index_symbols_replace(index_vals)

    def enumerate(self):
        idx_syms_list = self.index_symbols()
        combos = product(*map(lambda x : range(x.range), idx_syms_list))

        return [self(*combo) for combo in combos]

    def species(self):
        species_set = extract_bases(self._reactants) | extract_bases(self._products)
        species_list = list(species_set)

        return sorted(species_list)
    
    def __repr__(self):
        return "(" + repr(self.reactants) + ", " + repr(self.products) + ", " + repr(self.aux) + ")"

class MassActionReaction(Reaction):
    def __init__(self, reactants, products, rate_constant, **kwargs) -> None:
        super().__init__(reactants, products, rate_constant, **kwargs)

    @property
    def rate_constant(self):
        return self._aux
    
    def bases(self):
        species_bases_set = set(self.species())
        return species_bases_set | extract_bases(self.rate_constant)

    def indexed(self):
        res_set = extract_indexed_bases(self._reactants) | extract_indexed_bases(self._products)

        if isinstance(self.rate_constant, Number):
            return res_set
        else:
            return res_set | extract_indexed_bases(self._aux)

    def flux(self):
        diff = (self.products - self.reactants)
        diff_c_dict = diff.as_coefficients_dict()

        reactant_c_dict = self.reactants.as_coefficients_dict()


        power_list = [Pow(s, c) for s, c in reactant_c_dict.items()]

        res_dict = dict()
        for s, c in diff_c_dict.items():
            res_dict[s] = c * toeinsum(self.rate_constant, Tuple(*power_list))

        return res_dict

class FastReaction(Reaction):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, fast=True, **kwargs)

    def flux(self):
        diff = (self.products - self.reactants)
        diff_c_dict = diff.as_coefficients_dict()

        reactant_c_dict = self.reactants.as_coefficients_dict()
        reactant_bases = {get_base_from_sym(s) : c for (s, c) in reactant_c_dict.items()}
        min_args = [(s / c) for (s, c) in reactant_bases.items()]
        min_expr = tomin(*min_args)
        res = dict()
        for s in diff_c_dict.keys():
            res[s] = - min_expr

        return res

    def bases(self):
        return set(self.species())

    def indexed(self):
        return extract_indexed_bases(self._reactants) | extract_indexed_bases(self._products)

class MichaelisMentenReaction(Reaction):
    pass

class ICRN():
    def __init__(self, reactions) -> None:
        self._reactions = reactions

    @property
    def reactions(self):
        return self._reactions
    
    def __repr__(self):
        return repr(self.reactions)

    def reactions_by_timescale(self):
        normal_reactions = []
        fast_reactions = []

        for rxn in self.reactions:
            if rxn.fast:
                fast_reactions.append(rxn)
            else:
                normal_reactions.append(rxn)

        return normal_reactions, fast_reactions

    def species(self):
        return {s for reaction in self._reactions for s in reaction.get_species()}

    def bases(self):
        return {b for reaction in self._reactions for b in reaction.bases()}

    def bases_list(self):
        bases_set = {b for reaction in self._reactions for b in reaction.bases()}
        return sorted(list(bases_set))
    
    def shapes(self):
        return {s : shape for reaction in self._reactions for s, shape in reaction.shapes().items()}

    def WM_dynamics_expr(self):
        print("separating reactions by timescale")
        normal_reactions, fast_reactions = self.reactions_by_timescale()

        normal_dynamics_expr = standardize(unify(normal_reactions))
        fast_dynamics_expr = standardize(unify(fast_reactions))

        return normal_dynamics_expr, fast_dynamics_expr

    def enumerate(self):
        return [enum_r for r in self.reactions for enum_r in r.enumerate()]

    def WM_enumerate_dynamics(self):
        enum_rxns = self.enumerate()
        u_dict = unify(enum_rxns)
        return {s : exp for _, f_dict in u_dict.items() for s, exp in f_dict.items()}
    
    def lambdify(self, expr):
        lambdify_expr(self.bases(), expr)