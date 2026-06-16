from collections import defaultdict

import jax.numpy as jnp

from ..symbols import (
    Complex,
    Numeric,
    TensorExpression,
)


def _tensor_expression_to_index_str(expr: TensorExpression) -> str:
    return "".join(map(str, expr.index_symbols))


def _get_diff_dict(reactants: Complex, products: Complex):
    diff_dict = {}
    keys = reactants.count_dict.keys() | products.count_dict.keys()
    for s in keys:
        r_c = reactants.count_dict.get(s, 0)
        p_c = products.count_dict.get(s, 0)
        d = p_c - r_c
        if d != 0:
            diff_dict[s] = d
    return diff_dict


def _get_all_index_symbols(*args):
    all_index_symbols = set()
    for arg in args:
        if isinstance(arg, Complex):
            for s in arg.count_dict.keys():
                all_index_symbols.update(s.index_symbols)
        elif isinstance(arg, TensorExpression):
            all_index_symbols.update(arg.index_symbols)
        else:
            raise ValueError(f"Unsupported argument type {type(arg)}")
    return all_index_symbols


def _get_base_einsum_str(
    reactants: Complex, rate_exp: TensorExpression, product_index_symbols: set
):
    parts = [_tensor_expression_to_index_str(rate_exp)]

    r_list = sorted(reactants.count_dict.keys())

    for s in r_list:
        parts.append(_tensor_expression_to_index_str(s))

    product_index_symbols_lst = list(product_index_symbols)
    product_index_symbols_lst.sort()
    product_index_symbols_str = list(map(str, product_index_symbols_lst))
    parts.extend(product_index_symbols_str)
    return ",".join(parts) + "->"


def _product_index_symbols(reactants, products, rate_exp):
    all_index_symbols = _get_all_index_symbols(reactants, products, rate_exp)
    reactants_rate_exp_index_symbols = _get_all_index_symbols(
        reactants, rate_exp
    )
    return all_index_symbols - reactants_rate_exp_index_symbols


def _get_diff_and_einsum_strs(
    diff_dict, reactants, rate_exp, product_index_symbols, base_einsum_str
):
    einsum_strs = defaultdict(list)

    for s, diff in diff_dict.items():
        einsum_str = base_einsum_str + _tensor_expression_to_index_str(s)
        einsum_strs[s[()]].append((diff, einsum_str, s))

    return einsum_strs


def _get_tensors(reactants, rate_expr, product_index_symbols):
    if reactants == Complex({}):
        r_pow_list = [(rate_expr, 1)]
    else:
        r_pow = reactants.count_dict
        r_pow_list = list(r_pow.items())
        r_pow_list.sort(key=lambda x: x[0])
        r_pow_list = [(rate_expr, 1)] + r_pow_list

    product_index_symbols_lst = sorted(product_index_symbols)

    for idx in product_index_symbols_lst:
        if idx.index_set > 0:
            r_pow_list.append(((jnp.ones(idx.index_set)), 1))
        else:
            raise ValueError(f"Index set must be specified for {idx}")

    def get_pow(pow_pair, state, non_state):
        x, p = pow_pair
        if isinstance(x, Numeric):
            return x

        if p == 1:
            return x.eval(non_state | state)
        return x.eval(non_state | state) ** p

    def get_tensor(state, non_state):
        return tuple(map(lambda x: get_pow(x, state, non_state), r_pow_list))

    return get_tensor


def _setup_einsums(reactants, products, rate_exp):
    product_index_symbols = _product_index_symbols(
        reactants, products, rate_exp
    )
    base_einsum_str = _get_base_einsum_str(
        reactants, rate_exp, product_index_symbols
    )

    diff_dict = _get_diff_dict(reactants, products)
    defaultdict(list)

    return _get_diff_and_einsum_strs(
        diff_dict, reactants, rate_exp, product_index_symbols, base_einsum_str
    )

    # for s, diff in diff_dict.items():
    #     mod, einsum_str = _get_mod_and_einsum_str(
    #         s, product_index_symbols, base_einsum_str
    #     )
    #     einsum_prep[s[()]].append((mod * diff, einsum_str))

    # return einsum_prepr


def mass_action_flux_f(
    reactants: Complex,
    products: Complex,
    rate_exp: TensorExpression,
):
    if not isinstance(reactants, Complex):
        raise ValueError(f"Reactants must be a Complex, got {type(reactants)}")
    if not isinstance(products, Complex):
        raise ValueError(f"Products must be a Complex, got {type(products)}")
    if not isinstance(rate_exp, TensorExpression):
        raise ValueError(
            f"Rate expression must be a TensorExpression, got {type(rate_exp)}"
        )

    einsum_prep = _setup_einsums(reactants, products, rate_exp)
    product_index_symbols = _product_index_symbols(
        reactants, products, rate_exp
    )
    get_tensors = _get_tensors(reactants, rate_exp, product_index_symbols)

    def _sum_of_list(lst, tensors):
        acc = 0
        for diff, einsum_str, spec in lst:
            try:
                acc += diff * jnp.einsum(einsum_str, *tensors)
            except Exception:
                # print(f"Error einsumming {einsum_str} with tensors {tensors}")
                raise Exception(
                    f"Error einsumming {einsum_str} with tensors {tensors} "
                    f"for species {spec} with reactants {reactants} and "
                    f"products {products} and rate expression {rate_exp}"
                )
        return acc

    def f(state, non_state):
        tensors = get_tensors(state, non_state)
        return {
            s: _sum_of_list(einsum_info_lst, tensors)
            for s, einsum_info_lst in einsum_prep.items()
        }

    return f


def mass_action_flux_pd_f(
    reactants: Complex,
    products: Complex,
    rate_exp: TensorExpression,
):
    if not isinstance(reactants, Complex):
        raise ValueError(f"Reactants must be a Complex, got {type(reactants)}")
    if not isinstance(products, Complex):
        raise ValueError(f"Products must be a Complex, got {type(products)}")
    if not isinstance(rate_exp, TensorExpression):
        raise ValueError(
            f"Rate expression must be a TensorExpression, got {type(rate_exp)}"
        )

    einsum_prep = _setup_einsums(reactants, products, rate_exp)
    product_index_symbols = _product_index_symbols(
        reactants, products, rate_exp
    )
    get_tensors = _get_tensors(reactants, rate_exp, product_index_symbols)

    def _prod_dest_of_list(lst, tensors):
        prod = 0
        dest = 0
        for diff, einsum_str, spec in lst:
            try:
                rate = jnp.einsum(einsum_str, *tensors)
            except Exception:
                raise Exception(
                    f"Error einsumming {einsum_str} with tensors {tensors} "
                    f"for species {spec} with reactants {reactants} and "
                    f"products {products} and rate expression {rate_exp}"
                )
            if diff > 0:
                prod = prod + diff * rate
            else:
                dest = dest - diff * rate
        return prod, dest

    def f(state, non_state):
        tensors = get_tensors(state, non_state)
        production = {}
        destruction = {}
        for s, einsum_info_lst in einsum_prep.items():
            prod, dest = _prod_dest_of_list(einsum_info_lst, tensors)
            production[s] = prod
            destruction[s] = dest
        return production, destruction

    return f


_VALID_MPE_SPLITS = ("uniform", "stoichiometry")


def _has_indexed_species(*complexes: Complex) -> bool:
    """Return whether any species in the given complexes is indexed."""
    for c in complexes:
        for s in c.count_dict.keys():
            if s.index_symbols:
                return True
    return False


def _split_weights(reactant_counts: dict, split: str) -> dict:
    """Compute the production-attribution weights for one reaction.

    The modified Patankar method attributes each unit of a product's
    production to the reactants that drive the reaction. This helper turns a
    reaction's reactant multiset into a normalised weight per reactant
    species, according to the chosen ``split`` strategy.

    Parameters
    ----------
    reactant_counts : dict[Species, int]
        Stoichiometric coefficient of each (base) reactant species.
    split : {"uniform", "stoichiometry"}
        Strategy used to distribute a product's production across the
        reactants:

        - ``"uniform"`` weights every distinct reactant equally.
        - ``"stoichiometry"`` weights each reactant by its stoichiometric
          coefficient.

    Returns
    -------
    dict[Species, float]
        Weights summing to 1, keyed by base species. Empty when the
        reaction has no reactants.

    Raises
    ------
    ValueError
        If ``split`` is not recognised.
    """
    if split not in _VALID_MPE_SPLITS:
        raise ValueError(
            f"Invalid split {split!r}; expected one of {_VALID_MPE_SPLITS}"
        )

    if not reactant_counts:
        return {}

    if split == "uniform":
        n = len(reactant_counts)
        return {s: 1.0 / n for s in reactant_counts}

    # split == "stoichiometry"
    total = sum(reactant_counts.values())
    return {s: m / total for s, m in reactant_counts.items()}


def _reaction_rate_f(reactants: Complex, rate_exp: TensorExpression):
    """Build the scalar mass-action rate of a single reaction.

    Parameters
    ----------
    reactants : Complex
        Reactant multiset of the reaction.
    rate_exp : TensorExpression
        Rate expression (e.g. a rate constant) of the reaction.

    Returns
    -------
    callable
        A function ``r(state, non_state)`` returning the reaction rate
        ``rate_exp * prod_s state[s] ** m_s`` for non-indexed species.
    """
    reactant_items = [(s[()], m) for s, m in reactants.count_dict.items()]

    def r(state, non_state):
        val = rate_exp.eval(non_state | state)
        for s, m in reactant_items:
            val = val * state[s] ** m
        return val

    return r


def mass_action_flux_pairs_f(
    reactants: Complex,
    products: Complex,
    rate_exp: TensorExpression,
    split: str = "uniform",
):
    """Build the production-destruction terms used by the modified Patankar
    step.

    Unlike [`mass_action_flux_pd_f`]
    [icrn._internal._mass_action.mass_action_flux_pd_f],
    which only lumps each species' production and destruction, this function
    also attributes every unit of production to the reactant species that
    caused it. That attribution is what lets the modified Patankar method
    weight each production term by its source concentration and assemble a
    linearly-implicit, positivity-preserving update.

    Parameters
    ----------
    reactants : Complex
        Reactant multiset of the reaction.
    products : Complex
        Product multiset of the reaction.
    rate_exp : TensorExpression
        Rate expression of the reaction.
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

        - ``destruction`` maps a base species to its lumped destruction
          rate (used to build the matrix diagonal),
        - ``pairs`` maps an ordered ``(product, reactant)`` pair of base
          species to the production of the product charged to that reactant
          (used to build the off-diagonal entries),
        - ``explicit`` maps a base species to production with no reactant
          source (a sourceless influx), to be added explicitly.

    Raises
    ------
    NotImplementedError
        If any reactant or product species is indexed; the modified
        Patankar step currently supports non-indexed species only.

    See Also
    --------
    mass_action_flux_pd_f : Lumped production/destruction without source
        attribution.
    """
    if _has_indexed_species(reactants, products):
        raise NotImplementedError(
            "The modified Patankar method currently supports non-indexed "
            "species only."
        )

    rate_f = _reaction_rate_f(reactants, rate_exp)
    diff_dict = _get_diff_dict(reactants, products)
    reactant_counts = {s[()]: m for s, m in reactants.count_dict.items()}
    weights = _split_weights(reactant_counts, split)

    def f(state, non_state):
        rate = rate_f(state, non_state)
        destruction = {}
        pairs = {}
        explicit = {}
        for s, diff in diff_dict.items():
            base = s[()]
            if diff > 0:
                produced = diff * rate
                if weights:
                    for j, w in weights.items():
                        key = (base, j)
                        pairs[key] = pairs.get(key, 0.0) + w * produced
                else:
                    explicit[base] = explicit.get(base, 0.0) + produced
            elif diff < 0:
                destruction[base] = destruction.get(base, 0.0) + (-diff) * rate
        return destruction, pairs, explicit

    return f
