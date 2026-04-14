from ..representation.symbols import Complex, Species, TensorExpression, TensorLiteral, Numeric
import jax
import jax.numpy as jnp
from collections import defaultdict


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
    for s in reactants.count_dict.keys():
        parts.append(_tensor_expression_to_index_str(s))

    product_index_symbols_lst = list(product_index_symbols)
    product_index_symbols_lst.sort()
    product_index_symbols_str = list(map(str, product_index_symbols_lst))
    parts.extend(product_index_symbols_str)
    return ",".join(parts) + "->"


def _product_index_symbols(reactants, products, rate_exp):
    all_index_symbols = _get_all_index_symbols(reactants, products, rate_exp)
    reactants_rate_exp_index_symbols = _get_all_index_symbols(reactants, rate_exp)
    return all_index_symbols - reactants_rate_exp_index_symbols


def _get_diff_and_einsum_strs(
    diff_dict, reactants, rate_exp, product_index_symbols, base_einsum_str
):
    einsum_strs = defaultdict(list)

    for s, diff in diff_dict.items():
        einsum_str = base_einsum_str + _tensor_expression_to_index_str(s)
        einsum_strs[s[()]].append((diff, einsum_str))

    return einsum_strs


def _get_tensors(reactants, rate_expr, product_index_symbols):
    if reactants == Complex({}):
        r_pow_list = [(rate_expr, 1)]
    else:
        r_pow = reactants.count_dict
        r_pow_list = list(r_pow.items())
        r_pow_list.sort(key=lambda x: x[0])
        r_pow_list = [(rate_expr, 1)] + r_pow_list

    product_index_symbols_lst = list(product_index_symbols)
    product_index_symbols_lst.sort()

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
            return x.eval_with_check(non_state | state)
        return x.eval_with_check(non_state | state) ** p

    def get_tensor(state, non_state):
        return tuple(map(lambda x: get_pow(x, state, non_state), r_pow_list))

    return get_tensor


def _setup_einsums(reactants, products, rate_exp):
    product_index_symbols = _product_index_symbols(reactants, products, rate_exp)
    base_einsum_str = _get_base_einsum_str(reactants, rate_exp, product_index_symbols)

    diff_dict = _get_diff_dict(reactants, products)
    einsum_prep = defaultdict(list)

    return _get_diff_and_einsum_strs(
        diff_dict, reactants, rate_exp, product_index_symbols, base_einsum_str
    )

    # for s, diff in diff_dict.items():
    #     mod, einsum_str = _get_mod_and_einsum_str(s, product_index_symbols, base_einsum_str)
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
    product_index_symbols = _product_index_symbols(reactants, products, rate_exp)
    get_tensors = _get_tensors(reactants, rate_exp, product_index_symbols)

    def _sum_of_list(lst, tensors):
        acc = 0
        for diff, einsum_str in lst:
            acc += diff * jnp.einsum(einsum_str, *tensors)
        return acc

    def f(state, non_state):
        tensors = get_tensors(state, non_state)
        return {
            s: _sum_of_list(einsum_info_lst, tensors)
            for s, einsum_info_lst in einsum_prep.items()
        }

    return f
