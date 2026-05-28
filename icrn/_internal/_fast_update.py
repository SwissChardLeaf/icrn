from collections import defaultdict

import jax.numpy as jnp


def _standard_reactants_and_indexing(reactants):
    reactants_lst = list(reactants.count_dict.keys())
    reactants_lst.sort()
    standard_reactants = tuple(reactants_lst)
    return standard_reactants, reactants_lst[0].index_symbols


def _build_base_einsum_str(standard_indexing):
    return "".join(map(str, standard_indexing)) + "->"


def _einsum_prep(reactants, products, standard_indexing):
    einsum_prep = defaultdict(list)
    base_einsum_str = _build_base_einsum_str(standard_indexing)

    for s in reactants.count_dict.keys() | products.count_dict.keys():
        product_count = products.count_dict.get(s, 0)
        reactant_count = reactants.count_dict.get(s, 0)

        diff = product_count - reactant_count

        if diff != 0:
            einsum_str = base_einsum_str + "".join(map(str, s.index_symbols))
            einsum_prep[s[()]].append((diff, einsum_str))

    return einsum_prep


def _get_reactant_unit(standard_reactants, reactants, state):
    ratioed_tensors = [
        s.eval(state) / reactants.count_dict[s] for s in standard_reactants
    ]

    n = len(ratioed_tensors)
    run_min = ratioed_tensors[0]
    for i in range(1, n):
        run_min = jnp.minimum(run_min, ratioed_tensors[i])

    return run_min


def _fast_update_f(reactants, products):
    standard_reactants, standard_indexing = _standard_reactants_and_indexing(
        reactants
    )
    einsum_prep = _einsum_prep(reactants, products, standard_indexing)

    def fast_update(state):
        reactant_unit = _get_reactant_unit(standard_reactants, reactants, state)

        def lst_helper(lst):
            acc = 0
            for ratio, einsum_str in lst:
                try:
                    acc += ratio * jnp.einsum(einsum_str, reactant_unit)
                except Exception as e:
                    raise Exception(
                        f"Error einsumming {einsum_str} with reactant unit {reactant_unit} for species {s} with reactants {reactants} and products {products}"
                    )
            return acc

        output = dict()
        for s, prep_lst in einsum_prep.items():
            output[s] = lst_helper(prep_lst)
        return output

    return fast_update
