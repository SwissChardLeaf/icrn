from sympy.printing.lambdarepr import LambdaPrinter
from sympy import lambdify, Function, Indexed, Atom, S, Idx, Add, Tuple, Min, Symbol
from collections import defaultdict
import jax.numpy as jnp
from jax import vmap, nn
from icrn.numerics import compute_lap_op, fast_react, diffuse, INT_METHOD_DICT
from icrn.dict_utils import SJDict, map1

_SUBINDEX = 0x03B1

def get_base_from_sym(sym):
    if isinstance(sym, Indexed):
        return sym.base
    else:
        return sym

def get_bases(exp):
    return {_get_base_from_sym(b) for b in exp.free_symbols if isinstance(b, Indexed | Symbol)}

def get_index_symbols(exp):
    return {sym for sym in exp.free_symbols if isinstance(sym, Idx)}

def get_indexed_bases(exp):
    for sym in exp.free_symbols:
        if isinstance(sym, Indexed):
            return sym
    return None

class toeinsum(Function):
    pass

class tomin(Function):
    pass

def my_tomin(*x):
    return jnp.minimum(*x)

class SympyStr(Atom):
    def __str__(self):
        return self.args[0]

class custom_printer(LambdaPrinter):
    def _print_SympyStr(self, expr):
        return "'" + expr.args[0] + "'"

    def _print_jnpeinsum(self, expr):
        return '%s(%s)' % (self._print(expr.func), ", ".join(map(self._print, expr.args)))
    
    def _print_relu(self, expr):
        return '%s(%s)' % (self._print(expr.func), ", ".join(map(self._print, expr.args)))
    
    def _print_tomin(self, expr):
        return '%s(%s)' % (self._print(expr.func), ", ".join(map(self._print, expr.args)))

class relu(Function):
    pass

def my_relu(x):
    return nn.relu(x)

class jnpeinsum(Function):
    pass

def my_jnpeinsum(einsum_str, args):
    return jnp.einsum(einsum_str, *args)

def _get_base_from_sym(sym):
    if isinstance(sym, Indexed):
        return sym.base
    else:
        return sym
    
def unify(rxns):
    print("unifying")
    flux_list = [rxn.flux() for rxn in rxns]

    ind_species_dict = defaultdict(list)

    for flux in flux_list:
        for s, exp in flux.items():
            ind_species_dict[s].append(exp)

    base_species_dict = defaultdict(dict)

    for ind_s, flux_list in ind_species_dict.items():
        base_species_dict[_get_base_from_sym(ind_s)][ind_s] = Add(*flux_list)

    return base_species_dict

def contains_index_symbol(indices):
    for idx in indices:
        if isinstance(idx, Idx):
            return True
    
    return False

def standardize(u_dict):
    print("standardizing")
    res_dict = dict()

    count = 0
    for base_species, flux_dict in u_dict.items():
        # get new standard indices
        rep_ind_species = list(flux_dict.keys())[0]

        if isinstance(rep_ind_species, Indexed):
            # need to standardize index symbols
            # get new standard indices
            original_indices = rep_ind_species.indices

            new_indices = list()
            for o_index in original_indices:
                new_indices.append(Idx(chr(_SUBINDEX + count), range=o_index.upper+1))
                count += 1

            new_ind_species = Indexed(base_species, *new_indices)
            res_dict[new_ind_species] = 0

            # replace indices in flux dict
            for ind_species, flux_exp in flux_dict.items():
                old_indices = ind_species.indices

                new_exp = flux_exp
                for (old_i, new_i) in zip(old_indices, new_indices):
                    new_exp = new_exp.replace(old_i, new_i)

                res_dict[new_ind_species] += new_exp
        else:
            res_dict[base_species] = flux_dict[base_species]

    return res_dict

def extract_indexed_base(rc_expr):
    return

def build_jnp_einsum(s_sub, rc_expr, sp_syms):
    str_list = list()
    tensor_list = list()

    rc_expr_str = ""

    rc_indexed_base = get_indexed_bases(rc_expr)

    if rc_indexed_base is not None:
        rc_expr_base = rc_expr.replace(rc_indexed_base, rc_indexed_base.base)
        rc_expr_str += "".join(list(map(str, rc_indexed_base.indices)))
        tensor_list.append(rc_expr_base)
    else:
        tensor_list.append(rc_expr)

    str_list.append(rc_expr_str)

    for s in sp_syms:
        s_str = ""

        s_indexed_base = get_indexed_bases(s)

        if s_indexed_base is not None:
            s_str += "".join(list(map(str, s_indexed_base.indices)))

            s_expr_base = s.replace(s_indexed_base, s_indexed_base.base)
            tensor_list.append(s_expr_base)

        elif s == S.One:
            tensor_list.append(s)
        else:
            tensor_list.append(s)

        str_list.append(s_str)

    einsum_str = ",".join(str_list) + "->"

    if isinstance(s_sub, Indexed):
        if contains_index_symbol(s_sub.indices):
            einsum_str += "".join(list(map(str, s_sub.indices)))

    return jnpeinsum(SympyStr(einsum_str), Tuple(*tensor_list))

def build_min(s_ind, *args):
    args_tup = tuple(map(lambda x : x.base, args))
    return Min(*args_tup)

def sympy_to_tensor(dynamics_expr):
    res_dict = dict()
    for s_ind, exp in dynamics_expr.items():
        key = s_ind
        if isinstance(s_ind, Indexed):
            if contains_index_symbol(s_ind.indices):
                key = _get_base_from_sym(s_ind)

        exp = exp.replace(toeinsum, lambda *args : build_jnp_einsum(s_ind, *args))
        res_dict[key] = exp
    return res_dict

def lambdify_expr(base_symbols_list, expr):
    f = lambdify(base_symbols_list, expr,
                printer=custom_printer,
                modules=[{"jnpeinsum": my_jnpeinsum}, "jax"]
            )
    
    def g(state):
        str_state = {str(k):v for k,v in state.items()}
        f(**str_state)
    return g

def lambdify_dict(expr_dict, base_symbols_list):
    tensor_exprs  = sympy_to_tensor(expr_dict)

    lambdified_dict = dict()
    for s, expr in tensor_exprs.items():
        lambdified_dict[s] = lambdify(base_symbols_list, expr,
                                    printer=custom_printer,
                                    modules=[{
                                                "relu" : my_relu,
                                                "tomin" : my_tomin,
                                                "jnpeinsum": my_jnpeinsum
                                             }, 
                                            "jax"]
                                    )
    return lambdified_dict


def lambdify_func(l_dict, base_symbols_list, spatial_dim):
    def f(tensor_data):
        args_tup = tuple(map(lambda x : tensor_data.get(x, 0), base_symbols_list))
        res_dict = SJDict()
        for s, lambdified in l_dict.items():
            res_dict[s] = lambdified(*args_tup)
        return res_dict

    return f
    
def build_forward_step(rsys, spatial_dim, batch, integration_method, **kwargs):
    print("generating dynamics expressions")
    normal_dynamics_expr, fast_dynamics_expr = rsys.WM_dynamics_expr()
    base_symbols_list = rsys.bases_list()

    n_l_dict = lambdify_dict(normal_dynamics_expr, base_symbols_list)
    f_l_dict = lambdify_dict(fast_dynamics_expr, base_symbols_list)

    print("lambdifying expressions")
    def identity_f(tensor_data):
        return 0

    if normal_dynamics_expr:
        normal_reaction_f = lambdify_func(n_l_dict, base_symbols_list, spatial_dim)
    else:
        print("no normal reactions")
        normal_reaction_f = identity_f

    if fast_dynamics_expr:
        fast_reaction_f = lambdify_func(f_l_dict, base_symbols_list, spatial_dim)
    else:
        print("no fast reactions")
        fast_reaction_f = identity_f

    integrator = INT_METHOD_DICT[integration_method]

    def wm_f(concs, rate_data, diff_data, dt):
        concs = fast_react(concs, fast_reaction_f)
        return integrator(concs, rate_data, dt, normal_reaction_f)
    
    reaction_in_axes = (0, None, None, None)
    vec_wm_f = vmap(wm_f, in_axes=reaction_in_axes)

    if spatial_dim is None:
        if batch:
            # concs shape is (B, N)
            print("well-mixed, batched")
            return vec_wm_f
        else:
            # concs shape is (N,)
            print("well-mixed, unbatched")
            return wm_f
    else:
        lap_op = compute_lap_op(spatial_dim, dh=kwargs["dh"])

        if batch:
            # concs shape is (B * H * W, N)
            diffuse_in_axes = (0, None, None, None)
            vec_diffuse = vmap(diffuse, in_axes=diffuse_in_axes)

            def rd_f(concs, rate_data, diff_data, dt):
                concs_rs = vec_diffuse(concs_rs, diff_data, lap_op, dt)

                # logic for flattening
                user_shape = concs.shape
                new_shape = (user_shape[0] * user_shape[1] * user_shape[2],) + user_shape[3:]
                concs_rs = jnp.reshape(concs, new_shape)

                concs_rs = vec_wm_f(concs_rs, rate_data, dt)

                # unflattening
                concs = jnp.reshape(concs_rs, user_shape)
                return concs
            
            print("reaction diffusion, batched")
            return rd_f
        
        else:
            def flatten(arr):
                user_shape = arr.shape
                new_shape = (user_shape[0] * user_shape[1],) + user_shape[2:]
                return jnp.reshape(arr, new_shape)
            
            def unflatten(arr):
                user_shape = spatial_dim + arr.shape[1:]
                return jnp.reshape(arr, user_shape)

            # concs shape is (H * W, N)
            def rd_f(concs, rate_data, diff_data, dt):
                concs = diffuse(concs, diff_data, lap_op, dt)
                
                # logic for flattening
                concs_rs = map1(flatten, concs)

                concs_rs = vec_wm_f(concs_rs, rate_data, diff_data, dt)

                # unflattening
                concs = map1(unflatten, concs_rs)
                return concs

            print("reaction diffusion, unbatched")
            return rd_f