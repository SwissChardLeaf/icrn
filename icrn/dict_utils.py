import yaml
import os
import jax.numpy as jnp
import jax.tree_util as jax_tree
#from jax.tree_util import register_pytree_node_class, tree_map, tree_flatten, register_pytree_with_keys

def _check_valid_binary_operator(s, o):
    pass

def map1(f, d):
    return jax_tree.tree_map(f, d)

def map2(f, d1, d2):
    # d2 is also a sjdict
    return jax_tree.tree_map(f, d1, d2)

def map2_scalar(f, d1, d2):
    # d2 is a scalar
    return jax_tree.tree_map(lambda x: f(x, d2), d1)
    
# def sjdict_allclose(d1, d2):
#     leaves, _ = tree_flatten(map2(jnp.allclose, d1, d2))
#     return jnp.all(jnp.array(leaves))

# def sjdict_allequal(d1, d2):
#     leaves, _ = tree_flatten(map2(jnp.array_equal, d1, d2))
#     return jnp.all(jnp.array(leaves))

def save_sjdict(d, save_path):
    for k, v in d.items():
        jnp_path = os.path.join(save_path, k + '.npy')
        jnp.save(jnp_path, v)

def load_sjdict(load_path):
    jnp_files = [item for item in os.listdir(load_path) if os.path.splitext(item)[1] == '.npy']
    jnpdict = dict()
    for jnp_file in jnp_files:
        jnpdict[os.path.splitext(jnp_file)[0]] = jnp.load(os.path.join(load_path, jnp_file))

    return jnpdict

def sjdict_builder(shapes_dict, concs_spec, rate_data_spec, diff_data_spec, batch_size=None, spatial_dim=None):
    concs = dict()
    rate_data = dict()
    diff_data = dict()
    
    for s, spec in concs_spec.items():
        init_shape = shapes_dict[s]
        
        if spatial_dim is not None:
            init_shape = spatial_dim + init_shape

        if batch_size is not None:
            init_shape = (batch_size,) + init_shape

        concs[s] = jnp.broadcast_to(jnp.array(spec), init_shape)

    for s, spec in rate_data_spec.items():
        rate_data[s] = jnp.broadcast_to(jnp.array(spec), shapes_dict[s])
    
    for s, spec in diff_data_spec.items():
        diff_data[s] = jnp.broadcast_to(jnp.array(spec), shapes_dict[s])

    return SJDict(concs), SJDict(rate_data), SJDict(diff_data)


@jax_tree.register_pytree_node_class
class SJDict(dict):
    def __add__(self, other):
        # return map2(jnp.add, self, other)
        return bin_op_helper(jnp.add, self, other)

    def __sub__(self, other):
        # return map2(jnp.subtract, self, other)
        return bin_op_helper(jnp.subtract, self, other)

    def __mul__(self, other):
        # return map2(jnp.multiply, self, other)
        return bin_op_helper(jnp.multiply, self, other)

    def __truediv__(self, other):
        # return map2(jnp.true_divide, self, other)
        return bin_op_helper(jnp.true_divide, self, other)

    def __pow__(self, other):
        # return map2(jnp.pow, self, other)
        return bin_op_helper(jnp.pow, self, other)

    def tree_flatten(self):
        return jax_tree.tree_flatten(dict(self))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(jax_tree.tree_unflatten(aux_data, children))

    def save(self, save_path):
        save_sjdict(self, save_path)

    @classmethod
    def load(cls, load_path):
        return load_sjdict(load_path)
    
def bin_op_helper(f, self, other):
    if isinstance(other, dict):
        res_dict = {k : f(v, other.get(k, 0)) for k, v in self.items()}
        return SJDict(res_dict)
    else:
        res_dict = {k : f(v, other) for k, v in self.items()}
        return SJDict(res_dict)

def save_dict_yaml(d, save_path):
    with open(save_path, "w") as file:
        yaml.dump(d, file)

def load_dict_yaml(load_path):
    with open(load_path, "r") as file:
        return yaml.safe_load(file)