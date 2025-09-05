import yaml
import os
import jax.numpy as jnp
import jax.tree_util as jax_tree
from .representation import BaseObject, ConcreteObject, Species, RateConstant
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
    
def sjdict_allclose(d1, d2):
    bool_dict = jax_tree.tree_map(lambda x, y : jnp.allclose(x, y), d1, d2)
    return all(bool_dict.dict.values())

def sjdict_allequal(d1, d2):
    bool_dict = jax_tree.tree_map(lambda x, y : jnp.all(x == y), d1, d2)
    return all(bool_dict.dict.values())
def save_sjdict(d, save_path):
    for k, v in d.items():
        jnp_path = os.path.join(save_path, str(k) + '.npy')
        jnp.save(jnp_path, v)

def load_sjdict(load_path, cls):
    jnp_files = [item for item in os.listdir(load_path) if os.path.splitext(item)[1] == '.npy']
    sjdict = SJDict()
    for jnp_file in jnp_files:
        file_name = os.path.splitext(jnp_file)[0]
        key_obj = cls(file_name)
        sjdict[key_obj] = jnp.load(os.path.join(load_path, jnp_file))

    return sjdict

def bin_op_helper(f, self, other):
    if isinstance(other, SJDict):
        res_dict = {k : f(v, other.get(k, 0)) for k, v in self.items()}
        return SJDict(res_dict)
    else:
        res_dict = {k : f(v, other) for k, v in self.items()}
        return SJDict(res_dict)
    
def sjdict_builder(shapes_dict, concs_spec, rate_constant_spec, diff_spec, batch_size=None, spatial_dim=None, spatial_rate_constant=False, dtype="float32"):
    concs_data = dict()
    rate_constant_data = dict()
    diff_data = dict()
    
    for base, shape in shapes_dict.items():
        if isinstance(base, Species):
            init_shape = shape
            
            if spatial_dim is not None:
                init_shape = spatial_dim + init_shape
                diff_data[base] = jnp.broadcast_to(jnp.array(diff_spec.get(base, 0)), shape).astype(dtype)


            if batch_size is not None:
                init_shape = (batch_size,) + init_shape

            concs_data[base] = jnp.broadcast_to(jnp.array(concs_spec.get(base, 0)), init_shape).astype(dtype)
        elif isinstance(base, RateConstant):
            init_shape = shape

            if spatial_rate_constant:
                init_shape = spatial_dim + init_shape

            if batch_size is not None:
                init_shape = (batch_size,) + init_shape

            rate_constant_data[base] = jnp.broadcast_to(jnp.array(rate_constant_spec.get(base, 0)), init_shape).astype(dtype)
        else:
            raise ValueError

    return SJDict(concs_data), SJDict(rate_constant_data), SJDict(diff_data)

@jax_tree.register_pytree_node_class
class SJDict(dict):
    def __init__(self, d=dict()):
        self._dict = d

    @property
    def dict(self):
        return self._dict
    
    def __add__(self, other):
        return bin_op_helper(jnp.add, self, other)
    
    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        return jax_tree.tree_map(lambda x : -x, self)

    def __sub__(self, other):
        return bin_op_helper(jnp.subtract, self, other)

    def __rsub__(self, other):
        return bin_op_helper(jnp.subtract, other, self)

    def __mul__(self, other):
        return bin_op_helper(jnp.multiply, self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return bin_op_helper(jnp.true_divide, self, other)

    def __rtruediv__(self, other):
        return bin_op_helper(jnp.true_divide, other, self)

    def __pow__(self, other):
        return bin_op_helper(jnp.power, self, other)
    
    def __rpow__(self, other):
        return bin_op_helper(jnp.power, other, self)
    
    def __or__(self, other):
        return SJDict(self.dict | other.dict)
    
    def __eq__(self, other):
        return sjdict_allequal(self, other)
    
    def __setitem__(self, key, value):
        if isinstance(key, BaseObject):
            self.dict[key] = value
        elif isinstance(key, ConcreteObject):
            self.dict[key.base] = self.dict[key.base].at[key.index_symbols].set(value)
        else:
            raise KeyError
    
    def __getitem__(self, key):
        if isinstance(key, BaseObject):
            return self.dict[key]
        elif isinstance(key, ConcreteObject):
            return self.dict[key.base][key.index_symbols]
        else:
            raise KeyError
    
    def __and__(self, other):
        return jax_tree.tree_map(lambda x, y : jnp.stack([x, y]), self, other)

    def add_with_dict(self, add_dict):
        for k, v in add_dict.items():
            if isinstance(k, BaseObject):
                if k in self._dict:
                    self.dict[k] = self.dict.get(k, 0) + v
            elif isinstance(k, ConcreteObject):
                if k.base in self._dict:
                    self.dict[k.base] = self.dict[k.base].at[k.index_symbols].add(v)
            else:
                raise TypeError
            
        return self
            
    def zeros(self):
        return jax_tree.tree_map(jnp.zeros_like, self)

    def init_with_dict(self, init_dict):
        return self.zeros().add_with_dict(init_dict)

    def tree_flatten(self):
        return jax_tree.tree_flatten(self.dict)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(jax_tree.tree_unflatten(aux_data, children))

    def save(self, save_path):
        save_sjdict(self, save_path)

    @classmethod
    def load(cls, load_path):
        return load_sjdict(load_path)
    
    def __str__(self):
        return repr(self)
    
    def __repr__(self):
        return repr(self.dict)
    
def bin_op_helper(f, a, b):
    if isinstance(a, SJDict) and isinstance(b, SJDict):
        return jax_tree.tree_map(f, a, b)
    elif isinstance(a, SJDict):
        try:
            return jax_tree.tree_map(lambda x : f(x, b), a)
        except:
            raise NotImplementedError
    elif isinstance(b, SJDict):
        try:
            return jax_tree.tree_map(lambda x : f(a, x), b)
        except:
            raise NotImplementedError
    else:
        raise NotImplementedError

def save_dict_yaml(d, save_path):
    with open(save_path, "w") as file:
        yaml.dump(d, file)

def load_dict_yaml(load_path):
    with open(load_path, "r") as file:
        return yaml.safe_load(file)