# Getting started

## Installation

```bash
pip install icrn
```

`icrn` depends on JAX. For GPU or TPU acceleration, follow the
[JAX install guide](https://docs.jax.dev/en/latest/installation.html) for the
right wheel for your platform; the default `pip install icrn` will pull a
CPU-only JAX.

Python 3.12 or newer is required.

## Highlights

- **Indexed reactions**: write `M[i] + M[j] -> D[i, j]` and have it compile to a single `jnp.einsum`.
- **Well-mixed and reaction–diffusion** in one API (`solve_well_mixed`, `solve_reaction_diffusion`).
- **Lie–Trotter / Strang** operator splitting with spectral diffusion.
- **Fast reactions** for limiting-reagent-style annihilation (`FastReaction`).
- **JAX-native**: works under `jit`, `vmap`, and `grad`.
- Optional non-negativity guards (`mode="strict"` via `checkify`, or `mode="relu"`).

## Quick Start

### Well Mixed Exponential Decay

```python
import jax.numpy as jnp
from icrn import many_species, many_rate_constants, MassActionReaction, solve_well_mixed

A = many_species("A")
k = many_rate_constants("k")

rxns = [MassActionReaction(A, 0, k)]  # A -> 0 with rate k

result = solve_well_mixed(
    rxns,
    conc_vals={A: jnp.array(1.0)},
    rate_constant_vals={k: jnp.array(1.0)},
    times=jnp.array([0.0, jnp.log(2), 1.0]),
    dt=0.005,
)
print(result[A])  # ~ [1.0, 0.5, 0.3679]
```

### Indexed Reactions

The thing that makes `icrn` different: species and rate constants can carry indices, and reactions involving them are compiled to a single tensor contraction.

```python
import jax.numpy as jnp
from icrn import (
    many_species, many_rate_constants, many_index_symbols,
    MassActionReaction, solve_well_mixed,
)

n = 10
M, D = many_species("M, D")
K1, K2 = many_rate_constants("K_1, K_2")
i, j = many_index_symbols("i, j", n)

# Reversible all-vs-all dimerization
rxns = [
    MassActionReaction(M[i] + M[j], D[i, j], K1[i, j]),
    MassActionReaction(D[i, j], M[i] + M[j], K2[i, j]),
]

conc = {M: jnp.ones(n), D: jnp.zeros((n, n))}
rates = {K1: jnp.ones((n, n)) * 0.1, K2: jnp.ones((n, n)) * 0.05}

out = solve_well_mixed(rxns, conc, rates, times=jnp.array([1.0]), dt=1e-3)
```

### Reaction–diffusion

A Gray–Scott-style call (omitting the initial-condition loading for brevity):

```python
from icrn import solve_reaction_diffusion

U, V = many_species("U, V")
F, k = many_rate_constants("F, k")

rxns = [
    MassActionReaction(U + 2 * V, 3 * V, 1),
    MassActionReaction(V, 0, F + k),
    MassActionReaction(0, U, F),
    MassActionReaction(U, 0, F),
]

sim = solve_reaction_diffusion(
    rxns,
    conc_vals={U: U0, V: V0},                       # 2-D fields
    rate_constant_vals={F: jnp.array(0.037), k: jnp.array(0.06)},
    diffusion_constant_vals={U: jnp.array(0.2), V: jnp.array(0.1)},
    times=jnp.array([5e3]),
    dt=1.0,
    spatial_dims=(101, 101),
    dspaces=(1.0, 1.0),
    mode="relu",
)
```

See the full Gray–Scott, Hopfield-Turing, and winner-take-all tests in `icrn/test_solver.py` for runnable end-to-end examples.

## Differentiable simulation

Because everything is JAX, you can `jit`, `vmap`, and `grad` straight through a solve:

```python
import jax

@jax.jit
def loss(rate_vals):
    out = solve_well_mixed(rxns, conc, {k: rate_vals}, times, dt)
    return jnp.mean((out[A][-1] - target) ** 2)

grad_fn = jax.grad(loss)
```

This is the foundation for training rate constants or initial conditions against a target behavior.

## Where to next

- [Tutorial notebooks (Colab)](https://github.com/SwissChardLeaf/icrn#tutorials)
- The full [API reference](api.md) is in the sidebar.
