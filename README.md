# icrn

[![PyPI](https://img.shields.io/pypi/v/icrn.svg)](https://pypi.org/project/icrn/)
[![Python versions](https://img.shields.io/pypi/pyversions/icrn.svg)](https://pypi.org/project/icrn/)
[![License](https://img.shields.io/pypi/l/icrn.svg)](https://github.com/SwissChardLeaf/icrn/blob/main/LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.4230%2FLIPIcs.DNA.31.4-blue)](https://doi.org/10.4230/LIPIcs.DNA.31.4)
[![Tests](https://github.com/SwissChardLeaf/icrn/actions/workflows/tests.yml/badge.svg)](https://github.com/SwissChardLeaf/icrn/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/SwissChardLeaf/icrn/branch/main/graph/badge.svg)](https://codecov.io/gh/SwissChardLeaf/icrn)
[![Lint](https://github.com/SwissChardLeaf/icrn/actions/workflows/tests_linting.yml/badge.svg)](https://github.com/SwissChardLeaf/icrn/actions/workflows/tests_linting.yml)
[![Docs](https://github.com/SwissChardLeaf/icrn/actions/workflows/docs.yml/badge.svg)](https://github.com/SwissChardLeaf/icrn/actions/workflows/docs.yml)

**Indexed Chemical Reaction Networks in a differentiable, tensor-based framework.**

`icrn` is a JAX library for specifying chemical reaction networks with **indexed species and rate constants**, and simulating them as either well-mixed ODEs or reaction–diffusion PDEs. Because everything compiles to `jax.numpy` operations, simulations are JIT-able, batchable with `jax.vmap`, and differentiable end-to-end with `jax.grad` — so rate constants and initial conditions can be **trained**.

> interfaces are subject to change.

## Highlights

- **Indexed reactions**: write `M[i] + M[j] -> D[i, j]` and have it compile to a single `jnp.einsum`.
- **Well-mixed and reaction–diffusion** in one API (`solve_well_mixed`, `solve_reaction_diffusion`).
- **Lie–Trotter / Strang** operator splitting with spectral diffusion.
- **Fast reactions** for limiting-reagent-style annihilation (`FastReaction`).
- **JAX-native**: works under `jit`, `vmap`, and `grad`.
- Optional non-negativity guards (`mode="strict"` via `checkify`, or `mode="relu"`).

## Installation

```bash
pip install icrn
```

This installs **JAX** and **NumPy** as runtime dependencies. The default JAX
wheel is CPU-only; for GPU/TPU, install JAX for your platform after `icrn`
([JAX installation guide](https://docs.jax.dev/en/latest/installation.html)).

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

The Gray-Scott system.

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

## Tutorials

- [Getting started with icrn](https://colab.research.google.com/github/SwissChardLeaf/icrn/blob/main/docs/tutorials/getting_started.ipynb)
- [Reaction–diffusion with icrn](https://colab.research.google.com/drive/1DacnTGoGNVpuGdc-yRuHE2z47cNPHbHy?usp=share_link)
- [Training parameters with icrn](https://colab.research.google.com/drive/1KDA67p3XToZwTD6Soe9eHtf1nq36I0OH?usp=sharing)
- [Batching simulations](https://colab.research.google.com/drive/18QSaKDSVSacO036ComUYycWPDbwtwWZG?usp=sharing)
- [DNA31 examples](https://colab.research.google.com/drive/1ioMWwmCQ8ulOcybkvNppgLtMFgGeCscy)

## Project layout

```
icrn/
  symbols.py        # symbolic DSL: Species, IndexSymbol, Complex, TensorExpression
  reactions.py      # MassActionReaction, FastReaction
  solver.py         # solve_well_mixed, solve_reaction_diffusion, solve_with_ops
  operator.py       # operator construction for the splitting integrator
  _internal/        # private JAX kernels (mass-action, diffusion, time stepping)
  utils/            # small helpers (dict_utils)
docs/               # MkDocs site (Material + mkdocstrings)
test/               # reference data for end-to-end tests
```

Contributing: see [CONTRIBUTING.md](CONTRIBUTING.md). Release notes:
[CHANGELOG.md](CHANGELOG.md). PyPI releases: tag `vX.Y.Z` after bumping
`icrn/__init__.py` `__version__` (see
[Developer notes — Publishing to PyPI](docs/developer_notes.md#publishing-to-pypi)).

## Development

```bash
git clone https://github.com/SwissChardLeaf/icrn
cd icrn
pip install -e ".[dev]"

# tests
python -m unittest discover -s icrn -t .

# lint
ruff check .

# docs
pip install -r docs/requirements.txt
mkdocs serve            # live preview at http://127.0.0.1:8000
mkdocs build --strict   # one-shot build into ./site
```

## Status

- **Stable-ish**: symbolic DSL (`Species`, `IndexSymbol`, `Complex`), `MassActionReaction`, `solve_well_mixed`, `solve_reaction_diffusion` with spectral diffusion.
- **Experimental**: `FastReaction`, `mode="strict"` runtime checks, Strang splitting.
- **Planned**: dedicated training utilities, convolutional diffusion solver, benchmarks.

## License

MIT — see [`LICENSE`](LICENSE).

## Citation

If you use `icrn` in academic work, please cite the DNA 31 paper:

> Inhoo Lee, Salvador Buse, and Erik Winfree. **Differentiable Programming of Indexed Chemical Reaction Networks and Reaction-Diffusion Systems.** In *31st International Conference on DNA Computing and Molecular Programming (DNA 31)*. Leibniz International Proceedings in Informatics (LIPIcs), Volume 347, pp. 4:1–4:23. Schloss Dagstuhl – Leibniz-Zentrum für Informatik, 2025. <https://doi.org/10.4230/LIPIcs.DNA.31.4>

```bibtex
@InProceedings{lee_et_al:LIPIcs.DNA.31.4,
  author    = {Lee, Inhoo and Buse, Salvador and Winfree, Erik},
  title     = {{Differentiable Programming of Indexed Chemical Reaction Networks and Reaction-Diffusion Systems}},
  booktitle = {31st International Conference on DNA Computing and Molecular Programming (DNA 31)},
  pages     = {4:1--4:23},
  series    = {Leibniz International Proceedings in Informatics (LIPIcs)},
  ISBN      = {978-3-95977-399-7},
  ISSN      = {1868-8969},
  year      = {2025},
  volume    = {347},
  editor    = {Schaeffer, Josie and Zhang, Fei},
  publisher = {Schloss Dagstuhl -- Leibniz-Zentrum f{\"u}r Informatik},
  address   = {Dagstuhl, Germany},
  URL       = {https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.DNA.31.4},
  URN       = {urn:nbn:de:0030-drops-238534},
  doi       = {10.4230/LIPIcs.DNA.31.4},
  annote    = {Keywords: Differentiable Programming, Chemical Reaction Networks, Reaction-Diffusion Systems}
}
```
