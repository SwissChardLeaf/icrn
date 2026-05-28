# icrn

**Indexed Chemical Reaction Networks in a differentiable, tensor-based framework.**

`icrn` is a JAX library for specifying chemical reaction networks with **indexed
species and rate constants**, and simulating them as either well-mixed ODEs or
reaction–diffusion PDEs. Because everything compiles to `jax.numpy` operations,
simulations are JIT-able with `jax.jit`, batchable with `jax.vmap`, and
differentiable end-to-end with `jax.grad` — so rate constants and initial
conditions can be **trained**.

!!! note
    Research code; interfaces are subject to change.

## Where to start

- **[Getting started](getting_started.md)** — installation and quickstart
  examples for well-mixed, indexed, and reaction-diffusion systems, plus a
  taste of differentiable simulation.
- **[API reference](api.md)** — the full public surface of `icrn`.

## Install

```bash
pip install icrn
```

For GPU/TPU acceleration, follow the [JAX install
guide](https://docs.jax.dev/en/latest/installation.html); the default wheel is
CPU-only.

## Cite

If you use `icrn` in academic work, please cite the DNA 31 paper:

> Inhoo Lee, Salvador Buse, and Erik Winfree. **Differentiable Programming of
> Indexed Chemical Reaction Networks and Reaction-Diffusion Systems.** In *31st
> International Conference on DNA Computing and Molecular Programming (DNA
> 31)*, LIPIcs Vol. 347, pp. 4:1–4:23. Schloss Dagstuhl, 2025.
> <https://doi.org/10.4230/LIPIcs.DNA.31.4>
