# Developer notes

Notes for contributors working on the `icrn` library itself. For installation
and usage examples, see [Getting started](getting_started.md).

## Development setup

Clone the repo and install in editable mode with test tooling:

```bash
git clone https://github.com/SwissChardLeaf/icrn.git
cd icrn
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Enable pre-commit hooks once:

```bash
pre-commit install
```

## Running tests

Tests use the standard library `unittest`.
The following can be run from the top-level:

```bash
# Full discovery (can be memory-heavy on some machines)
coverage run -m unittest discover -s icrn -t . -v

# Target a module or class
python -m unittest icrn.test_jax
python -m unittest icrn.test_solver.TestBoundaryCondition
```

## Linting, formatting, and type checking

Quality checks run through **pre-commit** (see `.pre-commit-config.yaml`).
Run all hooks manually:

```bash
pre-commit run --all-files
```

Configuration lives in `pyproject.toml`:

CI runs the same hooks via the **Linting and Tests** workflow
(`.github/workflows/tests_linting.yml`).

## Building documentation

Docs are built with **MkDocs Material**, **mkdocstrings**, and
**mkdocs-jupyter**. Source pages live in `docs/`; config is `mkdocs.yml` at
the repo root.

```bash
pip install -r docs/requirements.txt
pip install -e .
mkdocs serve          # local preview at http://127.0.0.1:8000
mkdocs build --strict # writes static site to site/ (gitignored)
```

To add a page, create a Markdown file under `docs/` and register it in the
`nav` section of `mkdocs.yml`. API docs are generated from NumPy-style
docstrings in Python source via `docs/api.md`.

Public functions and classes should use **NumPy-style docstrings** with
Markdown-friendly cross-links (e.g.
`` [`Species`][icrn.Species] ``). Templates with `<TODO>` markers remain in
some symbols; fill these in as the API stabilizes. mkdocstrings picks up
anything listed in `docs/api.md` that is exported from the package.

On push to `main`, the **Docs: build and deploy** workflow builds the site and
publishes to [GitHub Pages](https://swisschardleaf.github.io/icrn/).

## Continuous integration

| Workflow | When | What it does |
|----------|------|----------------|
| `tests.yml` | push/PR to `main` | `unittest discover` with coverage; uploads to Codecov |
| `tests_linting.yml` | push/PR | `pre-commit run --all-files` |
| `docs.yml` | push/PR to `main` | `mkdocs build --strict`; deploys to GitHub Pages on `main` |
| `publish.yml` | push tag `v*` | Tests, then build and upload to PyPI (trusted publishing) |
