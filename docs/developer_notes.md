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

After a full run, print a summary or open an HTML report locally:

```bash
coverage report -m
coverage html   # open htmlcov/index.html in a browser
```

On GitHub Actions, the **Tests** workflow uploads `coverage.xml` and `htmlcov/`
as a **coverage** artifact on each run (download from the workflow summary).

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
| `tests.yml` | push/PR to `main` | `unittest discover` with coverage; report in logs + artifacts |
| `tests_linting.yml` | push/PR | `pre-commit run --all-files` |
| `docs.yml` | push/PR to `main` | `mkdocs build --strict`; deploys to GitHub Pages on `main` |
| `release.yml` | push to `main` | If `__version__` was bumped: tests, then build, upload to PyPI, and create the matching `vX.Y.Z` tag + GitHub release |

## Publishing to PyPI

Releases are continuously deployed from `main`. There is no manual tagging step.

1. In your PR, bump `__version__` in `icrn/__init__.py` to the new
   [semantic version](https://semver.org/) and move the relevant
   `CHANGELOG.md` entries from **Unreleased** into a section for the new
   version.
2. Merge the PR into `main`.
3. The `release.yml` workflow compares `__version__` against existing
   `vX.Y.Z` tags:
   - If the version is unchanged (tag already exists), it does nothing.
   - If the version is new, it runs the test suite, builds the
     distributions, publishes them to
     [PyPI](https://pypi.org/project/icrn/) via trusted publishing, and
     creates the `vX.Y.Z` git tag and a GitHub release with generated notes.

Because publishing only happens when the version changes, ordinary merges
that do not touch `__version__` never produce a release. To release, the only
required action is bumping `__version__` and merging to `main`.
