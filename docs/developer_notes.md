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
| `changelog-bot.yml` | push to `main` | Cursor agent updates `CHANGELOG.md`; may push a follow-up bot commit |
| `release.yml` | push to `main` | If `__version__` was bumped: tests, then build, upload to PyPI, and create the matching `vX.Y.Z` tag + GitHub release |

## Changelog bot

On every push to `main`, the **Changelog bot** workflow
(`.github/workflows/changelog-bot.yml`) runs a Cursor agent that maintains
[CHANGELOG.md](changelog.md) in [Keep a Changelog](https://keepachangelog.com/)
format.

### What it does

**Ordinary merges** (no version bump):

- Reads the latest commit on `main`.
- Adds a concise bullet under **`[Unreleased]`** when the change is
  user-facing (skips CI-only or formatting-only work).
- Pushes a follow-up commit authored by `github-actions[bot]`, for example:
  `docs(changelog-bot): update changelog for abc1234 [skip changelog]`.

**Release merges** (`__version__` bumped in `icrn/__init__.py`):

- Promotes **`[Unreleased]`** to **`## [X.Y.Z] - YYYY-MM-DD`**.
- Opens a fresh empty **`[Unreleased]`** section at the top.
- Pushes a follow-up commit such as:
  `docs(changelog-bot): release changelog for vX.Y.Z [skip changelog]`.

The bot never rewrites your original merge commit. PyPI publishing and git
tagging still come from `release.yml` on the merge commit itself; the bot's
changelog commit may land one commit later on `main`.

### Setup

The workflow requires a repository secret:

| Secret | Purpose |
|--------|---------|
| `CURSOR_API_KEY` | User API key from the **API** section of the [Cursor dashboard](https://cursor.com/dashboard), or a [team service account](https://cursor.com/docs/account/enterprise/service-accounts) key (Enterprise) |

### Dry run

Test version detection and the agent prompt without calling Cursor or changing
files. From the repo root:

```bash
python .github/scripts/update_changelog.py --dry-run
# or
DRY_RUN=1 python .github/scripts/update_changelog.py
```

Dry run prints commit metadata, `is_release`, and the full prompt to the
terminal. It does not require `CURSOR_API_KEY` or `cursor-sdk`. To exercise
release mode, make a test commit that bumps `__version__` in
`icrn/__init__.py` relative to its parent commit.

Automated tests live in `.github/scripts/test_update_changelog.py`:

```bash
python .github/scripts/test_update_changelog.py -v
```

They run in CI alongside the main test suite.

### Skipping the bot

Include **`[skip changelog]`** anywhere in the **head commit message** of the
push to `main`. The workflow checks that tag and exits without running the
agent.

**Direct push:**

```bash
git commit -m "docs: fix typo in README [skip changelog]"
git push origin main
```

**Squash merge (GitHub UI):** add `[skip changelog]` to the squash commit
message before confirming the merge.

**Create a merge commit (GitHub UI):** edit the merge commit message on the
confirmation screen and add `[skip changelog]`. If you already merged and
forgot, the bot may still run on that push; use a follow-up commit with
`[skip changelog]` only when you need to land other work without triggering
the bot again.

**Rebase merge:** ensure at least one commit on the branch (typically the
last) includes `[skip changelog]` before merging, or edit the resulting merge
commit message if your merge strategy produces one.

Use this when:

- The change should not appear in the changelog (docs-only typos, CI tweaks,
  mechanical refactors you do not want summarized).
- You are editing `CHANGELOG.md` manually and do not want the agent to adjust
  it on that push.
- The workflow is misbehaving and you need a one-off merge without bot
  involvement.

Bot commits always include `[skip changelog]` in their own messages so they
do not trigger another bot run.

## Publishing to PyPI

Releases are continuously deployed from `main`. There is no manual tagging step.

1. Merge feature work to `main` as usual. The changelog bot accumulates
   user-facing notes under **`[Unreleased]`** (unless you skip it for a given
   push; see [Skipping the bot](#skipping-the-bot)).
2. When ready to release, open a PR that bumps `__version__` in
   `icrn/__init__.py` to the new [semantic version](https://semver.org/).
   You do not need to move changelog entries by hand—the bot performs the
   release ritual when that merge lands.
3. Merge the PR into `main`.
4. The `release.yml` workflow compares `__version__` against existing
   `vX.Y.Z` tags:
   - If the version is unchanged (tag already exists), it does nothing.
   - If the version is new, it runs the test suite, builds the
     distributions, publishes them to
     [PyPI](https://pypi.org/project/icrn/) via trusted publishing, and
     creates the `vX.Y.Z` git tag and a GitHub release with generated notes.

Because publishing only happens when the version changes, ordinary merges
that do not touch `__version__` never produce a release. To release, the only
required action is bumping `__version__` and merging to `main`.
