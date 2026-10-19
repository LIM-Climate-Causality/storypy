# Contributing to storypy

## Setup

```bash
git clone git@github.com:LIM-Climate-Causality/storypy.git
cd storypy
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install pre-commit && pre-commit install
```

For the ESMValTool workflow (Option A), create the ESMValTool conda environment first and
install storypy into it. Everything else, including the tests, works without ESMValTool.

## Tests

The suite has four layers; each is a folder in `tests/` and a pytest marker.

| Layer | Folder | What it checks |
|---|---|---|
| smoke | `tests/smoke` | the package imports, the public functions exist, sample data is installed |
| unit | `tests/unit` | one function at a time on small synthetic data |
| integration | `tests/integration` | several components through real NetCDF/CSV files |
| e2e | `tests/e2e` | CMIP-style files → targets → drivers → regression → storylines → stippling → plots |

```bash
pytest                 # everything
pytest -m smoke        # one layer: smoke, unit, integration, e2e
pytest --cov=storypy   # with coverage
```

`tests/conftest.py` builds a small synthetic CMIP archive whose regression coefficients are
known, so the end-to-end test checks that the results are right, not only that files appear.

Rules of thumb:

- A bug fix comes with a test that fails without the fix.
- No machine-specific paths (`/climca/...`) in tests; use pytest's `tmp_path`.
- Code that needs ESMValTool imports it inside the function, or uses
  `storypy._compat.group_metadata`, so the package still imports without it.

## Documentation

```bash
pip install -r docs/requirements.txt
make -C docs html
open docs/build/html/index.html
```

Read the Docs builds from `.readthedocs.yaml`. Before pushing doc changes, check that
`sphinx-build -W --keep-going -b html docs/source docs/build/html` reports no warnings.

## Releasing

1. Update `version` in `pyproject.toml`.
2. Move the "Unreleased" entries in `CHANGELOG.md` under the new version.
3. Tag `vX.Y.Z` and push the tag; `.github/workflows/publish.yml` publishes to TestPyPI and,
   after approval, to PyPI.
