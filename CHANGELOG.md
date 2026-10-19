# Changelog

Notable changes to storypy. Format: [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Fixed
- `import storypy.compute` and `import storypy.evaluate.plot` failed: they imported
  `_leave_one_out`, `_jet_index` and `_plot_loo`, which were never committed.
- The package needed ESMValTool at import time. It is now only needed by `parse_config` and the
  ESMValTool diagnostic entry point; `storypy._compat.group_metadata` works with or without it.
- Relative importance called an undefined R object (`robjects`). The error was hidden by a bare
  `except`, so it returned zeros on older NumPy and crashed `perform_regression` on NumPy 2.
  It is now computed in Python (LMG metric, sums to R²).
- `perform_regression` crashed with a single driver.
- `_diagnostics.py` used `pd` and `regionmask` without importing them (`extract_metadata`,
  `apply_region_mask`, `region_method='mask'`).
- `clim_change(..., season=None)` crashed on `int('1960-01')`.
- `seasonal_data_months` moved December into the next year for any season containing it, which
  split e.g. OND over two years. Only seasons that wrap into January shift now.
- `ESMValProcessor`: a skipped alias shifted later models onto wrong names; all target variables
  were written to one file named after the last variable; driver data were relabelled with
  sorted model names that need not match the data order.
- `ModelDataPreprocessor`: a mis-indented block wrote driver files only when an ensemble spread
  existed, otherwise a stale dataset or nothing.
- `plot_regression_coef_map` raised `NameError` (`im` never assigned) and had no colour-bar axes
  for variables other than `ua`/`sst`; its significance hatching failed when every gridpoint
  was significant.
- `compute_regression` returned `[None]`; it now returns the paths it wrote.
- `__version__` reported 0.1.4; it is now read from the installed package metadata, so it
  always matches `pyproject.toml`.
- `tests/test_data.py` ran a comparison against `/climca/...` paths at import time, so the
  tests could not run on other machines.

### Changed
- `driver_indices` accepts `config['exclude_vars']` (replaces the default `EXCLUDE_VARS`) and
  `config['extra_exclude_vars']` (adds to it).
- `SpatialRegression.perform_regression` names outputs from the regressor names instead of an
  `if var == 'ua' / 'sst' / 'tas' / 'pr'` chain, so any target variable name works.
- Minimum dependency versions: numpy ≥ 1.24, pandas ≥ 2.0, xarray ≥ 2023.1,
  regionmask ≥ 0.11, scipy ≥ 1.10, matplotlib ≥ 3.7, cartopy ≥ 0.22.
- New optional dependency groups: `test`, `dev`, `docs`, `yaml`; Python 3.13 listed as supported.

### Documentation
- Read the Docs: fixed broken navigation (missing `picontrol_recipe` and `reference/utils`
  pages), empty plotting and storyline API sections, and examples using functions that do not
  exist (`run_regression`, `compute_drivers_from_netcdf`, `storypy.evaluate._plotting`).
- The γ criterion is documented with the formula the code computes,
  `sqrt(mean over models of beta_m^2 / sigma_m^2)`, instead of a median `|ΔP|/σ`.
- New pages: Quickstart (runs on the bundled sample data), Configuration, Utilities,
  Contributing and Changelog.
- `.readthedocs.yaml` installs `docs/requirements.txt` (adds `myst-parser`) and storypy itself.
- The piControl page moved from getting_started/picontrol.html to getting_started/picontrol_recipe.html, so old links to it no longer work.

### Added
- Test suite: smoke, unit, integration and end-to-end tests on synthetic data with known
  answers.
- `.pre-commit-config.yaml`, `CONTRIBUTING.md`, this changelog.

## [0.1.7]
- Distribution name `storypy` reclaimed on PyPI; publishing workflow for TestPyPI and PyPI.
