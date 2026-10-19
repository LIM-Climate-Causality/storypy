# storypy review: findings and changes

A review of storypy 0.1.7 in October 2026. Each finding below was reproduced by running the
original code. The fixes are deliberately small and stay inside the existing code structure.
[CHANGELOG.md](CHANGELOG.md) lists every change.

## Findings

### The package did not work as installed

- `import storypy.compute` and `import storypy.evaluate.plot` failed: they import
  `_leave_one_out`, `_jet_index` and `_plot_loo`, which were never committed.
- `storypy.compute` and `storypy.preprocess` imported ESMValTool at module level, so nothing
  worked outside an ESMValTool conda environment.
- `tests/test_data.py` ran a comparison against `/climca/...` paths when imported, so pytest
  stopped before running any test.

### Crashes

- Relative importance called `robjects`, which is never defined. A bare `except` turned that
  into zeros on older NumPy (the bundled `..._relative_importance.nc` files are all zeros) and a
  crash of `perform_regression`, and therefore `compute_regression`, on NumPy 2.
- Regression with a single driver crashed.
- `pd` and `regionmask` were used but not imported in `_diagnostics.py`.
- `clim_change` without a season crashed.
- `plot_regression_coef_map` used an undefined `im`; the hatching levels failed when every
  gridpoint was significant.

### Silent wrong results

- Seasons containing December but not January (e.g. OND) were split across two years.
- `ESMValProcessor` could attach a model's data to another model's name after one alias was
  skipped, and relabelled driver data with sorted names.
- `ModelDataPreprocessor` could write a previous driver's data into a driver file.

### Hard-coded values

- `EXCLUDE_VARS` could not be changed per run → now `config['exclude_vars']` /
  `config['extra_exclude_vars']`.
- `perform_regression` renamed outputs through `if var == 'ua' / 'sst' / 'tas' / 'pr'`, which
  failed for other names → names now come from the regressor names.
- Still fixed in code (documented on the Configuration page): `gw` / `tas` for global warming,
  `pr` × 86400, the `MEM` intercept name, plot years and periods.

### Documentation

The Read the Docs build had 13 warnings and errors. The piControl page was unreachable, the
plotting and storyline API sections were empty, examples used functions that do not exist,
and the γ formula did not match the code. All of this is fixed, and the docs now include
Quickstart, Configuration, Utilities, Contributing and Changelog pages.

## Open questions

1. **Bundled sample data is inconsistent.** `remote_drivers/scaled_standardized_drivers.csv`
   differs from what `drivers/zs17_drivers.nc` gives (by up to 2.7 standard deviations), and
   `regression_output/pr/*.nc` does not match a regression of any bundled target on the bundled
   drivers. The relative-importance files are all zeros. Regenerate them with the fixed code.
2. **`_leave_one_out`, `_jet_index`, `_plot_loo`**: commit them if they exist elsewhere, then
   add their imports back.
3. **Not fixed, low impact**: `ModelDataPreprocessor` stores `n_members = 1` for every model
   (`member_ids` is correct, and nothing reads `n_members`); `extract_metadata` reads the
   variable from the wrong file-name field; `replace_nans_with_zero` fills with a random number.
   None of these is used by the pipeline.
