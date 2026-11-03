"""
storypy.compute.leave_one_out
==============================

Leave-one-out (LOO) sensitivity analysis for the storypy spatial regression.

For each model in the ensemble, the regression is re-run with that model
excluded. The resulting set of coefficient maps is used to assess how
sensitive the regression is to any single model.

Summary statistics across LOO iterations are computed for each regression
coefficient:

- **mean**   - average coefficient across all LOO iterations
- **std**    - standard deviation across iterations (spread = sensitivity)
- **range**  - max − min across iterations
- **cv**     - coefficient of variation (std / |mean|); high cv → high sensitivity

Outputs are saved to::

    {work_dir}/regression_output/{var}/loo/
    ├── loo_coefficients.nc        # full (n_models, lat, lon) stack per driver
    ├── loo_mean.nc                # mean coefficient across LOO iterations
    ├── loo_std.nc                 # std across LOO iterations
    ├── loo_range.nc               # range (max - min) across LOO iterations
    └── loo_cv.nc                  # coefficient of variation (std / |mean|)

Example
-------
>>> from storypy.compute.leave_one_out import run_loo_sensitivity
>>> results = run_loo_sensitivity(
...     work_dir        = './output',
...     target_variable = 'pr',
... )
>>> # results['std'] is an xr.Dataset with one variable per driver
>>> print(results['std']['MEM'].max().values)
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import statsmodels.api as sm
from tqdm import tqdm

from storypy.compute._regres import SpatialRegression


# ---------------------------------------------------------------------------
# Core LOO function
# ---------------------------------------------------------------------------

def run_loo_sensitivity(
    work_dir:        str,
    target_variable: str = 'pr',
    driver_csv:      str = None,
    drivers:         list = None,
    save:            bool = True,
) -> dict:
    """
    Run leave-one-out sensitivity analysis on the spatial regression.

    For each model in the ensemble the regression is re-run with that model
    excluded, producing one set of coefficient maps per LOO iteration. Summary
    statistics across all iterations are computed and optionally saved.

    Parameters
    ----------
    work_dir : str
        Directory containing ``target_<var>.nc`` and the driver CSV.
        Output is written to ``{work_dir}/regression_output/{var}/loo/``.
    target_variable : str
        Target variable name, e.g. ``'pr'`` or ``'ua'``. Default ``'pr'``.
    driver_csv : str, optional
        Path to the scaled standardized driver CSV. If ``None``, defaults to::

            {work_dir}/storyline_analysis/multiple_regression/remote_drivers/
                scaled_standardized_drivers.csv
    save : bool
        Whether to save outputs to NetCDF. Default ``True``.

    Returns
    -------
    dict with keys:
        - ``'coefficients'``: ``xr.Dataset`` - full LOO stack,
          dims ``(left_out_model, lat, lon)`` per driver variable
        - ``'mean'``:  ``xr.Dataset`` - mean coefficient across LOO iterations
        - ``'std'``:   ``xr.Dataset`` - std across LOO iterations
        - ``'range'``: ``xr.Dataset`` - max − min across LOO iterations
        - ``'cv'``:    ``xr.Dataset`` - coefficient of variation (std / |mean|)
        - ``'models'``: list of str - model names in LOO order

    Raises
    ------
    FileNotFoundError
        If ``target_<var>.nc`` or the driver CSV is not found.
    ValueError
        If fewer than 3 common models are found between the target NetCDF
        and the driver CSV.

    Notes
    -----
    The regression is run via :class:`storypy.compute._regres.SpatialRegression`
    using the same NaN handling as the full pipeline.
    Each LOO iteration runs on ``n_models - 1`` models.
    """
    var = target_variable

    # --- Load target ---
    target_path = os.path.join(work_dir, f'target_{var}.nc')
    if not os.path.exists(target_path):
        raise FileNotFoundError(
            f'Target file not found: {target_path}. '
            'Run ESMValProcessor.process_var() or ModelDataPreprocessor first.'
        )
    target_ds = xr.open_dataset(target_path)
    target_da = target_ds[var]   # dims: (model, lat, lon)

    # --- Load drivers ---
    if driver_csv is None:
        driver_csv = os.path.join(
            work_dir,
            'storyline_analysis', 'multiple_regression', 'remote_drivers',
            'scaled_standardized_drivers.csv',
        )
    if not os.path.exists(driver_csv):
        raise FileNotFoundError(f'Driver CSV not found: {driver_csv}')

    regressors = pd.read_csv(driver_csv, index_col=0)
    regressors.index = regressors.index.str.strip()

    # --- Driver selection ---
    if drivers is not None:
        if isinstance(drivers, str):
            drivers = [drivers]
        missing = [d for d in drivers if d not in regressors.columns]
        if missing:
            raise ValueError(
                f'The following drivers were requested but not found in the '
                f'driver CSV: {missing}. '
                f'Available drivers: {regressors.columns.tolist()}'
            )
        regressors = regressors[drivers]
        print(f'Using {len(drivers)} selected drivers: {drivers}')
    else:
        print(f'Using all {len(regressors.columns)} drivers: '
              f'{regressors.columns.tolist()}')

    # --- Align models ---
    target_models  = target_da['model'].values.tolist()
    common_models  = [m for m in target_models if m in regressors.index]

    if len(common_models) < 3:
        raise ValueError(
            f'Only {len(common_models)} common models found. '
            'Need at least 3 for LOO analysis.'
        )

    target_aligned    = target_da.sel(model=common_models)
    regressors_aligned = regressors.loc[common_models]
    regressor_names   = regressors_aligned.columns.insert(0, 'MEM')

    n_models = len(common_models)
    print(
        f'LOO sensitivity analysis: {var}, {n_models} models, '
        f'{len(regressor_names)} regressors (incl. MEM)'
    )

    # --- LOO loop ---
    loo_coef_list = []   # one Dataset per iteration

    for i, left_out in enumerate(tqdm(common_models, desc='LOO iterations')):

        # Drop one model
        remaining_models  = [m for m in common_models if m != left_out]
        target_loo        = target_aligned.sel(model=remaining_models)
        regressors_loo    = regressors_aligned.loc[remaining_models]

        # Run regression
        SR = SpatialRegression()
        SR.regression_data(target_loo, regressors_loo, regressor_names)

        # target_clean = xr.apply_ufunc(replace_nans_with_zero, target_loo)
        results = xr.apply_ufunc(
            SR.linear_regression,
            target_loo,
            input_core_dims=[['model']],
            output_core_dims=[[] for _ in range(SR.rd_num)],
            vectorize=True,
            dask='parallelized',
        )

        # Assemble into Dataset
        coef_ds = _assemble_coef_dataset(results, regressor_names, var)
        coef_ds = coef_ds.expand_dims({'left_out_model': [left_out]})
        loo_coef_list.append(coef_ds)

        print(f'  [{i+1}/{n_models}] Left out: {left_out}')

    # --- Stack all iterations ---
    loo_stack = xr.concat(loo_coef_list, dim='left_out_model')

    # --- Summary statistics ---
    loo_mean  = loo_stack.mean(dim='left_out_model')
    loo_std   = loo_stack.std(dim='left_out_model')
    loo_range = loo_stack.max(dim='left_out_model') - loo_stack.min(dim='left_out_model')

    # CV = std / |mean|; mask where mean is near zero to avoid division noise
    loo_cv = xr.where(
        np.abs(loo_mean) > 1e-6,
        loo_std / np.abs(loo_mean),
        np.nan,
    )

    results_dict = {
        'coefficients': loo_stack,
        'mean':         loo_mean,
        'std':          loo_std,
        'range':        loo_range,
        'cv':           loo_cv,
        'models':       common_models,
    }

    # --- Save ---
    if save:
        out_dir = os.path.join(work_dir, 'regression_output', var, 'loo')
        os.makedirs(out_dir, exist_ok=True)
        _save_results(results_dict, out_dir)
        print(f'\nLOO outputs saved to: {out_dir}')

    return results_dict


# ---------------------------------------------------------------------------
# Helper: assemble coefficient Dataset from apply_ufunc output
# ---------------------------------------------------------------------------

def _assemble_coef_dataset(results, regressor_names, var: str) -> xr.Dataset:
    """
    Assemble the tuple of DataArrays returned by apply_ufunc into a Dataset
    with one variable per regressor, using the variable renaming logic from
    SpatialRegression.perform_regression.

    Parameters
    ----------
    results : tuple of xr.DataArray
        Output from apply_ufunc - one DataArray per regressor.
    regressor_names : sequence of str
        Regressor names in the same order as results.
    var : str
        Target variable name (used for renaming the first DataArray).

    Returns
    -------
    xr.Dataset
    """
    # First result comes out named after the variable - rename to MEM
    ds = results[0].to_dataset()
    rename_map = {
        'ua':  regressor_names[0],
        'sst': regressor_names[0],
        'tos': regressor_names[0],
        'tas': regressor_names[0],
        'pr':  regressor_names[0],
    }
    if var in rename_map:
        ds = ds.rename({var: rename_map[var]})
    else:
        ds = ds.rename({var: regressor_names[0]})

    # Add remaining regressors
    for i in range(1, len(regressor_names)):
        ds[regressor_names[i]] = results[i]

    return ds


# ---------------------------------------------------------------------------
# Helper: save results to NetCDF
# ---------------------------------------------------------------------------

def _save_results(results: dict, out_dir: str) -> None:
    """Save LOO summary statistics to NetCDF files."""
    results['coefficients'].to_netcdf(
        os.path.join(out_dir, 'loo_coefficients.nc')
    )
    results['mean'].to_netcdf(os.path.join(out_dir, 'loo_mean.nc'))
    results['std'].to_netcdf(os.path.join(out_dir, 'loo_std.nc'))
    results['range'].to_netcdf(os.path.join(out_dir, 'loo_range.nc'))
    results['cv'].to_netcdf(os.path.join(out_dir, 'loo_cv.nc'))


# ---------------------------------------------------------------------------
# Convenience: print a summary table
# ---------------------------------------------------------------------------

def loo_summary_table(results: dict) -> pd.DataFrame:
    """
    Print and return a summary table of spatial-mean LOO statistics
    for each regression coefficient.

    Parameters
    ----------
    results : dict
        Output from :func:`run_loo_sensitivity`.

    Returns
    -------
    pd.DataFrame
        One row per driver, columns: mean, std, range, cv (all spatial means).

    Example
    -------
    >>> from storypy.compute.leave_one_out import run_loo_sensitivity, loo_summary_table
    >>> results = run_loo_sensitivity('./output', 'pr')
    >>> df = loo_summary_table(results)
    >>> print(df)
    """
    rows = []
    for name in results['mean'].data_vars:
        rows.append({
            'driver':       name,
            'mean':         float(results['mean'][name].mean()),
            'std':          float(results['std'][name].mean()),
            'range':        float(results['range'][name].mean()),
            'cv':           float(results['cv'][name].mean()),
        })
    df = pd.DataFrame(rows).set_index('driver')
    print('\nLOO sensitivity summary (spatial means):')
    print(df.to_string(float_format='{:.4f}'.format))
    return df


def load_loo_results(work_dir: str, target_variable: str = 'pr') -> dict:
    """
    Load saved LOO results from disk and return the results dict
    expected by plot_loo functions.

    Parameters
    ----------
    work_dir : str
        Same work_dir used in run_loo_sensitivity.
    target_variable : str
        Target variable name. Default 'pr'.

    Returns
    -------
    dict with keys: coefficients, mean, std, range, cv, models
    """
    loo_dir = os.path.join(
        work_dir, 'regression_output', target_variable, 'loo'
    )
    if not os.path.exists(loo_dir):
        raise FileNotFoundError(
            f'LOO output directory not found: {loo_dir}. '
            'Run run_loo_sensitivity() first.'
        )
    results = {}
    for key in ('coefficients', 'mean', 'std', 'range', 'cv'):
        path = os.path.join(loo_dir, f'loo_{key}.nc')
        if not os.path.exists(path):
            raise FileNotFoundError(f'Missing LOO file: {path}')
        results[key] = xr.open_dataset(path)

    results['models'] = (
        results['coefficients']['left_out_model'].values.tolist()
    )
    return results

# ---------------------------------------------------------------------------
# Mask and diagnostic computation
# ---------------------------------------------------------------------------
def loo_sensitivity_masks(
    results,
    coef_full,
    drivers=None,
    sign_threshold=1.0,
    leverage_threshold=0.5,
    spread_threshold=0.2,
    spread_abs=None,
    amp_frac=0.1,
    amp_percentile=90,
):
    """
    Compute per-gridpoint jackknife sensitivity diagnostics and masks.
 
    Parameters
    ----------
    results : dict
        Output of :func:`run_loo_sensitivity`. Must contain ``'coefficients'``
        with dimension ``left_out_model``.
    coef_full : xr.Dataset
        Full-ensemble regression coefficients, i.e. the contents of
        ``regression_coefficients.nc``. Variable names must match those in
        ``results['coefficients']``.
    drivers : list of str, optional
        Subset of coefficient names to process. Default: all shared variables.
    sign_threshold : float, default 1.0
        Grid points with ``sign_consistency < sign_threshold`` are flagged in
        ``mask_sign``. The default flags any grid point where *at least one*
        iteration flips the sign. Use e.g. 0.95 for a more permissive mask.
    leverage_threshold : float, default 0.5
        Grid points with ``leverage > leverage_threshold`` are flagged in
        ``mask_leverage`` (0.5 = one model moves the coefficient by >50%).
    spread_threshold : float, default 0.2
        Grid points where ``SE_jack`` exceeds ``spread_threshold`` times the
        domain amplitude of ``|beta_full|`` are flagged in ``mask_spread``.
        Unlike the other two masks this one is normalised by a *domain-wide*
        amplitude rather than by the local coefficient, so it does not
        concentrate along the zero contour. The default is a starting point
        only - calibrate it against your own fields (print
        ``median(SE_jack) / p90(|beta|)`` per driver and pick a value that
        separates the thermodynamic from the dynamical coefficients).
    spread_abs : float, optional
        Absolute threshold in coefficient units, overriding
        ``spread_threshold``. The most readable choice is one contour interval
        of the map, so the stippling says "model choice moves this coefficient
        by more than one colour step".
    amp_frac, amp_percentile : float
        Near-zero guard. ``leverage`` and ``mask_leverage`` are only defined
        where ``|beta_full|`` exceeds ``amp_frac`` times the
        ``amp_percentile``-th percentile of ``|beta_full|`` over the domain.
        Without this, leverage explodes wherever the coefficient passes through
        zero and the stippling becomes an artefact of the zero contour.
 
    Returns
    -------
    xr.Dataset
        One set of variables per driver, suffixed
        ``_sign_consistency``, ``_leverage``, ``_jack_se``, ``_jack_snr``,
        ``_mask_sign``, ``_mask_leverage``, ``_mask_spread``. Masks are
        boolean.
        ``attrs['n_models']`` records the ensemble size.
    """
    stack = results['coefficients']
    if 'left_out_model' not in stack.dims:
        raise ValueError(
            "results['coefficients'] has no 'left_out_model' dimension - "
            "pass the output of run_loo_sensitivity(), not loo_mean/loo_std."
        )
 
    shared = [v for v in stack.data_vars if v in coef_full.data_vars]
    if drivers is not None:
        shared = [v for v in drivers if v in shared]
    if not shared:
        raise ValueError(
            f"No shared coefficient names. LOO has {list(stack.data_vars)}; "
            f"full-ensemble file has {list(coef_full.data_vars)}."
        )
 
    n = stack.sizes['left_out_model']
    out = xr.Dataset(attrs={'n_models': n})
 
    for name in shared:
        loo  = stack[name]                      # (left_out_model, lat, lon)
        full = coef_full[name]                  # (lat, lon)
 
        # align grids defensively - LOO and full run may differ in lon convention
        if not np.array_equal(np.asarray(full['lon']), np.asarray(loo['lon'])):
            full = full.interp_like(loo.isel(left_out_model=0), method='nearest')
 
        # --- 1. sign consistency -------------------------------------------
        agree = xr.where(np.sign(loo) == np.sign(full), 1.0, 0.0)
        agree = agree.where(np.isfinite(loo) & np.isfinite(full))
        sign_consistency = agree.mean(dim='left_out_model', skipna=True)
 
        # --- 2. near-zero guard --------------------------------------------
        amp = float(np.nanpercentile(np.abs(full.values), amp_percentile))
        strong = np.abs(full) > (amp_frac * amp)
 
        # --- 3. leverage of the single most influential model ---------------
        dev = np.abs(loo - full).max(dim='left_out_model', skipna=True)
        leverage = (dev / np.abs(full)).where(strong)
 
        # --- 4. proper jackknife standard error and SNR --------------------
        s0 = loo.std(dim='left_out_model', skipna=True)      # ddof=0
        jack_se = np.sqrt(n - 1) * s0
        jack_snr = (np.abs(full) / jack_se).where(jack_se > 0)
 
        out[f'{name}_sign_consistency'] = sign_consistency
        out[f'{name}_leverage']         = leverage
        out[f'{name}_jack_se']          = jack_se
        out[f'{name}_jack_snr']         = jack_snr
        out[f'{name}_mask_sign']        = sign_consistency < sign_threshold
        out[f'{name}_mask_leverage']    = (
            (leverage > leverage_threshold).fillna(False) & strong
        )
        cut = spread_abs if spread_abs is not None else spread_threshold * amp
        out[f'{name}_mask_spread'] = (jack_se > cut).fillna(False)
 
        out[f'{name}_jack_se'].attrs['long_name'] = (
            f'jackknife standard error of {name} '
            f'(sqrt(n-1) x LOO spread, n={n})'
        )
        out[f'{name}_leverage'].attrs['long_name'] = (
            f'max single-model fractional influence on {name}'
        )
 
    return out