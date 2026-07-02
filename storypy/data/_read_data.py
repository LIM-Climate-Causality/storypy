"""
storypy.data._read_data
=======================

Utilities for loading sample datasets bundled with storypy.

All data files are stored under ``storypy/data/`` and accessed via
:mod:`importlib.resources` (Python 3.9+).

Directory layout
----------------
::

    storypy/data/
    ├── drivers/                        # Raw driver NetCDF files per study
    │   ├── zs17_drivers.nc
    │   └── monerie23_drivers.nc
    ├── regression_output/              # Pre-computed regression results
    │   ├── pr/
    │   │   ├── regression_coefficients.nc
    │   │   ├── regression_coefficients_pvalues.nc
    │   │   ├── regression_coefficients_relative_importance.nc
    │   │   └── R2.nc
    │   └── ua/
    │       ├── regression_coefficients.nc
    │       ├── regression_coefficients_pvalues.nc
    │       ├── regression_coefficients_relative_importance.nc
    │       └── R2.nc
    ├── remote_drivers/                 # Scaled/standardized driver CSVs
    │   ├── drivers.csv
    │   ├── scaled_drivers.csv
    │   └── scaled_standardized_drivers.csv
    └── targets/                        # Pre-computed target fields per study/season
        ├── zs17_target_pr_NDJFM.nc
        ├── zs17_target_u850_NDJFM.nc
        ├── mindlin20_target_pr_DJF.nc
        ├── mindlin20_target_pr_JJA.nc
        └── monerie23_target_pr_JAS.nc
"""

import xarray as xr
import pandas as pd
from importlib.resources import files


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _data_path(relative_path: str):
    """
    Resolve a path relative to the storypy/data directory.

    Parameters
    ----------
    relative_path : str
        Path relative to ``storypy/data/``, e.g.
        ``'regression_output/pr/regression_coefficients.nc'``.

    Returns
    -------
    pathlib.Path
    """
    return files('storypy').joinpath('data') / relative_path


def _open_nc(relative_path: str, lat_slice=(-88, 88)) -> xr.Dataset:
    """
    Open a NetCDF file from the storypy data directory.

    Parameters
    ----------
    relative_path : str
        Path relative to ``storypy/data/``.
    lat_slice : tuple of float, optional
        Latitude bounds to subset on load. Default ``(-88, 88)``.

    Returns
    -------
    xarray.Dataset
    """
    path = _data_path(relative_path)
    ds = xr.open_dataset(path)
    if lat_slice is not None:
        ds = ds.sel(lat=slice(*lat_slice))
    return ds


def _open_csv(relative_path: str) -> pd.DataFrame:
    """
    Open a CSV file from the storypy data directory.

    Parameters
    ----------
    relative_path : str
        Path relative to ``storypy/data/``.

    Returns
    -------
    pandas.DataFrame
    """
    path = _data_path(relative_path)
    return pd.read_csv(path, index_col=0)


# ---------------------------------------------------------------------------
# Regression outputs  (variable + diagnostic)
# ---------------------------------------------------------------------------

#: Valid regression diagnostic names
REGRESSION_DIAGNOSTICS = (
    'regression_coefficients',
    'regression_coefficients_pvalues',
    'regression_coefficients_relative_importance',
    'R2',
)


def read_regression(variable: str, diagnostic: str = 'regression_coefficients') -> xr.Dataset:
    """
    Load a pre-computed regression output for a given variable.

    Parameters
    ----------
    variable : str
        Target variable name, e.g. ``'pr'`` or ``'ua'``.
    diagnostic : str, optional
        Which regression output to load. One of:

        - ``'regression_coefficients'`` (default)
        - ``'regression_coefficients_pvalues'``
        - ``'regression_coefficients_relative_importance'``
        - ``'R2'``

    Returns
    -------
    xarray.Dataset

    Examples
    --------
    >>> from storypy.data import read_regression
    >>> ds = read_regression('pr')
    >>> ds = read_regression('ua', diagnostic='R2')
    """
    if diagnostic not in REGRESSION_DIAGNOSTICS:
        raise ValueError(
            f"Unknown diagnostic '{diagnostic}'. "
            f"Choose from: {REGRESSION_DIAGNOSTICS}"
        )
    return _open_nc(f'regression_output/{variable}/{diagnostic}.nc')


# ---------------------------------------------------------------------------
# Target fields
# ---------------------------------------------------------------------------

def load_change_field(study: str, variable: str, season: str) -> xr.Dataset:
    """
    Load a pre-computed target precipitation or wind change field.

    Parameters
    ----------
    study : str
        Study identifier. Available options:

        - ``'zs17'`` — Zappa & Shepherd (2017)
        - ``'mindlin20'`` — Mindlin et al. (2020)
        - ``'monerie23'`` — Monerie et al. (2023)

    variable : str
        Variable name, e.g. ``'pr'``, ``'u850'``.
    season : str
        Season string, e.g. ``'NDJFM'``, ``'DJF'``, ``'JJA'``, ``'JAS'``.

    Returns
    -------
    xarray.Dataset

    Examples
    --------
    >>> from storypy.data import load_change_field
    >>> ds = load_change_field('zs17', 'pr', 'NDJFM')
    >>> ds = load_change_field('mindlin20', 'pr', 'DJF')
    """
    filename = f'{study}_target_{variable}_{season}.nc'
    return _open_nc(f'targets/{filename}')


def list_targets() -> list:
    """
    List all available target files bundled with storypy.

    Returns
    -------
    list of str
        Filenames available under ``storypy/data/targets/``.

    Examples
    --------
    >>> from storypy.data import list_targets
    >>> list_targets()
    ['mindlin20_target_pr_DJF.nc', 'mindlin20_target_pr_JJA.nc', ...]
    """
    target_dir = _data_path('targets')
    return sorted(p.name for p in target_dir.iterdir() if p.suffix == '.nc')


# ---------------------------------------------------------------------------
# Driver NetCDF files
# ---------------------------------------------------------------------------

def load_driver_field(study: str) -> xr.Dataset:
    """
    Load the raw driver NetCDF file for a given study.

    Parameters
    ----------
    study : str
        Study identifier, e.g. ``'zs17'`` or ``'monerie23'``.

    Returns
    -------
    xarray.Dataset

    Examples
    --------
    >>> from storypy.data import load_driver_field
    >>> ds = load_driver_field('zs17')
    """
    return _open_nc(f'drivers/{study}_drivers.nc', lat_slice=None)


# ---------------------------------------------------------------------------
# Remote driver CSV files
# ---------------------------------------------------------------------------

def read_drivers() -> pd.DataFrame:
    """
    Load the raw (unscaled) remote driver indices.

    Returns
    -------
    pandas.DataFrame
        Index: model names. Columns: driver names.

    Examples
    --------
    >>> from storypy.data import read_drivers
    >>> df = read_drivers()
    """
    return _open_csv('remote_drivers/drivers.csv')


def read_scaled_drivers() -> pd.DataFrame:
    """
    Load the scaled remote driver indices (divided by global warming).

    Returns
    -------
    pandas.DataFrame

    Examples
    --------
    >>> from storypy.data import read_scaled_drivers
    >>> df = read_scaled_drivers()
    """
    return _open_csv('remote_drivers/scaled_drivers.csv')


def read_scaled_standardized_drivers() -> pd.DataFrame:
    """
    Load the scaled and standardized remote driver indices.

    These are the drivers used directly as regressors in
    :func:`storypy.compute._mlr.compute_regression`.

    Returns
    -------
    pandas.DataFrame

    Examples
    --------
    >>> from storypy.data import read_scaled_standardized_drivers
    >>> df = read_scaled_standardized_drivers()
    """
    return _open_csv('remote_drivers/scaled_standardized_drivers.csv')