import xarray as xr
import numpy as np
import pandas as pd
import os
import glob

from esmvaltool.diag_scripts.shared import group_metadata, run_diagnostic

def stand_numpy(dato):
    """
    Standardize values (z-score) along the given array axis.

    Parameters
    ----------
    dato : ndarray
        Array of values to standardize.

    Returns
    -------
    ndarray
        Standardized array: ``(x - mean) / std`` computed with population std (ddof=0).

    Notes
    -----
    This helper is applied column-wise to data frames produced by
    :func:`compute_drivers_from_netcdf`.
    """
    anom = (dato - np.mean(dato)) / np.std(dato)
    return anom

def stand_pandas(series: pd.Series) -> pd.Series:
    """
    Standardize a pandas Series across models (z-score).

    Parameters
    ----------
    series : pandas.Series
        Values (typically one driver across models).

    Returns
    -------
    pandas.Series
        Standardized series. If the standard deviation is 0 or NaN,
        the function returns all-NaN (avoids divide-by-zero).
    """
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return (series - series.mean()) * np.nan
    return (series - series.mean()) / std

def compute_drivers_from_netcdf(driver_config):
    """
    Build driver regressors from preprocessed NetCDF files.

    This function scans ``driver_config['work_dir']`` for driver NetCDF
    files (one per driver), extracts model-wise scalar values, scales them
    by a global-warming index ``gw`` (read from ``driver_gw.nc``), and
    returns three data frames (raw, scaled, standardized). CSV files are
    also written to ``{work_dir}/remote_drivers/``.

    Parameters
    ----------
    driver_config : dict
        Configuration for reading drivers. Keys:
        - ``work_dir`` (str): directory with driver NetCDFs.
        - ``var_name`` (list[str]): variable names present in NetCDF files.
        - ``short_name`` (list[str], optional): output names used for
          regressors; defaults to ``var_name``.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame, pandas.DataFrame)
        Tuple ``(df_raw, df_scaled, df_standardized)`` with models as index
        and driver names as columns.

    Raises
    ------
    FileNotFoundError
        If no driver NetCDF files are found or the GW file is missing.
    ValueError
        If lengths of ``var_name`` and ``short_name`` mismatch or the GW
        dataset lacks required variables/coords.

    Notes
    -----
    * The global-warming file must be named ``driver_gw.nc`` and contain a
      variable ``gw`` with a ``model`` coordinate.
    * Standardization uses :func:`stand_numpy` column-wise.

    Examples
    --------
    >>> cfg = {"work_dir": "./out", "var_name": ["sst", "uas"], "short_name": ["SST", "UAS"]}
    >>> raw, scaled, std = compute_drivers_from_netcdf(cfg)
    >>> list(raw.columns)
    ['SST', 'UAS']
    """
    work_dir = driver_config['work_dir']
    var_names = driver_config['var_name']
    short_names = driver_config.get('short_name', var_names)

    if len(var_names) != len(short_names):
        raise ValueError("Length of 'var_name' and 'short_name' must match.")

    # Map short_name to var_name
    var_map = dict(zip(short_names, var_names))

    # Find all .nc files in the directory
    driver_files = glob.glob(os.path.join(work_dir, "*.nc"))
    if not driver_files:
        raise FileNotFoundError(f"No .nc files found in {work_dir}")

    # Load the global warming index file
    gw_file = os.path.join(work_dir, "driver_gw.nc")
    if not os.path.exists(gw_file):
        raise FileNotFoundError("Missing required global warming file: remote_driver_gw.nc")
    gw_ds = xr.open_dataset(gw_file)

    if 'gw' not in gw_ds:
        raise ValueError("Global warming dataset must contain variable 'gw'")

    gw_vals = gw_ds['gw']
    if 'model' not in gw_vals.coords:
        raise ValueError("'gw' variable must have 'model' coordinate")

    models = gw_vals['model'].values.tolist()

    # Initialize structure for models and regressor values
    model_set = set()
    regressor_values = {sn: {} for sn in short_names}
    regressor_values_scaled = {sn: {} for sn in short_names}

    available_model_sets = {sn: set() for sn in short_names}

    # Iterate over each NetCDF file
    for f in driver_files:
        ds = xr.open_dataset(f)

        if 'model' not in ds.coords:
            print(f"Skipping file without 'model' coord: {f}")
            continue

        # For each short_name, check if it's in the file and process
        for short_name in short_names:
            if short_name not in os.path.basename(f):
                continue  # Skip if the file does not match the variable

            if short_name not in ds:
                print(f"Variable {short_name} not found in {f}, skipping.")
                continue

            # Extract data for each model in the dataset
            for model in models:
                model = str(model)
                model_set.add(model)

                try:
                    # Extract the mean value of the variable for the model
                    val = ds[short_name].sel(model=model).mean().item()
                    
                    # Get the global warming value for the model
                    gw_val = gw_vals.sel(model=model).mean().item()
                    
                    # Scale the value by the global warming value, if not 0
                    scaled_val = val / gw_val if gw_val != 0 else np.nan

                except Exception as e:
                    print(f"Could not extract {short_name} for model {model} in {f}: {e}")
                    val = np.nan
                    scaled_val = np.nan

                # Store the raw and scaled values
                regressor_values[short_name][model] = val
                regressor_values_scaled[short_name][model] = scaled_val

    # Find intersection of model sets for all variables
    model_sets = [set(driver.keys()) for driver in regressor_values.values()]
    common_models = sorted(set.intersection(*model_sets))
    
    # Build a dataframe from the collected values
    # models = sorted(model_set)
    data_raw = {sn: [regressor_values[sn].get(m, np.nan) for m in common_models] for sn in short_names}
    data_scaled = {sn: [regressor_values_scaled[sn].get(m, np.nan) for m in common_models] for sn in short_names}

    df_raw = pd.DataFrame(data_raw, index=common_models)
    df_scaled = pd.DataFrame(data_scaled, index=common_models)

    # Standardize the scaled data (z-score normalization)
    df_standardized = df_scaled.apply(stand_numpy, axis=0)

    # Create the output directory and save the dataframes as CSV
    out_dir = os.path.join(work_dir, "remote_drivers")
    os.makedirs(out_dir, exist_ok=True)

    df_raw.to_csv(os.path.join(out_dir, "drivers.csv"))
    df_scaled.to_csv(os.path.join(out_dir, "scaled_drivers.csv"))
    df_standardized.to_csv(os.path.join(out_dir, "scaled_standardized_drivers.csv"))

    print(f"Saved driver regressors to: {out_dir}")
    return df_raw, df_scaled, df_standardized



EXCLUDE_VARS = {"u850", "tas", "sst", "psl", "pr", "pr_djf", "pr_jja", "u850_djf", "u850_jja"}


def _select_one_file_per_var(dataset_list, dataset_name, var_group):
    """
    Select one entry for a dataset/variable group.

    Preference is given to entries with ``alias == 'EnsembleMean'``.
    Falls back to the first matching entry if no ensemble mean is present.

    Parameters
    ----------
    dataset_list : list[dict]
        Metadata entries from ESMValTool.
    dataset_name : str
        Dataset identifier.
    var_group : str
        Variable group name.

    Returns
    -------
    dict or None
        The selected metadata record or ``None`` if not found.
    """
    for m in dataset_list:
        if (
            m.get("dataset") == dataset_name
            and m.get("variable_group") == var_group
            and m.get("alias") == "EnsembleMean"
        ):
            return m
    for m in dataset_list:
        if m.get("dataset") == dataset_name and m.get("variable_group") == var_group:
            return m
    return None


def _collect_scalar_drivers(meta, work_dir, exclude_vars=None):
    """
    Collect one scalar per model/driver from ESMValTool metadata.

    Reads files pointed to by ``meta`` (grouped by dataset), computes a
    time-mean (if necessary), and builds a list of dictionaries (one per
    model) containing driver values. Writes a convenience NetCDF with a
    ``model`` dimension to ``{work_dir}/remote_drivers/drivers.nc``.

    Parameters
    ----------
    meta : dict
        Metadata grouped by dataset (from :func:`esmvaltool.diag_scripts.shared.group_metadata`).
    work_dir : str
        Output directory used to persist the NetCDF.
    exclude_vars : set[str], optional
        Driver names to ignore. Defaults to :data:`EXCLUDE_VARS`.

    Returns
    -------
    (list[dict], list[str])
        Tuple ``(rd_list, models)`` where each dict maps driver name -> value.
    """
    rd_list = []
    models = []

    for dataset, dataset_list in meta.items():
        ts_dict = {}
        var_groups = sorted({m["variable_group"] for m in dataset_list})

        for var in var_groups:
            if var in EXCLUDE_VARS:
                continue
            sel = _select_one_file_per_var(dataset_list, dataset, var)
            if sel is None:
                continue
            da = xr.open_dataset(sel["filename"])[sel["short_name"]]
            if "time" in da.dims:
                value = da.mean(dim="time", skipna=True).item()
            else:
                value = float(da.item()) if da.size == 1 else float(da.mean().item())
            ts_dict[var] = value

        if "gw" in ts_dict:
            rd_list.append(ts_dict)
            models.append(dataset)
        else:
            print(f"There is no 'gw' for model {dataset}; skipping.")

    if rd_list:
        drivers_ds = _build_drivers_dataset(rd_list, models)
        out_dir = os.path.join(work_dir, "remote_drivers")
        os.makedirs(out_dir, exist_ok=True)
        drivers_nc = os.path.join(out_dir, "drivers.nc")
        drivers_ds.to_netcdf(drivers_nc)
        print(f"Saved drivers (one scalar per model/driver) to {drivers_nc}")
    else:
        print("No models with 'gw' found; NetCDF not written.")

    return rd_list, models


def _build_drivers_dataset(rd_list, models):
    """
    Assemble a dataset with one variable per driver and a ``model`` dimension.

    Parameters
    ----------
    rd_list : list[dict]
        One dict per model with driver -> value mappings.
    models : list[str]
        Model names aligned with ``rd_list``.

    Returns
    -------
    xarray.Dataset
        Dataset with variables named after drivers and a ``model`` coordinate.
    """
    all_vars = sorted(set().union(*[d.keys() for d in rd_list]))
    data_vars = {}
    for var in all_vars:
        vals = [d.get(var, np.nan) for d in rd_list]
        data_vars[var] = ("model", np.asarray(vals, dtype=np.float64))
    return xr.Dataset(data_vars, coords={"model": models})


def driver_indices(config):
    """
    Compute driver indices from ESMValTool metadata and write CSV files.

    Uses the ``input_data`` structure from an ESMValTool configuration
    to collect scalar driver values by dataset, scales them by ``gw``,
    standardizes across models, and writes:

    * ``scaled_standardized_drivers.csv``
    * ``drivers.csv`` (raw)
    * ``scaled_drivers.csv`` (scaled by ``gw``)

    Files are written under ``{work_dir}/remote_drivers/``.

    Parameters
    ----------
    config : dict
        ESMValTool run configuration. Must contain:
        - ``input_data``: mapping of variable metadata.
        - ``work_dir``: output directory.

    Returns
    -------
    None
        Results are written to disk.

    See Also
    --------
    compute_drivers_from_netcdf
        Alternative pathway that reads drivers directly from preprocessed NetCDF files.
    """
    meta = group_metadata(config["input_data"].values(), "dataset")

    rd_list, models = _collect_scalar_drivers(meta, config["work_dir"])
    if not rd_list:
        print("No drivers collected; aborting downstream steps.")
        return

    regressor_names = list(rd_list[0].keys())
    if "gw" in regressor_names:
        regressor_names.remove("gw")

    regressors_scaled = {}
    regressors = {}

    for rd in regressor_names:
        numer = np.array([rd_list[m_idx][rd] for m_idx, _ in enumerate(rd_list)], dtype=float)
        denom = np.array([rd_list[m_idx]["gw"] for m_idx, _ in enumerate(rd_list)], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            scaled = numer / denom
        regressors_scaled[rd] = scaled
        regressors[rd] = numer

    out_dir = os.path.join(config["work_dir"], "remote_drivers")
    os.makedirs(out_dir, exist_ok=True)

    df_raw = pd.DataFrame(regressors, index=models)
    df_scaled = pd.DataFrame(regressors_scaled, index=models)
    df_stand = df_scaled.apply(stand_pandas, axis=0)

    df_stand.to_csv(os.path.join(out_dir, "scaled_standardized_drivers.csv"))
    df_raw.to_csv(os.path.join(out_dir, "drivers.csv"))
    df_scaled.to_csv(os.path.join(out_dir, "scaled_drivers.csv"))

if __name__ == "__main__":
    with run_diagnostic() as config:
        driver_indices(config)
