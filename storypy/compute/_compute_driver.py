import xarray as xr
import numpy as np
import pandas as pd
import os
import glob

from esmvaltool.diag_scripts.shared import group_metadata, run_diagnostic

def stand_numpy(dato):
    """
    Standardize the data (z-score normalization)
    """
    anom = (dato - np.mean(dato)) / np.std(dato)
    return anom

def stand_pandas(series: pd.Series) -> pd.Series:
    """Standardize across models (z-score)."""
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return (series - series.mean()) * np.nan
    return (series - series.mean()) / std

def compute_drivers_from_netcdf(driver_config):
    """
    Compute driver values from preprocessed NetCDF files stored in driver_config['work_dir'].

    The function handles both variable extraction and global warming scaling.

    driver_config:
        var_name: list of variable names in NetCDF
        short_name: list of output variable names for regression/CSV
        work_dir: path where NetCDF files are stored
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
    """Pick exactly one entry for a (dataset, variable_group).
    Prefer esmvaltool's alias == 'EnsembleMean'; otherwise first match.
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
    """Return (rd_list, models) and persist a NetCDF with one scalar per model/driver."""
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
    all_vars = sorted(set().union(*[d.keys() for d in rd_list]))
    data_vars = {}
    for var in all_vars:
        vals = [d.get(var, np.nan) for d in rd_list]
        data_vars[var] = ("model", np.asarray(vals, dtype=np.float64))
    return xr.Dataset(data_vars, coords={"model": models})


def driver_indices(config):
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
