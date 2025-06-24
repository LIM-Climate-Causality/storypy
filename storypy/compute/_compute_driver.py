import xarray as xr
import numpy as np
import pandas as pd
import os
import glob

def stand(dato):
    """
    Standardize the data (z-score normalization)
    """
    anom = (dato - np.mean(dato)) / np.std(dato)
    return anom

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
    gw_file = os.path.join(work_dir, "remote_driver_gw.nc")
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
    df_standardized = df_scaled.apply(stand, axis=0)

    # Create the output directory and save the dataframes as CSV
    out_dir = os.path.join(work_dir, "remote_drivers")
    os.makedirs(out_dir, exist_ok=True)

    df_raw.to_csv(os.path.join(out_dir, "drivers.csv"))
    df_scaled.to_csv(os.path.join(out_dir, "scaled_drivers.csv"))
    df_standardized.to_csv(os.path.join(out_dir, "scaled_standardized_drivers.csv"))

    print(f"Saved driver regressors to: {out_dir}")
    return df_raw, df_scaled, df_standardized




driver_config = dict(
        var_name=['psl', 'tas', 'psl', 'psl'],            # <- actual variable names in NetCDF
        short_name=['ubi', 'utas', 'esi', 'ctp'],           # <- names for regression/CSV outputs
        period1=['1960', '1979'],
        period2=['2070', '2099'],
        # season=[12, 1, 2],
        #box={'lat_min': -15, 'lat_max': 15, 'lon_min': -180, 'lon_max': 180}, # ta
        box={'lat_min': -90, 'lat_max': 90, 'lon_min': -180, 'lon_max': 180}, # pw
        work_dir='/climca/people/storylinetool/test_user/driver_test_outputs'
    )