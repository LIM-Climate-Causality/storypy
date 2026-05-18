from .plotting import plot_function, plot_precipitation_change
from .processing import seasonal_data_months, apply_region_mask
from .diagnostics import clim_change, test_mean_significance
#from esmvaltool.diag_scripts.shared import run_diagnostic, get_cfg, group_metadata
import os
import fnmatch
from .utils import np, xr

# CMIP data direct processing logic from data directory
def main_direct(user_config):
    """
    This function processes netCDF files from CMIP6 data without esmvaltool metadata.
    It calculates seasonal climatological changes (both spatial and as a time series)
    normalized by a global warming (gw) variable, and then writes the combined ensemble
    changes to a netCDF file. It also produces time series plots for each target variable.
    """
    xr.set_options(keep_attrs=True)

    # Calculate the overall bounding box from region_extents
    all_lat_min = min(region[0] for region in user_config['region_extents'])
    all_lat_max = max(region[1] for region in user_config['region_extents'])
    all_lon_min = min(region[2] for region in user_config['region_extents'])
    all_lon_max = max(region[3] for region in user_config['region_extents'])

    # Expand by 5 degrees and update user_config with a new "box"
    user_config['box'] = {
        'lat_min': max(all_lat_min - 5, -90),
        'lat_max': min(all_lat_max + 5, 90),
        'lon_min': all_lon_min - 5 if all_lon_min - 5 >= -180 else -180,
        'lon_max': all_lon_max + 5 if all_lon_max + 5 <= 180 else 180
    }

    # Unpack configuration
    var_names = user_config["var_name"]         # e.g. ['pr', 'uas']
    data_dir = user_config["data_dir"]            # e.g. '/climca/data/cmip6-ng'
    work_dir = user_config["work_dir"]
    plot_dir = user_config["plot_dir"]
    exp_name = user_config["exp_name"]            # e.g. 'ssp585'
    freq = user_config["freq"]                    # e.g. 'mon'
    grid = user_config["grid"]                    # e.g. 'g025'
    region_method = user_config["region_method"]
    box = user_config["box"]
    period1 = user_config["period1"]
    period2 = user_config["period2"]
    region_id = user_config["region_id"]
    season = user_config["season"]                # now provided as a tuple, e.g. (11,12,1,2,3)
    region_extents = user_config["region_extents"]
    titles = user_config["titles"]

    # Create dictionaries to store the computed changes (for each variable)
    ensemble_changes = {var: [] for var in var_names}
    time_series_changes = {var: [] for var in var_names}
    target = {}  # to store the concatenated target DataArrays (for each variable and for 'gw')

    # Determine common models across all variables
    model_sets = {}
    for var in var_names:
        var_path = os.path.join(data_dir, var, freq, grid)
        if not os.path.exists(var_path):
            print(f"Directory not found for variable {var}: {var_path}")
            continue
        file_list = os.listdir(var_path)
        # Using a set to ensure models are unique.
        models = {f.split('_')[2] for f in file_list if f.endswith('.nc')}
        model_sets[var] = models

    if model_sets:
        # Compute the intersection of all model sets.
        common_models = set.intersection(*model_sets.values())
        print("Common models across all variables:", common_models)
    else:
        common_models = set()

    # Loop over each target variable
    for var in var_names:
        path = os.path.join(data_dir, var, freq, grid)
        if not os.path.exists(path):
            print(f"Directory not found: {path}")
            continue  # skip if the directory does not exist

        listOfFiles = os.listdir(path)
        # Extract model names assuming the filename structure splits by '_' and index 2 is the model name.
        # model_names = [f.split('_')[2] for f in listOfFiles if f.endswith('.nc')]
        models = sorted(common_models)
        
        for m in models:  # Process only the first 5 models
            # Select files for this model that are either historical or scenario (exp_name)
            ensemble_list = [f for f in listOfFiles if m in f and ('historical' in f or f.find('_'+exp_name+'_') != -1)]
            ens_members_hist = []
            ens_members_scenario = []
            ens_members_hist_gw = []
            ens_members_scenario_gw = []
            
            for entry in ensemble_list:
                parts = entry.split('_')
                # Construct the corresponding tas filename for the global warming variable
                entry_tas = 'tas_' + '_'.join(parts[1:])
                pattern_hist = var + "*_" + m + "_*historical*.nc"
                pattern_scenario = var + "*_" + m + "_*" + exp_name + "*.nc"
                
                if fnmatch.fnmatch(entry, pattern_hist):
                    ds_hist = xr.open_dataset(os.path.join(data_dir, var, freq, grid, entry))
                    ens_members_hist.append(ds_hist)
                    ds_hist_gw = xr.open_dataset(os.path.join(data_dir, 'tas', freq, grid, entry_tas))
                    ens_members_hist_gw.append(ds_hist_gw)
                elif fnmatch.fnmatch(entry, pattern_scenario):
                    ds_scenario = xr.open_dataset(os.path.join(data_dir, var, freq, grid, entry))
                    ens_members_scenario.append(ds_scenario)
                    ds_scenario_gw = xr.open_dataset(os.path.join(data_dir, 'tas', freq, grid, entry_tas))
                    ens_members_scenario_gw.append(ds_scenario_gw)
            
            # If there are no historical or scenario ensemble members, skip this model.
            if not ens_members_hist or not ens_members_scenario:
                continue

            # Concatenate historical and scenario ensemble members along the 'ensemble' dimension
            model_hist = xr.concat(ens_members_hist, dim='ensemble')
            model_scenario = xr.concat(ens_members_scenario, dim='ensemble')
            # For the target variable, we take the ensemble mean for historical and scenario separately and then concatenate along time.
            target[var] = xr.concat([model_hist.mean(dim='ensemble'), model_scenario.mean(dim='ensemble')], dim='time')

            # Process corresponding global warming data (tas)
            if ens_members_hist_gw and ens_members_scenario_gw:
                model_hist_gw = xr.concat(ens_members_hist_gw, dim='ensemble')
                model_scenario_gw = xr.concat(ens_members_scenario_gw, dim='ensemble')
                # For gw, take the ensemble mean then average over lat and lon so that gw is a 1D time series.
                target['gw'] = xr.concat([model_hist_gw.mean(dim='ensemble'), model_scenario_gw.mean(dim='ensemble')], dim='time')
                target['gw'] = target['gw'].mean(dim='lat').mean(dim='lon')
        
        # At this point, target[var] is a DataArray with a time dimension.
        # Apply seasonal subsetting using seasonal_data_months() function.
            try:
                # Pass the season tuple as a list to seasonal_data_months.
                target_var = seasonal_data_months(target[var], list(season))
                if var == 'pr':
                    target_var = target_var * 86400  # Convert precipitation to mm/day
                
                # Compute the climatological change for the target variable.
                target_var_change = clim_change(
                    target_var, period1=period1, period2=period2,
                    region_method=region_method, box=box, region_id=region_id, season=season, preserve_time_series=False
                )
                
                # Compute the time series of changes (preserving the time dimension)
                target_var_change_time_series = clim_change(
                    target_var, period1=period1, period2=period2,
                    region_method=region_method, box=box, region_id=region_id, season=season, preserve_time_series=True
                )
                
                # Process global warming (gw) similarly.
                seasonal_gw = seasonal_data_months(target['gw'], list(season))
                has_spatial_dims_gw = ('lat' in seasonal_gw.dims) and ('lon' in seasonal_gw.dims)
                if has_spatial_dims_gw:
                    target_gw_change = clim_change(
                        seasonal_gw, period1=period1, period2=period2,
                        region_method=region_method, box=box, region_id=region_id, season=season, preserve_time_series=False
                    )
                else:
                    target_gw_change = clim_change(
                        seasonal_gw, period1=period1, period2=period2,
                        season=season, preserve_time_series=False
                    )
                if not has_spatial_dims_gw:
                    target_gw_change = target_gw_change.expand_dims({'lat': target_var_change['lat'], 'lon': target_var_change['lon']})
                
                # Assume that the global warming variable in target_gw_change is stored in variable 'tas'
                # gw_var = target_gw_change['tas']
                
                # # Append the computed changes, normalizing by the gw variable.
                # ensemble_changes[var].append(target_var_change / gw_var)
                # time_series_changes[var].append(target_var_change_time_series / gw_var)

                # Assume that the global warming variable in target_gw_change is stored in variable 'tas'
                gw_var = target_gw_change['tas']
                
                # Append the computed changes, normalizing by the gw variable.
                norm_change = (target_var_change / gw_var).expand_dims({'model': [m]})
                norm_change_ts = (target_var_change_time_series / gw_var).expand_dims({'model': [m]})

                ensemble_changes[var].append(norm_change)
                time_series_changes[var].append(norm_change_ts)
            except KeyError as e:
                print(f"KeyError: {e}. Skipping variable {var}.")
                continue

    # # Plot time series for each variable separately.
    # years = np.arange(1950, 2100)
    # for var in var_names:
    #     fig_time_series = plot_precipitation_change(
    #         time_series_changes[var],
    #         region_extents=region_extents,
    #         years=years,
    #         var_name=var
    #     )
    #     if fig_time_series:
    #         os.makedirs(plot_dir, exist_ok=True)
    #         fig_time_series.savefig(os.path.join(plot_dir, f"time_series_plot_{var}.png"))

        # Combine ensemble changes into one Dataset (one variable per target)
    combined_changes_ar = xr.Dataset({
        var: xr.concat([ds.to_dataarray() for ds in ensemble_changes[var]], dim='model', coords='minimal', compat='override')
        for var in var_names
    })
    for var in var_names:
        combined_changes_ar[var] = combined_changes_ar[var].assign_coords(model=np.unique(combined_changes_ar[var]['model']))
    
    # combined_changes_ar.attrs["model_names"] = ','.join(sorted(common_models))

    output_file = os.path.join(work_dir, "combined_changes.nc")
    combined_changes_ar.to_netcdf(output_file)
    print(f"Saved all model ensemble changes to {output_file}")

    # Plot time series for each variable separately.
    start_year = 1950
    end_year = 2100  # adjust as needed
    years = np.arange(start_year, end_year)
    # years = np.arange(1950, 2099)
    for var in var_names:
        fig_time_series = plot_precipitation_change(
            time_series_changes[var],
            region_extents=region_extents,
            years=years,
            var_name=var
        )
        if fig_time_series:
            os.makedirs(plot_dir, exist_ok=True)
            fig_time_series.savefig(os.path.join(plot_dir, f"time_series_plot_{var}.png"))


# Esmvaltool-based logic
def main_esmval(config, user_config):
    """
    Integrated main function for StoryPy that processes CMIP data using esmvaltool metadata.
    This version processes any target variable(s) listed in user_config["var_name"] and computes 
    climatological changes (both spatial and as a time series) normalized by a global warming variable.
    
    The function preserves the original logic while adding flexibility for multiple variables.
    """
    xr.set_options(keep_attrs=True)
    
    # Calculate the overall bounding box from region extents (assumed to be in the 0-360, -90-90 format)
    all_lat_min = min(region[0] for region in user_config['region_extents'])
    all_lat_max = max(region[1] for region in user_config['region_extents'])
    all_lon_min = min(region[2] for region in user_config['region_extents'])
    all_lon_max = max(region[3] for region in user_config['region_extents'])
    
    # Expand the bounding box by 5 degrees on all sides
    user_config['box'] = {
        'lat_min': max(all_lat_min - 5, -90),
        'lat_max': min(all_lat_max + 5, 90),
        'lon_min': all_lon_min - 5 if all_lon_min - 5 >= -180 else -180,
        'lon_max': all_lon_max + 5 if all_lon_max + 5 <= 180 else 180
    }
    
    # Unpack configuration
    region_method = user_config["region_method"]
    box = user_config["box"]
    region_id = user_config["region_id"]
    season = user_config["season"]            # e.g., "DJF" or a tuple like (12, 1, 2)
    region_extents = user_config["region_extents"]
    var_names = user_config["var_name"]         # e.g., ['pr', 'psl']
    titles = user_config["titles"]
    
    # Group metadata from esmvaltool input data
    meta_dataset = group_metadata(config["input_data"].values(), "dataset")
    meta = group_metadata(config["input_data"].values(), "alias")
    
    os.makedirs(config["plot_dir"], exist_ok=True)
    
    # Prepare dictionaries to store results for each variable
    ensemble_changes = {var: [] for var in var_names}
    time_series_changes = {var: [] for var in var_names}
    target = {}  # To store the concatenated target DataArrays for each variable and for 'gw'
    
    # Loop over each dataset (grouped by esmvaltool metadata)
    for dataset, dataset_list in meta_dataset.items():
        meta_group = group_metadata(dataset_list, "alias")
        # For each alias (i.e., each ensemble member grouping)
        for alias, alias_list in meta_group.items():
            for var in var_names:
                # Open the target variable dataset and the global warming (gw) dataset.
                # (Assuming that 'pr', 'psl', etc. are in one group and 'gw' is in another.)
                target_data = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]]
                               for m in alias_list if (m["dataset"] == dataset) and (m["variable_group"] == var)}
                target_gw = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]]
                             for m in alias_list if (m["dataset"] == dataset) and (m["variable_group"] == 'gw')}
                try:
                    # Apply seasonal subsetting and adjust longitudes
                    target_data[var] = seasonal_data_months(target_data[var], [12, 1, 2])
                    target_data[var] = adjust_longitudes(target_data[var], lon_dim='lon')
                    if var == 'pr':
                        target_data[var] *= 86400  # Convert precipitation to mm/day
                    
                    # Compute climatological change (spatial; time collapsed)
                    target_var_change = clim_change(
                        target_data[var], ['1950', '1979'], ['2070', '2099'],
                        region_method=region_method, box=box, region_id=region_id,
                        season=season, preserve_time_series=False
                    )
                    # Compute climatological change time series (preserving time)
                    target_var_change_ts = clim_change(
                        target_data[var], ['1950', '1979'], ['2070', '2099'],
                        region_method=region_method, box=box, region_id=region_id,
                        season=season, preserve_time_series=True
                    )
                    
                    # Process the global warming variable similarly
                    seasonal_gw = seasonal_data_months(target_gw['gw'], [12, 1, 2])
                    seasonal_gw = adjust_longitudes(seasonal_gw, lon_dim='lon')
                    has_spatial_dims_gw = ('lat' in seasonal_gw.dims) and ('lon' in seasonal_gw.dims)
                    if has_spatial_dims_gw:
                        target_gw_change = clim_change(
                            seasonal_gw, ['1950', '1979'], ['2070', '2099'],
                            region_method=region_method, box=box, region_id=region_id,
                            season=season, preserve_time_series=False
                        )
                    else:
                        target_gw_change = clim_change(
                            seasonal_gw, ['1950', '1979'], ['2070', '2099'],
                            season=season, preserve_time_series=False
                        )
                    if not has_spatial_dims_gw:
                        target_gw_change = target_gw_change.expand_dims({'lat': target_var_change['lat'], 'lon': target_var_change['lon']})
                    
                    # Assume that the global warming variable is stored under 'tas'
                    #gw_var = target_gw_change['tas']
                    
                    # Append the normalized changes for the current ensemble grouping
                    ensemble_changes[var].append(target_var_change / target_gw_change)
                    time_series_changes[var].append(target_var_change_ts / target_gw_change)
                except KeyError as e:
                    print(f"KeyError: {e}. Skipping {alias} for dataset {dataset}.")
                    continue
    
    # Process and plot spatial ensemble mean for each variable
    for var in var_names:
        if ensemble_changes[var]:
            ensemble_changes_ar = xr.concat(ensemble_changes[var], dim='model')
            ensemble_mean = ensemble_changes_ar.mean(dim='model')
    
            # Compute p-values (if applicable)
            if 'model' in ensemble_changes_ar.dims and ensemble_changes_ar.sizes['model'] > 1:
                ensemble_maps_pval = xr.apply_ufunc(
                    test_mean_significance, ensemble_changes_ar,
                    input_core_dims=[["model"]],
                    output_core_dims=[[]], vectorize=True, dask="parallelized"
                )
            else:
                print("Only one model available; cannot compute p-values.")
                ensemble_maps_pval = None
    
            # Load stippling models (from file)
            positives_model = xr.open_dataset(os.path.join(config["work_dir"], 'stippling', 'number_of_models_positive_trend_CMIP6.nc'))
            negatives_model = xr.open_dataset(os.path.join(config["work_dir"], 'stippling', 'number_of_models_negative_trend_CMIP6.nc'))
    
            # Plot the spatial map for this variable
            fig_spatial = plot_function(ensemble_mean, ensemble_maps_pval, positives_model, negatives_model, region_extents)
            if fig_spatial:
                fig_spatial.savefig(os.path.join(config["plot_dir"], f"beta_forced_{var}_plot.png"))
    
    # Process and plot time series for each variable
    years = np.arange(1950, 2100)
    for var in var_names:
        fig_time_series = plot_precipitation_change(
            time_series_changes[var],
            region_extents=region_extents,
            years=years,
            var_name=var
        )
        if fig_time_series:
            os.makedirs(user_config["plot_dir"], exist_ok=True)
            fig_time_series.savefig(os.path.join(user_config["plot_dir"], f"time_series_plot_{var}.png"))
            

if __name__ == "__main__":
    user_config = dict(
        data_dir='/climca/data/cmip6-ng',
        work_dir='/climca/people/storylinetool/test_user/work_dir',
        plot_dir='/climca/people/storylinetool/test_user/plot_dir',
        var_name=['pr'],
        exp_name='ssp585',
        freq='mon',
        grid='g025',
        region_method='box',
        period1 = ['1950', '1979'],
        period2 = ['2070', '2099'],
        region_id=18,
        season=[11, 12, 1, 2, 3],  # Here you can also supply a tuple of months, e.g. (12, 1, 2)
        region_extents=[(30, 45, -10, 40), (45, 55, 5, 20), (-40, 0, -60, -30)], #[(46, 49, 16, 19), (42, 45, 12, 15), (35, 40, 20, 25)],
        titles=["Region A", "Region B"]
    )
    main_esmval(config, user_config)
    main_direct(user_config)
