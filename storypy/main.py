from .plotting import plot_function, plot_precipitation_change_two_regions
from .processing import seasonal_data_months, apply_region_mask
from .diagnostics import clim_change, test_mean_significance
#from esmvaltool.diag_scripts.shared import run_diagnostic, get_cfg, group_metadata
import os
from .utils import np, xr

# Main function
def main(config, user_config):
    """Run the diagnostic."""
    cfg = get_cfg(os.path.join(config["run_dir"], "settings.yml"))
    xr.set_options(keep_attrs=True)
    #print(cfg)

    region_method = user_config["region_method"]
    box = user_config["box"]
    region_id = user_config["region_id"]
    season = user_config["season"]

    # Group metadata
    meta_dataset = group_metadata(config["input_data"].values(), "dataset")
    meta = group_metadata(config["input_data"].values(), "alias")

    os.makedirs(config["plot_dir"], exist_ok=True)

    # Lists to store spatial changes and time series changes
    ensemble_changes = []
    time_series_changes = []

    for dataset, dataset_list in meta_dataset.items():
        meta = group_metadata(dataset_list, "alias")
        model_changes = []
        for alias, alias_list in meta.items():
            target_pr = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]]
                             for m in alias_list if (m["dataset"] == dataset) & (m["variable_group"] == 'pr')}
            target_gw = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]]
                             for m in alias_list if (m["dataset"] == dataset) & (m["variable_group"] == 'gw')}
            
            #target_pr = assign_custom_season(target_pr)
            #target_gw = assign_custom_season(target_gw)

            try:
                target_pr = seasonal_data_months(target_pr['pr'], [12,1,2]) * 86400  # Convert to mm/day

                # Convert the time calendar for `target_pr`
                #if 'time' in target_pr.dims:
                #    target_pr = target_pr.convert_calendar('proleptic_gregorian', align_on='year')
                # Convert the time calendar for `target_gw`
                #if 'time' in target_gw['gw'].dims:
                #    target_gw['gw'] = target_gw['gw'].convert_calendar('proleptic_gregorian', align_on='year')

                #Get the seasonal data
                #target_ch

                # Compute spatial changes
                target_pr_change = clim_change(
                    target_pr, ['1950', '1979'], ['2070', '2099'],
                    region_method=region_method, box=box, region_id=region_id, season=season, preserve_time_series=False
                )

                # Compute time series changes
                target_pr_change_time_series = clim_change(
                    target_pr, ['1950', '1979'], ['2070', '2099'],
                    region_method=region_method, box=box, region_id=region_id, season=season, preserve_time_series=True
                )

                # Handle global warming variable (target_gw)
                has_spatial_dims_gw = 'lat' in seasonal_data_months(target_gw['gw'], [12,1,2]).dims and 'lon' in seasonal_data_months(target_gw['gw'], [12,1,2]).dims
                if has_spatial_dims_gw:
                    target_gw_change = clim_change(
                        seasonal_data_months(target_gw['gw'], [12,1,2]), ['1950', '1979'], ['2070', '2099'],
                        region_method=region_method, box=box, region_id=region_id, season=season, preserve_time_series=False
                    )
                else:
                    target_gw_change = clim_change(
                        seasonal_data_months(target_gw['gw'], [12,1,2]), ['1950', '1979'], ['2070', '2099'],
                        season=season, preserve_time_series=False
                    )

                # Append the results for spatial and time series changes
                model_changes.append(target_pr_change / target_gw_change)
                time_series_changes.append(target_pr_change_time_series)
            except KeyError as e:
                print(f"KeyError: {e}. Skipping {alias} for dataset {dataset}.")
                continue

        if model_changes:
            # Concatenate model changes for this dataset
            model_changes_ar = xr.concat(model_changes, dim='ensemble')
            model_mean = model_changes_ar.mean(dim='ensemble')

            # Compute p-values for significance testing
            if 'ensemble' in model_changes_ar.dims and model_changes_ar.sizes['ensemble'] > 1:
                model_maps_pval = xr.apply_ufunc(
                    test_mean_significance, model_changes_ar,
                    input_core_dims=[["ensemble"]],
                    output_core_dims=[[]], vectorize=True, dask="parallelized"
                )
            else:
                print("Only one ensemble member available; cannot compute p-values.")
                model_maps_pval = None

            # Append to ensemble-level changes
            ensemble_changes.append(model_mean)

    # Process spatial ensemble mean and plot spatial map
    if ensemble_changes:
        ensemble_changes_ar = xr.concat(ensemble_changes, dim='model')
        ensemble_mean = ensemble_changes_ar.mean(dim='model')

        # Compute p-values for ensemble models
        if 'model' in ensemble_changes_ar.dims and ensemble_changes_ar.sizes['model'] > 1:
            ensemble_maps_pval = xr.apply_ufunc(
                test_mean_significance, ensemble_changes_ar,
                input_core_dims=[["model"]],
                output_core_dims=[[]], vectorize=True, dask="parallelized"
            )
        else:
            print("Only one model available; cannot compute p-values.")
            ensemble_maps_pval = None

        # Load stippling models
        positives_model = xr.open_dataset(os.path.join(config["work_dir"], 'stippling', 'number_of_models_positive_trend_CMIP6.nc'))
        negatives_model = xr.open_dataset(os.path.join(config["work_dir"], 'stippling', 'number_of_models_negative_trend_CMIP6.nc'))

        # Plot spatial map
        fig_spatial = plot_function(ensemble_mean, ensemble_maps_pval, positives_model, negatives_model)
        if fig_spatial:
            fig_spatial.savefig(os.path.join(config["plot_dir"], "beta_forced_CMIP6_plot.png"))

    # Process time series data and plot time series for two regions
    years = np.arange(1950, 2100)  # Time dimension of data
    fig_time_series = plot_precipitation_change_two_regions(
        time_series_changes,
        region_a_extent=(46, 49, 16, 19),  # A regions
        region_b_extent=(42, 45, 12, 15),  # B regions
        years=years
    )
    if fig_time_series:
        fig_time_series.savefig(os.path.join(config["plot_dir"], "time_series_plot.png"))

if __name__ == "__main__":

    user_config = dict(
        # Choose region selection method: 'box' or 'mask'
        region_method = 'box',  # Set to 'box' or 'mask'
        # Define the bounding box for the desired region if using 'box'
        box = {
            'lon_min': 10, 'lon_max': 20,  # Example region
            'lat_min': 40, 'lat_max': 50
        },
        # Define the region_id for regionmask if using 'mask'
        #region_id = int(input("Please enter a region ID from the list of IPCC region list: "))
        region_id = 18,  # Adjusted region ID
        season = "DJF"  # Specify the season here (e.g., "DJF" for winter)
    )


    with run_diagnostic() as config:
        main(config, user_config)
