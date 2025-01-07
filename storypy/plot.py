import xarray as xr
import numpy as np
import os
from esmvaltool.diag_scripts.shared import run_diagnostic, get_cfg, group_metadata
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs
from scipy import stats
import regionmask


# Function to determine season based on the month
def get_season(month):
    """Return season based on the month."""
    if month in [12, 1, 2]:
        return 'DJF'
    elif month in [3, 4, 5]:
        return 'MAM'
    elif month in [6, 7, 8]:
        return 'JJA'
    elif month in [9, 10, 11]:
        return 'SON'
    else:
        return 'Unknown'


# Function to assign seasons to a DataArray
def assign_season(da):
    """Assign custom seasons to the DataArray based on months."""
    month = da['time.month']
    season = xr.apply_ufunc(
        get_season,
        month,
        vectorize=True,
        dask='parallelized',
        output_dtypes=[str]
    )
    da = da.assign_coords(season=('time', season.values))
    return da

def seasonal_data_months(data, months):
    """
    Selects specified months from an xarray object and averages the data for those months within each year.
    
    Parameters:
    - data: xarray.DataArray or xarray.Dataset
        The input data to process. It should have a 'time' coordinate.
    - months: list of int
        The months to select for averaging (1 = January, 2 = February, ..., 12 = December).
    
    Returns:
    - xarray.DataArray or xarray.Dataset
        The averaged data for the selected months within each year, accounting for months that span across years.
    """
    # Ensure 'time' coordinate is in a format that supports .dt accessor
    if np.issubdtype(data['time'].dtype, np.datetime64):
        time_coord = data['time']
    else:
        time_coord = xr.cftime_range(start=data.indexes['time'].to_datetimeindex()[0], periods=data['time'].size, freq='ME')
        data = data.assign_coords(time=time_coord)

    # Select the relevant months and keep track of the original years
    selected_months_data = data.sel(time=data['time'].dt.month.isin(months))

    # Create a new time coordinate for grouping
    new_years = selected_months_data['time'].dt.year.values.copy()

    # Shift the year for December, if necessary
    if 12 in months:
        dec_mask = selected_months_data['time'].dt.month == 12
        new_years[dec_mask] += 1  # Increment year for December

    # Assign the new year as a coordinate to the selected data
    selected_months_data = selected_months_data.assign_coords(new_year=("time", new_years))

    # Now group by the new year and calculate the mean
    averaged_data = selected_months_data.groupby("new_year").mean(dim="time")

    # Rename the new year dimension to 'time' for consistency
    averaged_data = averaged_data.rename({"new_year": "time"})

    return averaged_data

# Function to test if the mean of the sample data is significantly different from zero
def test_mean_significance(sample_data):
    """Test if the mean of the sample_data is significantly different from zero."""
    if sample_data.size < 2:
        return np.nan  # Cannot compute p-value with less than 2 samples
    _, p_value = stats.ttest_1samp(sample_data, 0)
    return p_value


# Function to apply region mask for IPCC AR6 regions
def apply_region_mask(data, region_id, lat_dim='lat', lon_dim='lon'):
    """Apply regionmask for IPCC AR6 regions and return the masked data."""
    lat_dim = 'lat' if 'lat' in data.coords else 'latitude'
    lon_dim = 'lon' if 'lon' in data.coords else 'longitude'

    # Validate lat and lon dimensions
    if lat_dim not in data.coords or lon_dim not in data.coords:
        raise ValueError(
            f"Latitude or longitude dimension missing. Available coords: {list(data.coords)}"
        )
    
    lon = data.coords[lon_dim]
    lat = data.coords[lat_dim]

    # Adjust longitudes if necessary
    if lon.max() > 180:
        lon = (((lon + 180) % 360) - 180)
        data = data.assign_coords({lon_dim: lon}).sortby(lon_dim)

    # Apply regionmask
    mask_IPCC = regionmask.defined_regions.ar6.land.mask(lon_or_obj=lon, lat=lat)
    masked_data = data.where(mask_IPCC == region_id)
    
    print(f"Region mask applied. Data min: {masked_data.min().values}, max: {masked_data.max().values}")
    return masked_data


# Function to compute climatological change
def clim_change(target, period1, period2, region_method='box', box=None, region_id=None, season=None, preserve_time_series=False):
    """
    Calculate the difference in climatological mean between two periods for a specific season and geographical region.
    """
    # Existing code to handle spatial dimensions
    spatial_dims = set(target.dims)
    lat_names = {'lat', 'latitude', 'Latitude', 'LAT'}
    lon_names = {'lon', 'longitude', 'Longitude', 'LON'}

    has_lat = spatial_dims.intersection(lat_names)
    has_lon = spatial_dims.intersection(lon_names)

    if has_lat and has_lon:
        has_spatial_dims = True
        lat_dim = has_lat.pop()
        lon_dim = has_lon.pop()
    else:
        has_spatial_dims = False

    if has_spatial_dims:
        if region_method == 'box' and box:
            target = target.sel(**{lon_dim: slice(box['lon_min'], box['lon_max']),
                                   lat_dim: slice(box['lat_min'], box['lat_max'])})
            target.attrs['region_extent'] = [box['lon_min'], box['lon_max'], box['lat_min'], box['lat_max']]
        elif region_method == 'mask' and region_id is not None:
            target = apply_region_mask(target, region_id, lat_dim=lat_dim, lon_dim=lon_dim)
            region = regionmask.defined_regions.ar6.land[region_id]
            lon_min, lat_min, lon_max, lat_max = region.bounds
            target.attrs['region_extent'] = [lon_min, lon_max, lat_min, lat_max]
        else:
            raise ValueError("Invalid region selection. Specify 'box' with 'box' parameter or 'mask' with 'region_id'.")

    # Adjust time periods for seasonal or cross-year calculations
    def adjust_period_for_season(start_year, end_year, season):
        if season == 'DJF':
            period_start = f"{int(start_year)-1}-12"
            period_end = f"{int(end_year)}-02"
        else:
            period_start = f"{start_year}-01"
            period_end = f"{end_year}-12"
        return period_start, period_end

    if season:
        period1_start, period1_end = adjust_period_for_season(period1[0], period1[1], season)
        period2_start, period2_end = adjust_period_for_season(period2[0], period2[1], season)
    else:
        period1_start = f"{period1[0]}-01"
        period1_end = f"{period1[1]}-12"
        period2_start = f"{period2[0]}-01"
        period2_end = f"{period2[1]}-12"

    # Subset data for the two periods
    period1_data = target.sel(time=slice(period1_start, period1_end))
    period2_data = target.sel(time=slice(period2_start, period2_end))

    if preserve_time_series:
        if region_method == 'box' and box:
            target = target.sel(**{lon_dim: slice(box['lon_min'], box['lon_max']),
                                lat_dim: slice(box['lat_min'], box['lat_max'])})
        elif region_method == 'mask' and region_id is not None:
            target = apply_region_mask(target, region_id, lat_dim=lat_dim, lon_dim=lon_dim)
        # Preserve the time dimension
        output = target.sel(time=slice(period1_start, period2_end))
    else:
        # Collapse time dimension
        output = period2_data.mean(dim='time') - period1_data.mean(dim='time')

    # Add region extent metadata
    if 'region_extent' in target.attrs:
        output.attrs['region_extent'] = target.attrs['region_extent']

    return output

# Plotting function with stippling
def plot_function(target_change, p_values, positives_model, negatives_model, sig_level=0.05, sig=1):
    import cartopy.feature as cfeature

    # Determine the extent of the map based on `target_change`
    if 'region_extent' in target_change.attrs:
        extent = target_change.attrs['region_extent']
    else:
        # Default to global extent
        extent = [-180, 180, -90, 90]

    # Subset `target_change` to its actual extent
    target_change = target_change.sel(lon=slice(extent[0], extent[1]), lat=slice(extent[2], extent[3]))

    # Dynamically set extent based on `target_change` actual data bounds
    extent = [
        float(target_change['lon'].min()), float(target_change['lon'].max()),
        float(target_change['lat'].min()), float(target_change['lat'].max())
    ]

    # Debugging
    print("Target Change Extent:")
    print(f"Longitude: {target_change['lon'].min().values} to {target_change['lon'].max().values}")
    print(f"Latitude: {target_change['lat'].min().values} to {target_change['lat'].max().values}")

    # Align all datasets to the same grid as `target_change`
    if p_values is not None:
        print("Aligning p_values...")
        p_values = p_values.sel(lon=slice(extent[0], extent[1]), lat=slice(extent[2], extent[3]))
        #p_values = p_values.interp_like(target_change)  # Align to `target_change`

        # Debugging
        print("P-Values Extent:")
        print(f"Longitude: {p_values['lon'].min().values} to {p_values['lon'].max().values}")
        print(f"Latitude: {p_values['lat'].min().values} to {p_values['lat'].max().values}")

    if positives_model is not None and negatives_model is not None:
        print("Aligning positives_model and negatives_model...")
        positives_model_da = positives_model['pr']  # Replace 'pr' with the correct variable name
        negatives_model_da = negatives_model['pr']  # Replace 'pr' with the correct variable name

        # Subset their coordinates to match the extent
        positives_model_da = positives_model_da.sel(lon=slice(extent[0], extent[1]), lat=slice(extent[2], extent[3]))
        negatives_model_da = negatives_model_da.sel(lon=slice(extent[0], extent[1]), lat=slice(extent[2], extent[3]))

        # Interpolate to match `target_change`
        #positives_model_da = positives_model_da.interp_like(target_change)
        #negatives_model_da = negatives_model_da.interp_like(target_change)

        # Debugging
        print("Positives Model Extent:")
        print(f"Longitude: {positives_model_da['lon'].min().values} to {positives_model_da['lon'].max().values}")
        print(f"Latitude: {positives_model_da['lat'].min().values} to {positives_model_da['lat'].max().values}")

        print("Negatives Model Extent:")
        print(f"Longitude: {negatives_model_da['lon'].min().values} to {negatives_model_da['lon'].max().values}")
        print(f"Latitude: {negatives_model_da['lat'].min().values} to {negatives_model_da['lat'].max().values}")

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='black', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':', edgecolor='gray')

    # Plot `target_change` as the main contour
    target_change.plot.contourf(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='PuOr',
        levels=20,
        add_colorbar=True,
        cbar_kwargs={'shrink': 0.7, 'label': 'Precipitation Change (mm/day)'}
    )

    # Add stippling for `p_values` (significance)
    if p_values is not None and p_values.size > 0:
        print("Adding p_values stippling...")
        sig_mask = p_values < sig_level
        ax.contourf(
            p_values['lon'], p_values['lat'], sig_mask,
            levels=[0.5, 1], hatches=['o', ''], colors='none',
            transform=ccrs.PlateCarree()
        )

    # Add stippling for `positives_model` and `negatives_model`
    if positives_model is not None and negatives_model is not None:
        print("Adding positives/negatives stippling...")
        combined = (positives_model_da + negatives_model_da) == sig
        ax.contourf(
            positives_model_da['lon'], positives_model_da['lat'], combined,
            levels=[0.5, 1], hatches=['...', ''], colors='none',
            transform=ccrs.PlateCarree()
        )

    # Add title and gridlines
    ax.set_title("Target Change with Stippling")
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
    gl.top_labels = gl.right_labels = False

    return fig

def plot_precipitation_change_two_regions(model_changes, region, years, titles):
    """
    Create a two-panel plot showing precipitation changes over time for two regions.

    Parameters:
        model_changes: list of xr.DataArray
            List of time series data for each model. Each DataArray should have time as a dimension.
        region_a_extent: tuple
            Tuple defining the lat/lon bounds for Region A (lat_min, lat_max, lon_min, lon_max).
        region_b_extent: tuple
            Tuple defining the lat/lon bounds for Region B (lat_min, lat_max, lon_min, lon_max).
        years: np.ndarray
            Array of years corresponding to the time series data.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import linregress

    n_regions = len(region)
    # Create the plot
    fig, axes = plt.subplots(int((n_regions+1)/2), 2, figsize=(12, 4), sharey=True)
    colors = {'mean': 'blue', 'ci': 'red', 'ensemble': 'gray'}
    axes = axes.flatten()
    for i, region_extent in enumerate(region):
        # Helper function to calculate mean, ensemble spread, and confidence intervals
        def calculate_region_stats(model_changes, region):
            region_changes = []
            for model in model_changes:
                model_region = model.sel(
                    lon=slice(region_extent[2], region_extent[3]),
                    lat=slice(region_extent[0], region_extent[1])
                )
                model_mean = model_region.mean(dim=['lon', 'lat'])  # Reduce spatial dimensions
                region_changes.append(model_mean)

            # Concatenate over the ensemble dimension
            stacked_data = xr.concat(region_changes, dim='ensemble')

            # Ensure time dimension aligns with `years`
            #if 'time' in stacked_data.dims:
            #    stacked_data = stacked_data.sel(time=years)  # Align to `years`

            # Compute ensemble mean, upper, and lower confidence intervals
            ensemble_mean = stacked_data.mean(dim='ensemble')
            upper_ci = ensemble_mean + stacked_data.std(dim='ensemble')
            lower_ci = ensemble_mean - stacked_data.std(dim='ensemble')

            return {
                'ensemble': stacked_data.values,
                'mean': ensemble_mean.values,
                'upper': upper_ci.values,
                'lower': lower_ci.values,
            }

        # Calculate stats for Region A and Region B
        data = calculate_region_stats(model_changes, region)

        # Ensure `years` aligns with the shape of the computed means
        if len(years) != len(data['mean']):
            raise ValueError(
                f"Mismatch between years ({len(years)}) and computed data (Region A: {len(data['mean'])}"
            )

        # Calculate regression and R/p-values
        r_value, p_value = linregress(years, data['mean'])[2:4]
        # Panel A (Region A)
        ax = axes[i]
        ax.plot(years, data['ensemble'].T, color=colors['ensemble'], alpha=0.3, linewidth=0.8)
        ax.plot(years, data['mean'], color=colors['mean'], label='Mean')
        ax.fill_between(years, data['lower'], data['upper'], color=colors['ci'], alpha=0.2, label='Confidence Interval')
        ax.text(years[0] + 5, max(data['upper']) - 0.2, f"R = {r_value:.2f} (p = {p_value:.3f})", fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7))
        ax.set_title(titles[i], fontsize=12)
        ax.set_ylabel("pr change [mm day$^{-1}$]")
        ax.set_xlabel("Years")
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.legend()
        # Add subplot labels
        #ax.text(-0.1, 1.05, "a", transform=axes[0].transAxes, fontsize=14, fontweight='bold') FIND A WAY TO DEFINE ITERATIVELY

    plt.tight_layout()
    return fig

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
        time_series_changes, [(40, 50, 10, 20),(30, 40, 30, 40)],  # B regions
        years=years, titles=["Region A", "Region X"]
    )
    if fig_time_series:
        fig_time_series.savefig(os.path.join(config["plot_dir"], "time_series_plot_text.png"))

if __name__ == "__main__":

    user_config = dict(
        # Choose region selection method: 'box' or 'mask'
        region_method = 'mask',  # Set to 'box' or 'mask'
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
