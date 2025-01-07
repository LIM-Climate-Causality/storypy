from .utils import stats
from .processing import apply_region_mask

def test_mean_significance(sample_data):
    """Test if the mean of the sample_data is significantly different from zero."""
    if sample_data.size < 2:
        return np.nan  # Cannot compute p-value with less than 2 samples
    _, p_value = stats.ttest_1samp(sample_data, 0)
    return p_value

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