from storypy.utils import np, xr, stats

# Function to compute climatological change
def extract_metadata(files):
    """Extracts metadata from NetCDF files and returns a DataFrame."""
    data = []

    for file in files:
        with xr.open_dataset(file, use_cftime=True) as ds:
            # Extract metadata here, for example:
            model = ds.attrs.get('source_id', 'Unknown')
            experiment = ds.attrs.get('experiment_id', 'Unknown')
            variant = ds.attrs.get('variant_label', 'Unknown')
            variable = file.split('_')[-2]  # Assuming the variable is in the filename

            data.append({
                'filename': file,
                'model': model,
                'experiment': experiment,
                'variant': variant,
                'variable': variable
            })

    return pd.DataFrame(data)

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

def adjust_longitudes(data, lon_dim='lon'):
    """
    Adjust the longitude coordinate of an xarray DataArray or Dataset
    to the -180 to 180 range if they are originally in the 0 to 360 range.

    Parameters
    ----------
    data : xarray.DataArray or Dataset
        The data whose longitude coordinate will be adjusted.
    lon_dim : str
        The name of the longitude coordinate (default is 'lon').

    Returns
    -------
    data : xarray.DataArray or Dataset
        The data with adjusted longitude coordinate.
    """
    if lon_dim in data.coords:
        lon = data.coords[lon_dim]
        # If maximum longitude is greater than 180, adjust to -180 to 180
        if lon.max() > 180:
            new_lon = (((lon + 180) % 360) - 180)
            data = data.assign_coords({lon_dim: new_lon}).sortby(lon_dim)
    return data

def clim_change(target, period1, period2, region_method='box', box=None, region_id=None, season=None, preserve_time_series=False):
    """
    Calculate the difference in climatological mean between two periods for a specific season and geographical region.
    """

    target = adjust_longitudes(target, lon_dim='lon')
    
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
        if season == '12,1,2':
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