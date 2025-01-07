from .utils import np, xr

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
        time_coord = xr.cftime_range(start=data.indexes['time'].to_datetimeindex()[0], periods=data['time'].size, freq='M')
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