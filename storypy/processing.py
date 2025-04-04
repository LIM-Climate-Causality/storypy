from .utils import np, xr
from shapely.geometry.polygon import Polygon

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

def create_arc(lon_min, lon_max, lat_min, lat_max, n_points=100):
    lons = np.linspace(lon_min, lon_max, n_points)
    lats1 = np.full(n_points, lat_min)
    lats2 = np.full(n_points, lat_max)
    lons_combined = np.concatenate([lons, lons[::-1]])
    lats_combined = np.concatenate([lats1, lats2[::-1]])
    return Polygon(zip(lons_combined, lats_combined))

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