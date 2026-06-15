from storypy.utils import np, xr, plt, ccrs, gridspec, cfeature, pd
from shapely.geometry.polygon import Polygon

def create_arc(lon_min, lon_max, lat_min, lat_max, n_points=100):
    lons = np.linspace(lon_min, lon_max, n_points)
    lats1 = np.full(n_points, lat_min)
    lats2 = np.full(n_points, lat_max)
    lons_combined = np.concatenate([lons, lons[::-1]])
    lats_combined = np.concatenate([lats1, lats2[::-1]])
    return Polygon(zip(lons_combined, lats_combined))

# Plotting function with stippling
# def plot_function(target_change, p_values, positives_model, negatives_model, region_extents, sig_level=0.05, sig=1):
#     import cartopy.feature as cfeature

#     plt.rcParams.update({
#         "font.size": 18,
#         "axes.titlesize": 18,
#         "axes.labelsize": 18,
#         "xtick.labelsize": 18,
#         "ytick.labelsize": 18,
#         "figure.titlesize": 20
#     })

#     # Determine the extent of the map based on `target_change`
#     if 'region_extent' in target_change.attrs:
#         extent = target_change.attrs['region_extent']
#     else:
#         # Default to global extent
#         #extent = [-180, 180, -90, 90]
#         extent = [0, 360, -90, 90]

#     # Subset `target_change` to its actual extent
#     target_change = target_change.sel(lon=slice(extent[0], extent[1]), lat=slice(extent[2], extent[3]))

#     # Dynamically set extent based on `target_change` actual data bounds
#     extent = [
#         float(target_change['lon'].min()), float(target_change['lon'].max()),
#         float(target_change['lat'].min()), float(target_change['lat'].max())
#     ]

#     # Align all datasets to the same grid as `target_change`
#     if p_values is not None:
#         print("Aligning p_values...")
#         p_values = p_values.sel(lon=slice(extent[0], extent[1]), lat=slice(extent[2], extent[3]))
#         #p_values = p_values.interp_like(target_change)  # Align to `target_change`

#     if positives_model is not None and negatives_model is not None:
#         print("Aligning positives_model and negatives_model...")
#         positives_model_da = positives_model['pr']  # Replace 'pr' with the correct variable name
#         negatives_model_da = negatives_model['pr']  # Replace 'pr' with the correct variable name

#         # Subset their coordinates to match the extent
#         positives_model_da = positives_model_da.sel(lon=slice(extent[0], extent[1]), lat=slice(extent[2], extent[3]))
#         negatives_model_da = negatives_model_da.sel(lon=slice(extent[0], extent[1]), lat=slice(extent[2], extent[3]))

#         # Interpolate to match `target_change`
#         #positives_model_da = positives_model_da.interp_like(target_change)
#         #negatives_model_da = negatives_model_da.interp_like(target_change)

#     import matplotlib as mpl
#     mpl.rcParams['hatch.linewidth'] = 0.5
#     mpl.rcParams['hatch.color'] = 'black'

#     # Initialize the plot
#     fig, ax = plt.subplots(figsize=(15, 20), subplot_kw={'projection': ccrs.PlateCarree()})
#     ax.set_extent(extent, crs=ccrs.PlateCarree())
#     ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='black', linewidth=0.7)
#     ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle='--', edgecolor='gray')

#     # Plot `target_change` as the main contour
#     target_change.plot.contourf(
#         ax=ax,
#         transform=ccrs.PlateCarree(),
#         cmap='PuOr',
#         levels=20,
#         add_colorbar=True,
#         cbar_kwargs={'shrink': 0.7, 'label': 'Precipitation Change (mm/day)', "fraction": 0.03, "pad": 0.02,}
#     )

#     # Drawing arcs for the specified regions
#     for extent in region_extents:
#         arc = create_arc(extent[2], extent[3], extent[0], extent[1])  # assuming extents are (lat_min, lat_max, lon_min, lon_max)
#         ax.add_geometries([arc], crs=ccrs.PlateCarree(), edgecolor='blue', facecolor='none', linewidth=2)

#     # Add stippling for `p_values` (significance)
#     if p_values is not None and p_values.size > 0:
#         print("Adding p_values stippling...")
#         sig_mask = p_values < sig_level
#         ax.contourf(
#             p_values['lon'], p_values['lat'], sig_mask,
#             levels=[0.5, 1], hatches=['o', ''], colors='none',
#             transform=ccrs.PlateCarree()
#         )

#     # Add stippling for `positives_model` and `negatives_model`
#     if positives_model is not None and negatives_model is not None:
#         print("Adding positives/negatives stippling...")
#         combined = (positives_model_da + negatives_model_da) == sig
#         ax.contourf(
#             positives_model_da['lon'], positives_model_da['lat'], combined,
#             levels=[0.5, 1], hatches=['.', ''], colors='none',
#             transform=ccrs.PlateCarree()
#         )

#     # Add title and gridlines
#     ax.set_title("End of century changes in winter (NDJFM) precipitation in CMIP6 high-emission scenario")
#     gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
#     gl.top_labels = gl.right_labels = False

#     return fig

def plot_function(target_change, p_values, positives_model, negatives_model, region_extents,
                  sig_level=0.05, sig=1, map_extent=None, projection='platecarree'):
    import cartopy.feature as cfeature
    import matplotlib.lines as mlines
    import matplotlib as mpl
    import numpy as np
 
    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "figure.titlesize": 20
    })
 
    # Determine extent
    if map_extent is not None:
        extent = map_extent                          # user-specified
    elif 'region_extent' in target_change.attrs:
        extent = target_change.attrs['region_extent']
    else:
        extent = [-180, 180, -90, 90]
 
    target_change = target_change.sel(
        lon=slice(extent[0], extent[1]),
        lat=slice(extent[2], extent[3])
    )
    extent = [
        float(target_change['lon'].min()), float(target_change['lon'].max()),
        float(target_change['lat'].min()), float(target_change['lat'].max())
    ]
 
    print(f"extent used: {extent}")
 
    # Align p_values
    if p_values is not None:
        p_values = p_values.sel(
            lon=slice(extent[0], extent[1]),
            lat=slice(extent[2], extent[3])
        )
 
    # Align beta masks
    if positives_model is not None and negatives_model is not None:
        positives_model_da = positives_model['pr'].sel(
            lon=slice(extent[0], extent[1]),
            lat=slice(extent[2], extent[3])
        )
        negatives_model_da = negatives_model['pr'].sel(
            lon=slice(extent[0], extent[1]),
            lat=slice(extent[2], extent[3])
        )
 
    # Plot
    if projection == 'robinson':
        proj = ccrs.Robinson(central_longitude=0)
        figsize = (18, 10)
    elif projection == 'platecarree':
        proj = ccrs.PlateCarree()
        figsize = (15, 10)
    else:
        raise ValueError(f"projection must be 'robinson' or 'platecarree', "
                        f"got '{projection}'")
 
    fig, ax = plt.subplots(figsize=figsize,
                        subplot_kw={'projection': proj},
                        constrained_layout=True)
 
    # Robinson shows full globe — no set_extent needed
    if projection == 'platecarree' and map_extent is not None:
        ax.set_extent(map_extent, crs=ccrs.PlateCarree())
    elif projection == 'platecarree':
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'),
                   edgecolor='black', linewidth=0.7)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'),
                   linestyle='--', edgecolor='gray')
 
    target_change.plot.contourf(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='PuOr',
        levels=20,
        add_colorbar=True,
        cbar_kwargs={
            'shrink': 0.7,
            'label': 'Precipitation Change (mm/day)',
            'fraction': 0.03,
            'pad': 0.02,
        }
    )
 
    # Region boxes
    for ext in region_extents:
        arc = create_arc(ext[2], ext[3], ext[0], ext[1])
        ax.add_geometries([arc], crs=ccrs.PlateCarree(),
                          edgecolor='blue', facecolor='none', linewidth=2)
 
    legend_handles = []
 
    if positives_model is not None and negatives_model is not None \
            and p_values is not None:
 
        # Beta mask — models agree on sign
        beta_mask  = (positives_model_da + negatives_model_da) == sig
 
        # Gamma mask — signal large relative to noise
        gamma_mask = p_values < sig_level
 
        # Align gamma to beta grid if needed
        if not (gamma_mask['lon'].equals(positives_model_da['lon']) and
                gamma_mask['lat'].equals(positives_model_da['lat'])):
            gamma_mask = gamma_mask.astype(float).interp_like(positives_model_da)
            gamma_mask = gamma_mask > 0.5

        print("lon values first 5:", positives_model_da['lon'].values[:5])
        print("lon values last 5:", positives_model_da['lon'].values[-5:])
 
        lon2d, lat2d = np.meshgrid(
            positives_model_da['lon'].values,
            positives_model_da['lat'].values
        )
 
        # Robust: beta AND gamma → filled dots
        robust = (beta_mask & gamma_mask).values
        if robust.any():
            ax.scatter(
                lon2d[robust], lat2d[robust],
                color='black', s=8,
                transform=ccrs.PlateCarree(), zorder=5,
            )
            legend_handles.append(
                mlines.Line2D([0], [0], marker='.', color='black',
                              linestyle='None', markersize=6,
                              label='Robust (β ≥ 90% and γ > 1)')
            )
 
        # Large but non-robust: gamma only → open circles
        large_nonrobust = (~beta_mask & gamma_mask).values
        if large_nonrobust.any():
            ax.scatter(
                lon2d[large_nonrobust], lat2d[large_nonrobust],
                s=18, facecolors='none', edgecolors='black', linewidths=0.8,
                transform=ccrs.PlateCarree(), zorder=5,
            )
            legend_handles.append(
                mlines.Line2D([0], [0], marker='o', color='black',
                              linestyle='None', markersize=6,
                              markerfacecolor='none',
                              label='Large but non-robust (γ > 1 only)')
            )
 
    elif positives_model is not None and negatives_model is not None:
        # Beta only — filled dots
        combined = (positives_model_da + negatives_model_da) == sig
        lon2d, lat2d = np.meshgrid(
            positives_model_da['lon'].values,
            positives_model_da['lat'].values
        )
        ax.scatter(
            lon2d[combined.values], lat2d[combined.values],
            color='black', s=3,
            transform=ccrs.PlateCarree(), zorder=5,
        )
        legend_handles.append(
            mlines.Line2D([0], [0], marker='.', color='black',
                          linestyle='None', markersize=6,
                          label='Robust response (β ≥ 90%)')
        )
 
    elif p_values is not None and p_values.size > 0:
        # Gamma only — open circles
        gamma_mask = p_values < sig_level
        lon2d, lat2d = np.meshgrid(
            p_values['lon'].values,
            p_values['lat'].values
        )
        ax.scatter(
            lon2d[gamma_mask.values], lat2d[gamma_mask.values],
            s=16, facecolors='none', edgecolors='black', linewidths=0.8,
            transform=ccrs.PlateCarree(), zorder=5,
        )
        legend_handles.append(
            mlines.Line2D([0], [0], marker='o', color='black',
                          linestyle='None', markersize=6,
                          markerfacecolor='none',
                          label='Large signal (γ > 1)')
        )
 
    if legend_handles:
        ax.legend(handles=legend_handles, loc='lower left',
                  fontsize=12, framealpha=0.8)
 
    ax.set_title(
        "End of century changes in winter (NDJFM) precipitation "
        "in CMIP6 high-emission scenario"
    )
    gl = ax.gridlines(draw_labels=True, linewidth=0.5,
                      color='gray', alpha=0.7, linestyle='--')
    gl.top_labels = gl.right_labels = False
 
    return fig

def plot_precipitation_change(target_change, region_extents, years, var_name):
    """
    Plots precipitation changes for multiple regions.

    Parameters:
    - target_change : list of xarray.DataArrays
        List of precipitation change data for each model.
    - region_extents : list of tuples
        Each tuple contains the lat_min, lat_max, lon_min, and lon_max for a region.
    - years : np.ndarray
        Array of years corresponding to the time series data.
    """

    # Helper function to extract and average data for a specific region
    def extract_region_data(target_change, region_extent):
        region_data = []
        for model in target_change:
            model_region = model.sel(
                lon=slice(region_extent[2], region_extent[3]),
                lat=slice(region_extent[0], region_extent[1])
            )
            # Calculate the baseline mean for 1960-1990
            baseline = model_region.sel(time=slice(1960, 1990)).mean(dim='time')
            # Calculate the anomaly by subtracting the baseline from the entire series
            anomaly = model_region - baseline
            # Average over the spatial dimensions
            region_data.append(anomaly.mean(dim=['lat', 'lon']))
        return region_data

    # Create the figure with subplots
    num_regions = len(region_extents)
    fig, axes = plt.subplots(1, num_regions, figsize=(6 * num_regions, 4), sharey=True)

    if num_regions == 1:  # To handle the case of a single subplot
        axes = [axes]

    for i, region_extent in enumerate(region_extents):
        data_region = extract_region_data(target_change, region_extent)

        ax = axes[i]
        for model_data in data_region:
            rolling_mean = model_data.rolling(time=30, center=True, min_periods=1).mean()

            if isinstance(rolling_mean, xr.Dataset):
                rolling_mean = rolling_mean.to_dataarray()

            if rolling_mean.time.size != len(years):
                rolling_mean = rolling_mean.interp(time=years)
            ax.plot(years, rolling_mean.squeeze().values, alpha=0.3, linewidth=0.8)  # Plot the rolling mean for each model

        # Calculate and plot the model mean (thick black line)
        model_mean = xr.concat(data_region, dim='model', coords='minimal', compat='override').mean(dim='model')
        rolling_mean = model_mean.rolling(time=30, center=True, min_periods=1).mean()

        if isinstance(rolling_mean, xr.Dataset):
                rolling_mean = rolling_mean.to_dataarray()

        if rolling_mean.time.size != len(years):
            rolling_mean = rolling_mean.interp(time=years)
        ax.plot(years, rolling_mean.squeeze().values, color='black', linewidth=2, label='Model Mean')

        ax.set_title(f"Region {i+1}", fontsize=12)
        ax.set_ylabel(f"{var_name} change [mm day$^{-1}$]" if i == 0 else "")
        ax.set_xlabel("Years")
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.legend()

    plt.tight_layout()
    return fig