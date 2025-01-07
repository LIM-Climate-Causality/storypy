from .utils import np, xr, plt, ccrs

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

def plot_precipitation_change_two_regions(model_changes, region_a_extent, region_b_extent, years):
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

    # Helper function to calculate mean, ensemble spread, and confidence intervals
    def calculate_region_stats(model_changes, region_extent):
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
    data_a = calculate_region_stats(model_changes, region_a_extent)
    data_b = calculate_region_stats(model_changes, region_b_extent)

    # Ensure `years` aligns with the shape of the computed means
    if len(years) != len(data_a['mean']) or len(years) != len(data_b['mean']):
        raise ValueError(
            f"Mismatch between years ({len(years)}) and computed data (Region A: {len(data_a['mean'])}, "
            f"Region B: {len(data_b['mean'])})."
        )

    # Calculate regression and R/p-values
    r_value_a, p_value_a = linregress(years, data_a['mean'])[2:4]
    r_value_b, p_value_b = linregress(years, data_b['mean'])[2:4]

    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    colors = {'mean': 'blue', 'ci': 'red', 'ensemble': 'gray'}

    # Panel A (Region A)
    ax = axes[0]
    ax.plot(years, data_a['ensemble'].T, color=colors['ensemble'], alpha=0.3, linewidth=0.8)
    ax.plot(years, data_a['mean'], color=colors['mean'], label='Mean')
    ax.fill_between(years, data_a['lower'], data_a['upper'], color=colors['ci'], alpha=0.2, label='Confidence Interval')
    ax.text(years[0] + 5, max(data_a['upper']) - 0.2, f"R = {r_value_a:.2f} (p = {p_value_a:.3f})", fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7))
    ax.set_title("Region A", fontsize=12)
    ax.set_ylabel("pr change [mm day$^{-1}$]")
    ax.set_xlabel("Years")
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.legend()

    # Panel B (Region B)
    ax = axes[1]
    ax.plot(years, data_b['ensemble'].T, color=colors['ensemble'], alpha=0.3, linewidth=0.8)
    ax.plot(years, data_b['mean'], color=colors['mean'], label='Mean')
    ax.fill_between(years, data_b['lower'], data_b['upper'], color=colors['ci'], alpha=0.2, label='Confidence Interval')
    ax.text(years[0] + 5, max(data_b['upper']) - 0.2, f"R = {r_value_b:.2f} (p = {p_value_b:.3f})", fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7))
    ax.set_title("Region B", fontsize=12)
    ax.set_xlabel("Years")
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.legend()

    # Add subplot labels
    axes[0].text(-0.1, 1.05, "a", transform=axes[0].transAxes, fontsize=14, fontweight='bold')
    axes[1].text(-0.1, 1.05, "b", transform=axes[1].transAxes, fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def create_three_panel_figure(data_list, extent_list, levels_list, cmaps_list, titles, figsize=(15, 5)):
    """
    Creates a figure with three panels in a row.
    
    Parameters:
    - data_list: List of 2D arrays or datasets to plot.
    - extent_list: List of extents for each map [lon_min, lon_max, lat_min, lat_max].
    - levels_list: List of levels for contour plots.
    - cmaps_list: List of colormap names for each map.
    - titles: Titles for each subplot.
    - figsize: Size of the overall figure (default is (15, 5)).
    """
    
    # Create the figure and use a specific projection for Cartopy maps
    fig, axs = plt.subplots(1, 3, figsize=figsize, dpi=300, constrained_layout=True,
                            subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Iterate through each map and plot it
    for i, ax in enumerate(axs):
        # Set the extent for each map using the provided extents
        ax.set_extent(extent_list[i], crs=ccrs.PlateCarree())
        
        # Add coastlines and gridlines
        ax.coastlines()
        ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--')
        
        # Plot the data (assuming `data_list` contains 2D arrays or DataArrays)
        im = ax.contourf(data_list[i].lon, data_list[i].lat, data_list[i].values,
                         levels=levels_list[i], cmap=cmaps_list[i], transform=ccrs.PlateCarree())
        
        # Set the title for each subplot
        ax.set_title(titles[i], fontsize=12)
    
    # Add a single colorbar at the bottom of the plots
    fig.colorbar(im, ax=axs, orientation='horizontal', fraction=0.05, pad=0.05)
    
    # Show the figure
    plt.show()