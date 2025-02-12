from .utils import np, xr, plt, ccrs, gridspec, cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm
from cartopy.util import add_cyclic_point
import matplotlib.colors as mcolors
import matplotlib as mpl

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

def make_symmetric_colorbar(plot_range, num_steps=18):
    """Create arrays with color and tick levels that can be used as arguments
    in the Matplotlib contourf() function and the colorbar() function,
    respectively and that ensure symmetric color levels and colorbar ticks.
    
    Parameters
    ----------
    plot_range : numeric (int or float)
        Value range that should be covered by contour plot.
    
    num_steps : int
        Number of color levels. One level will be added if number is odd. Will
        be converted to integer if number is float.
        
    Returns
    -------
    (color_levels, tick_levels) : tuple
        Tuple containing numpy-array with color levels and numpy array with tick levels.
    """
    num_steps = int(num_steps) # convert to int if float is given
    if num_steps % 2 != 0:
        num_steps = num_steps + 1 # add one level in case odd number is given

    inc_color = (plot_range*2)/num_steps # calculate increment
    color_levels = np.arange(-plot_range-inc_color/2, plot_range+inc_color/2+plot_range/1000, inc_color)
    
    inc_ticks = inc_color
    tick_levels = np.arange(-plot_range, plot_range+plot_range/1000, inc_ticks*2)
    return color_levels, tick_levels


def create_three_panel_figure(data_list, extent_list, levels_list, cmaps_list, titles, colorbar_label='Colorbar Label', figsize=(15, 5), mask_range=(-0.02, 0.02)):
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

        data_cyclic, lon_cyclic = add_cyclic_point(data_list[i].values, coord=data_list[i].lon)

        # Mask values close to zero
        masked_data = np.ma.masked_inside(data_cyclic, mask_range[0], mask_range[1])
        
        # Plot the data (assuming `data_list` contains 2D arrays or DataArrays)
        norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=levels_list[i][0], vmax=levels_list[i][-1])
        im = ax.contourf(lon_cyclic, data_list[i].lat, masked_data,
                         levels=levels_list[i], cmap=cmaps_list[i], norm=norm, transform=ccrs.PlateCarree())
        
        # Set the title for each subplot
        ax.set_title(titles[i], fontsize=12)
    
    # Add a single colorbar at the bottom of the plots
    cbar = fig.colorbar(im, ax=axs, orientation='horizontal', fraction=0.05, pad=0.05)
    cbar.set_label(colorbar_label)
    
    # Show the figure
    plt.show()


# def create_five_panel_figure(map_data, extents, levels, colormaps, titles, colorbar_label='Colorbar Label', white_range=(-0.05, 0.05), mask_range=(-0.05, 0.05)):
#     """
#     Creates a figure with five panels: one in the center and four around it (at the corners).
#     A single colorbar is added below all panels.

#     Parameters:
#         map_data (list): A list of 5 data arrays to be plotted as maps.
#         extents (list): A list of tuples for map extents [(lon_min, lon_max, lat_min, lat_max), ...].
#         levels (list): A list of level arrays for contourf or pcolormesh.
#         colormaps (list): A list of colormaps to use for each map.
#         titles (list): A list of titles for the subplots.
#         white_range (tuple): The range of values to make white (min, max).

#     Returns:
#         fig: The created matplotlib figure.
#     """
#     # Create the figure and GridSpec layout
#     fig = plt.figure(figsize=(12, 7))
#     gs = gridspec.GridSpec(3, 3, figure=fig, wspace=0.02, hspace=0.02)  # Reduced spacing
    
#     # Define subplot positions for the maps
#     subplot_positions = [(0, 0), (0, 2), (2, 0), (2, 2), (1, 1)]  # Corners and center
    
#     # Keep track of the mappable objects for the colorbar
#     mappable = None
    
#     for i, pos in enumerate(subplot_positions):
#         # Add a GeoAxes at the specified position
#         ax = fig.add_subplot(gs[pos[0], pos[1]], projection=ccrs.PlateCarree())
#         lon_min, lon_max, lat_min, lat_max = extents[i]
        
#         # Set map extent
#         ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
#         # Add map features
#         ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
#         ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--')
#         ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        
#         # Extract the data
#         data = map_data[i]
        
#         ############################################################################################
#         # Custom colormap for white space
#         # cmapU850 = mpl.colors.ListedColormap(['#8b4513', '#e65c00', '#ff8c00', '#ffa500',
#         #                        '#f5deb3', 'white', 'white', '#fff5ee',
#         #                        '#eed2ee', '#d8bfd8', '#9a32cd', '#6a0dad'])
        
#         # cmapU850.set_over('maroon')
#         # cmapU850.set_under('midnightblue')
#         ############################################################################################

#         # colors = ["saddlebrown", "white", "rebeccapurple"]
#         # cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)

#         # cmap.set_over('darkslateblue')  # Darker shade of purple for high out-of-bound values
#         # cmap.set_under('darkgoldenrod')  # Darker shade of brown for low out-of-bound values

#         # clevs = np.linspace(-0.3, 0.3, 13)  # Levels centered around zero
#         # norm = mcolors.CenteredNorm(vcenter=0, halfrange=0.3)
#         data_min = data.min().values
#         data_max = data.max().values
#         plot_range = max(abs(data_min), abs(data_max))
#         num_steps = 12

#         # Create a custom colormap
#         base_cmap = plt.get_cmap(colormaps[i])
#         colors = base_cmap(np.linspace(0, 1, 256))
        
#         # Mask values within the white range
#         white_min, white_max = white_range
#         white_mask = (levels[i] >= white_min) & (levels[i] <= white_max)
#         for j in range(len(levels[i]) - 1):
#             if white_mask[j]:
#                 colors[j, :] = [1, 1, 1, 1]  # Set white color for the range
        
#         custom_cmap = ListedColormap(colors)

#         data_cyclic, lon_cyclic = add_cyclic_point(data.values, coord=data.lon)

#         ############################################################################################
#         ######## Using a mask ##########
#         # original_cmap = plt.get_cmap(colormaps[i])
#         # colors = original_cmap(np.linspace(0, 1, 256))
#         # min_col = np.min(data_cyclic)
#         # max_col = np.max(data_cyclic)
#         # mask_min_idx = int((mask_range[0] - min_col) / (max_col - min_col) * 256)
#         # mask_max_idx = int((mask_range[1] - min_col) / (max_col - min_col) * 256)
#         # colors[mask_min_idx:mask_max_idx, :] = [1, 1, 1, 1]  # RGBA for white
#         # modified_cmap = ListedColormap(colors)
#         ############################################################################################
#         # Plot the data
#         norm = BoundaryNorm(levels[i], ncolors=modified_cmap.N, clip=True)
#         # data_cyclic, lon_cyclic = add_cyclic_point(data.values, coord=data.lon)

#         #masked_data = np.ma.masked_inside(data_cyclic, mask_range[0], mask_range[1])

#         #norm = mcolors.CenteredNorm(vcenter=0, halfrange=0.05)
#         im = ax.contourf(lon_cyclic, data.lat, data_cyclic, levels=levels[i], cmap=modified_cmap, norm=norm, extend='both', transform=ccrs.PlateCarree())
        
#         # Set the title for each subplot
#         ax.set_title(titles[i], fontsize=10, pad=4)
        
#         # Keep the last plotted mappable object for the shared colorbar
#         if i == 4:  # Use the central plot's mappable for the colorbar
#             mappable = im

#     # Add a single colorbar below all plots
#     if mappable:
#         cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.02])  # [left, bottom, width, height]
#         cbar = fig.colorbar(mappable, cax=cbar_ax, orientation='horizontal')
#         cbar.set_label(colorbar_label)  # Add your label here

#     return fig

'''
def create_three_panel_figure(data_list, extent_list, levels_list, cmaps_list, titles, colorbar_label='Colorbar Label', figsize=(15, 5)):
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
    
    fig, axs = plt.subplots(1, 3, figsize=figsize, dpi=300, constrained_layout=True,
                            subplot_kw={'projection': ccrs.PlateCarree()})

    for i, ax in enumerate(axs):
        ax.set_extent(extent_list[i], crs=ccrs.PlateCarree())
        ax.coastlines()
        ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--')

        data = data_list[i]
        plot_range = max(abs(data.min()), abs(data.max()))
        color_levels, tick_levels = make_symmetric_colorbar(plot_range, num_steps=12)
        
        norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-plot_range, vmax=plot_range)
        data_cyclic, lon_cyclic = add_cyclic_point(data.values, coord=data.lon)
        
        im = ax.contourf(lon_cyclic, data.lat, data_cyclic,
                         levels=color_levels, cmap=cmaps_list[i], norm=norm, transform=ccrs.PlateCarree())
        
        ax.set_title(titles[i], fontsize=12)

    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal', fraction=0.05, pad=0.05, ticks=tick_levels)
    cbar.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
    cbar.set_label(colorbar_label)
    
    plt.show()
'''

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from cartopy import crs as ccrs, feature as cfeature
from cartopy.util import add_cyclic_point
import numpy as np
from matplotlib.ticker import FuncFormatter

def create_five_panel_figure(map_data, extents, levels, colormaps, titles, colorbar_label='Colorbar Label'):
    fig = plt.figure(figsize=(12, 7))
    gs = gridspec.GridSpec(3, 3, figure=fig, wspace=0.02, hspace=0.02)

    for i, pos in enumerate([(0, 0), (0, 2), (2, 0), (2, 2), (1, 1)]):
        ax = fig.add_subplot(gs[pos[0], pos[1]], projection=ccrs.PlateCarree())
        ax.set_extent(extents[i], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
        ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=0.5)
        ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        
        data = map_data[i]
        plot_range = max(abs(data.min()), abs(data.max()))
        color_levels, tick_levels = make_symmetric_colorbar(plot_range, num_steps=12)
        
        original_cmap = plt.get_cmap(colormaps[i])
        shifted_cmap = original_cmap(np.linspace(0, 1, len(color_levels)))
        mid_index = len(color_levels) // 2
        shifted_cmap[mid_index - 1:mid_index + 1] = [1, 1, 1, 1]
        new_cmap = mcolors.ListedColormap(shifted_cmap)
        
        norm = mcolors.TwoSlopeNorm(vmin=-plot_range, vcenter=0, vmax=plot_range)
        data_cyclic, lon_cyclic = add_cyclic_point(data.values, coord=data.lon)
        im = ax.contourf(lon_cyclic, data.lat, data_cyclic, levels=color_levels, cmap=new_cmap, norm=norm, transform=ccrs.PlateCarree())
        
        ax.set_title(titles[i], fontsize=10, pad=4)
        
    if im:
        cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.02])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', ticks=tick_levels)
        cbar.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}'))  # Formatting to 1 decimal place
        cbar.set_label(colorbar_label)

    plt.show()



def hemispheric_plot(data, levels, extent, cmap, title,
              central_longitude=0, central_latitude=90, colorbar_label='Colorbar Label'):
    """
    Plot data using a stereographic projection.

    Args:
    - data (xarray.DataArray): The data to plot.
    - levels (np.ndarray): Contour levels for the plot.
    - cmap (str): Colormap for the plot.
    - title (str): Title of the plot.
    - extent (list): Geographic extent for the plot [lon_min, lon_max, lat_min, lat_max].
    - projection (ccrs.Projection): Cartopy projection for the plot.
    - central_longitude (float): Central longitude for the Stereographic projection.
    - central_latitude (float): Central latitude for the Stereographic projection.
    - colorbar_label (str): Label for the colorbar.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection=ccrs.Stereographic(central_longitude=central_longitude, central_latitude=central_latitude))
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    data_cyclic, lon_cyclic = add_cyclic_point(data.values, coord=data.lon)
    im = ax.contourf(lon_cyclic, data.lat, data_cyclic, levels=levels, cmap=cmap, transform=ccrs.PlateCarree())
    cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(colorbar_label)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.gridlines(draw_labels=True)
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
    ax.add_feature(cfeature.LAKES, edgecolor='black', facecolor='lightblue')
    ax.set_title(title, fontsize=14)
    plt.show()