# from esmvaltool.diag_scripts.shared import group_metadata
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import gridspec
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from numpy import linalg as la
from sklearn.linear_model import LinearRegression
import pandas as pd

def model_change(config,var,model):
    meta = group_metadata(config["input_data"].values(), "dataset")
    for dataset, dataset_list in meta.items():
        target = [xr.open_dataset(m["filename"])[m["short_name"]].mean(dim='time') for m in dataset_list if (m["variable_group"] == var) & (m["dataset"] == model)]

    if len(target) == 0:
        print("No data for this variable or model")
    else:
        out = target[0]
    return out


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


def create_five_panel_figure(map_data, extents, levels, colormaps, titles, white_range=(-0.05, 0.05)):
    """
    Creates a figure with five panels: one in the center and four around it (at the corners).
    A single colorbar is added below all panels.

    Parameters:
        map_data (list): A list of 5 data arrays to be plotted as maps.
        extents (list): A list of tuples for map extents [(lon_min, lon_max, lat_min, lat_max), ...].
        levels (list): A list of level arrays for contourf or pcolormesh.
        colormaps (list): A list of colormaps to use for each map.
        titles (list): A list of titles for the subplots.
        white_range (tuple): The range of values to make white (min, max).

    Returns:
        fig: The created matplotlib figure.
    """
    # Create the figure and GridSpec layout
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig, wspace=0.02, hspace=0.02)  # Reduced spacing
    
    # Define subplot positions for the maps
    subplot_positions = [(0, 0), (0, 2), (2, 0), (2, 2), (1, 1)]  # Corners and center
    
    # Keep track of the mappable objects for the colorbar
    mappable = None
    
    for i, pos in enumerate(subplot_positions):
        # Add a GeoAxes at the specified position
        ax = fig.add_subplot(gs[pos[0], pos[1]], projection=ccrs.PlateCarree())
        lon_min, lon_max, lat_min, lat_max = extents[i]
        
        # Set map extent
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--')
        ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        
        # Extract the data
        data = map_data[i]
        
        # Create a custom colormap
        base_cmap = plt.get_cmap(colormaps[i])
        colors = base_cmap(np.linspace(0, 1, 256))
        
        # Mask values within the white range
        white_min, white_max = white_range
        white_mask = (levels[i] >= white_min) & (levels[i] <= white_max)
        for j in range(len(levels[i]) - 1):
            if white_mask[j]:
                colors[j, :] = [1, 1, 1, 1]  # Set white color for the range
        
        custom_cmap = ListedColormap(colors)
        
        # Plot the data
        norm = BoundaryNorm(levels[i], ncolors=custom_cmap.N, clip=True)
        im = ax.contourf(data.lon, data.lat, data, levels=levels[i], cmap=custom_cmap, norm=norm, transform=ccrs.PlateCarree())
        
        # Set the title for each subplot
        ax.set_title(titles[i], fontsize=10, pad=4)
        
        # Keep the last plotted mappable object for the shared colorbar
        if i == 4:  # Use the central plot's mappable for the colorbar
            mappable = im

    # Add a single colorbar below all plots
    if mappable:
        cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.02])  # [left, bottom, width, height]
        cbar = fig.colorbar(mappable, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Colorbar Label')  # Add your label here

    return fig
