from storypy.utils import np, xr, plt, ccrs, gridspec, cfeature, pd
from ._plotting import create_arc
from matplotlib.colors import ListedColormap, BoundaryNorm
from cartopy.util import add_cyclic_point
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib.patches import Ellipse
from sklearn.linear_model import LinearRegression
import matplotlib.transforms as transforms
from numpy import linalg as la

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


def confidence_ellipse(x ,y, ax, corr,chi_squared=3.21, facecolor='none',**kwargs):
 if x.size != y.size:
  raise ValueError('x and y must be the same size')

 cov = np.cov(x,y)
 pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
 eigval, eigvec = la.eig(cov)
 largest_eigval = np.argmax(eigval)
 largest_eigvec = eigvec[:,largest_eigval]
 smallest_eigval = np.argmin(eigval)
 smallest_eigvec = eigvec[:,smallest_eigval]
 lamda1 = np.max(eigval)
 lamda2 = np.min(eigval)

 scale_x = np.sqrt(lamda1)
 scale_y = np.sqrt(lamda2)
 if corr == 'no':
    angle = 90.0 #np.arctan(smallest_eigvec[0]/smallest_eigvec[1])*180/np.pi
 else:
    angle = np.arctan(smallest_eigvec[0]/smallest_eigvec[1])*180/np.pi

 # Using a special case to obtain the eigenvalues of this
 # two-dimensionl dataset. Calculating standard deviations

 ell_radius_x = scale_x*np.sqrt(chi_squared)
 ell_radius_y = scale_y*np.sqrt(chi_squared)
 ellipse = Ellipse((0, 0), width=ell_radius_x * 2,height=ell_radius_y * 2, angle = -angle, facecolor=facecolor,**kwargs)

 # Calculating x mean
 mean_x = np.mean(x)
 # calculating y mean
 mean_y = np.mean(y)

 transf = transforms.Affine2D() \
     .translate(mean_x, mean_y)

 ellipse.set_transform(transf + ax.transData)
 return ax.add_patch(ellipse), print(angle), ellipse


def plot_ellipse(models,x,y,corr='no',x_label='Eastern Pacific Warming [K K$^{-1}$]',y_label='Central Pacific Warming [K K$^{-1}$]'):
    #Compute regression y on x
    x1 = x.reshape(-1, 1)
    y1 = y.reshape(-1, 1)
    linear_regressor = LinearRegression()  # create object for the class
    reg = linear_regressor.fit(x1, y1)  # perform linear regression
    X_pred = np.linspace(np.min(x)-1, np.max(x)+0.5, 31)
    X_pred = X_pred.reshape(-1, 1)
    Y_pred = linear_regressor.predict(X_pred)  # make predictions
    c = reg.coef_

    #Compute regression x on y
    reg2 = linear_regressor.fit(y1, x1)  # perform linear regression
    Y_pred2 = np.linspace(np.min(y), np.max(y), 31)
    Y_pred2 = Y_pred2.reshape(-1, 1)
    X_pred2 = linear_regressor.predict(Y_pred2)  # make predictions
    c2 = reg2.coef_

    #Define limits
    min_x = np.min(x) - 0.2*np.abs(np.max(x) - np.min(x))
    max_x = np.max(x) + 0.2*np.abs(np.max(x) - np.min(x))
    max_y = np.max(y) + 0.2*np.abs(np.max(y) - np.min(y))
    max_y = np.min(y) - 0.2*np.abs(np.max(y) - np.min(y))
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    #Calcular las rectas x = y, x = -y
    Sx = np.std(x)
    Sy = np.std(y)
    S_ratio = Sy/Sx
    YeqX = S_ratio*X_pred - S_ratio*mean_x + mean_y
    YeqMinsX = S_ratio*mean_x + mean_y - S_ratio*X_pred


    #Plot-----------------------------------------------------------------------
    markers = ['<','<','v','*','D','x','x','p','+','+','d','8','X','X','^','d','d','1','2','>','>','D','D','s','.','P', 'P', '3','4','h','H', '>','X','s','o','o',]
    print(models)
    fig, ax = plt.subplots()
    for px, py, t, l in zip(x, y, markers, models):
       ax.scatter(px, py, marker=t,label=l)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
    confidence_ellipse(x, y, ax,corr,edgecolor='red',label=r'80 $\%$ confidence region')
    confidence_ellipse(x, y, ax,corr,chi_squared=4.6,edgecolor='k',linestyle='--',alpha=0.5,label=r'$\pm$ 10 $\%$ confidence regions')
    confidence_ellipse(x, y, ax,corr,chi_squared=2.4,edgecolor='k',linestyle='--',alpha=0.5)
    ax.axvline(mean_x, c='grey', lw=1)
    ax.axhline(mean_y, c='grey', lw=1)
    ax.grid()
    ax.tick_params(labelsize=18)
    if corr == 'yes':
        r = np.corrcoef(x,y)[0,1]; chi = (1.26**2)*2
        ts1 = np.sqrt(((1-r**2)/(2*(1-r)))*chi)
        ts2 = np.sqrt(((1-r**2)/(2*(1+r)))*chi)
        story_x1 = [mean_x + ts1*np.std(x)]
        story_x2 = [mean_x - ts1*np.std(x)]
        story_y_red1 = [mean_y + ts1*np.std(y)]
        story_y_red2 =[mean_y - ts1*np.std(y)]
        ax.plot(story_x1, story_y_red1, 'ro',alpha = 0.6,markersize=10,label='storylines')
        ax.plot(story_x2, story_y_red2, 'ro',alpha = 0.6,markersize=10)
    elif corr == 'ma':
        r = np.corrcoef(x,y)[0,1]; chi = (1.26**2)*2
        ts1 = np.sqrt(((1-r**2)/(2*(1-r)))*chi)
        ts2 = np.sqrt(((1-r**2)/(2*(1+r)))*chi)
        story_x1 = [mean_x + ts1*np.std(x)]
        story_x2 = [mean_x - ts1*np.std(x)]
        story_y_red1 = [mean_y + ts1*np.std(y)]
        story_y_red2 =[mean_y - ts1*np.std(y)]
        ax.plot(story_x1, story_y_red1, 'ro',alpha = 0.6,markersize=10,label='storylines')
        ax.plot(story_x2, story_y_red2, 'ro',alpha = 0.6,markersize=10) 
    elif corr == 'pacific':
        r = np.corrcoef(x,y)[0,1]; chi = (1.26**2)*2
        ts1 = np.sqrt(((1-r**2)/(2*(1-r)))*chi)
        ts2 = np.sqrt(((1-r**2)/(2*(1+r)))*chi)
        story_x1 = [mean_x + ts1*np.std(x)]
        story_x2 = [mean_x - ts1*np.std(x)]
        story_y_red1 = [mean_y + ts1*np.std(y)]
        story_y_red2 =[mean_y - ts1*np.std(y)]
        ax.plot(story_x2, story_y_red2, 'bo',alpha = 0.6,markersize=10,label='Low asym Pacific Warming')
        ax.plot(story_x1, story_y_red1, 'ro',alpha = 0.6,markersize=10,label='High asym Pacific Warming')  
    elif corr == 'nada':
        r = np.corrcoef(x,y)[0,1]; chi = (1.26**2)*2
    else:
        story_x = [mean_x + 1.26*np.std(x),mean_x - 1.26*np.std(x)]
        story_y_red = [mean_y + 1.26*np.std(y),mean_y - 1.26*np.std(y)]
        story_y_blue =[mean_y - 1.26*np.std(y),mean_y + 1.26*np.std(y)]
        ax.plot(story_x, story_y_red, 'ro',alpha = 0.6,markersize=10,label='High asym Pacific Warming')
        ax.plot(story_x, story_y_blue, 'bo',alpha = 0.6,markersize=10,label='Low asym Pacific Warming')    
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=5)
    plt.subplots_adjust(bottom=0.05)
    plt.xlabel(x_label,fontsize=18)
    plt.ylabel(y_label,fontsize=18)
    plt.title('R='+str(round(np.corrcoef(x,y)[0,1],3)))
    #plt.clf
    return fig