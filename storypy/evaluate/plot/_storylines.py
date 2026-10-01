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
import matplotlib.path as mpath
import matplotlib.ticker as mticker
import os
from matplotlib.ticker import FuncFormatter
import math

def bivariate_dist(confidence_level=0.8, n_drivers=2, r=0.0):
    """
    Compute storyline coefficients following Zappa & Shepherd (2017)
    and Monerie et al. (2023).

    When r=0 (independent drivers), returns a single coefficient i
    equivalent to Zappa & Shepherd (2017).

    When r≠0 (correlated drivers), returns two coefficients following
    Monerie et al. (2023):
    - i1: for co-varying storylines (A+M+ or A-M-)
    - i2: for opposing storylines (A+M- or A-M+)

    Parameters
    ----------
    confidence_level : float
        Confidence level in [0, 1]. Default 0.8.
    n_drivers : int
        Degrees of freedom. Default 2.
    r : float
        Correlation coefficient between the two drivers.
        Default 0.0 reproduces Zappa & Shepherd (2017).

    Returns
    -------
    float or tuple(float, float)
        If r=0: single float coefficient (as before).
        If r≠0: tuple (i1, i2) where i1 is for same-direction
        storylines and i2 for opposing storylines.

    Examples
    --------
    >>> bivariate_dist(0.8, 2, r=0.0)     # → 1.2649 (Zappa & Shepherd)
    >>> bivariate_dist(0.8, 2, r=0.35)    # → (1.46, 1.01) (Mindlin 2020 and Monerie 2023)
    """
    from scipy.stats import chi2

    if not 0 < confidence_level < 1:
        raise ValueError(
            f"confidence_level must be between 0 and 1, got {confidence_level}"
        )
    if not -1 < r < 1:
        raise ValueError(
            f"r must be between -1 and 1, got {r}"
        )

    c = chi2.ppf(confidence_level, df=n_drivers)

    if r == 0.0:
        # As in Zappa & Shepherd (2017) — single symmetric coefficient r=0 (means independent drivers)
        return float(np.sqrt(c / n_drivers))
    else:
        # As in Mindlin et al. (2020) and Monerie et al. (2023) — correlated drivers, two coefficients
        i1 = float(np.sqrt((1 - r**2) / (2 * (1 - r)) * c))  # same direction
        i2 = float(np.sqrt((1 - r**2) / (2 * (1 + r)) * c))  # opposing
        return i1, i2

def storyline_evaluation(main_config, target, drivers,
                         storyline_coefficient=None,
                         coeffs_high=None, coeffs_low=None,
                         confidence_level=0.8,
                         use_correlation=False,
                         gw_level=1):

    data = xr.open_dataset(
        os.path.join(main_config["work_dir"], "regression_output",
                     target, "regression_coefficients.nc")
    )

    d0, d1 = drivers[0], drivers[1]

    if use_correlation:
        # correlated drivers, two coefficients
        from storypy.utils import find_driver_csv
        stand_path = find_driver_csv(
            main_config["work_dir"], explicit=main_config.get("driver_csv")
        )
        df_stand = pd.read_csv(stand_path, index_col=0)
        r        = float(df_stand[d0].corr(df_stand[d1]))
        i1, i2   = bivariate_dist(confidence_level, 2, r=r)
        print(f"r={r:.3f}, i1={i1:.4f} (same dir), i2={i2:.4f} (opposing)")

        storylines = [
            data['MEM'] + i2 * data[d1] - i2 * data[d0],   # high d1, low d0
            data['MEM'] + i1 * data[d1] + i1 * data[d0],   # high d1, high d0
            data['MEM'] - i1 * data[d1] - i1 * data[d0],   # low d1, low d0
            data['MEM'] - i2 * data[d1] + i2 * data[d0],   # low d1, high d0
            data['MEM']
        ]

    elif coeffs_high is not None and coeffs_low is not None:
        # Data-driven quantiles — different value per driver
        ch = coeffs_high
        cl = coeffs_low
        storylines = [
            data['MEM'] + ch[d1] * data[d1] + cl[d0] * data[d0],
            data['MEM'] + ch[d1] * data[d1] + ch[d0] * data[d0],
            data['MEM'] + cl[d1] * data[d1] + cl[d0] * data[d0],
            data['MEM'] + cl[d1] * data[d1] + ch[d0] * data[d0],
            data['MEM']
        ]

    else:
        # Default to r=0 or explicit fixed coefficient passed by user
        sc = storyline_coefficient if storyline_coefficient is not None \
            else bivariate_dist(confidence_level=confidence_level,
                                n_drivers=2, r=0.0)
        print(f"Using storyline coefficient: {sc:.4f}")
        storylines = [
            data['MEM'] + sc * data[d1] - sc * data[d0],
            data['MEM'] + sc * data[d1] + sc * data[d0],
            data['MEM'] - sc * data[d1] - sc * data[d0],
            data['MEM'] - sc * data[d1] + sc * data[d0],
            data['MEM']
        ]

    storyline_labels = [
        f"high {d1} low {d0}",
        f"high {d1} high {d0}",
        f"low {d1} low {d0}",
        f"low {d1} high {d0}",
        "MEM"
    ]

    storyline_da = xr.concat(storylines, dim="storyline") * gw_level
    storyline_da["storyline"] = storyline_labels
    storyline_da.name = "storylines"
    return storyline_da, storyline_labels

def regression_coefficient(main_config, target, drivers, storyline_coefficient=None, gw_level=1):
    """
    Load regression coefficients for the specified target and drivers.

    Parameters:
        main_config (dict): Configuration containing 'work_dir'.
        target (str): Target variable (e.g., 'pr').
        drivers (list[str]): List of driver variable names.
        storyline_coefficient (float, optional): Not used here, kept for compatibility.
        gw_level (float): Not used here, kept for compatibility.

    Returns:
        list[xarray.DataArray]: List of coefficient DataArrays for each driver.
        list[str]: Titles (driver names) for plotting.
    """
    target_path = os.path.join(main_config["work_dir"], "regression_output", target, "regression_coefficients.nc")
    data = xr.open_dataset(target_path)

    if target == 'pr':
        data = data # * 86400  # Convert from kg/m2/s to mm/day

    # Ensure all requested drivers exist
    missing = [drv for drv in drivers if drv not in data]
    if missing:
        raise ValueError(f"Missing drivers in dataset: {missing}")

    coefficients = [data[drv] for drv in drivers]
    return coefficients, drivers


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

def create_multi_panel_figure(
    data_list,
    extent_list,
    colormaps,
    titles,
    colorbar_label="Colorbar Label",
    ncols=3,
    figsize_per_plot=(5, 4),

    # optional args
    shared_colorbar=True,
    fixed_range=None,          # e.g. 0.06 to match paper; None = auto from data
    tick_levels=None,          # e.g. [-0.06, -0.03, 0, 0.03, 0.06]
    extend="both"              # paper-like saturation at ends
):
    """
    Flexible multi-panel map plotting function.

    - If shared_colorbar=True: one shared colorbar for all panels.
    - If fixed_range is provided: all panels share vmin/vmax = ±fixed_range.
    """

    nplots = len(data_list)
    nrows = math.ceil(nplots / ncols)
    figsize = (figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)

    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=figsize, dpi=300, constrained_layout=True,
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    axs = np.array(axs).flatten()

    # choose one shared range (paper-like), or global max from data
    if fixed_range is None:
        fixed_range = float(max(np.nanmax(np.abs(d.values)) for d in data_list))

    # build shared levels/ticks once
    color_levels, default_ticks = make_symmetric_colorbar(fixed_range, num_steps=12)
    if tick_levels is None:
        tick_levels = default_ticks

    ims = []

    for i, ax in enumerate(axs[:nplots]):
        data = data_list[i]

        ax.set_extent(extent_list[i], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
        ax.add_feature(cfeature.BORDERS, linestyle="--", linewidth=0.5)
        ax.gridlines(draw_labels=False, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")

        # centered white band colormap (keep your logic)
        original_cmap = plt.get_cmap(colormaps[i])
        shifted_cmap = original_cmap(np.linspace(0, 1, len(color_levels)))
        mid_index = len(color_levels) // 2
        shifted_cmap[mid_index - 1:mid_index + 1] = [1, 1, 1, 1]
        new_cmap = mcolors.ListedColormap(shifted_cmap)

        norm = mcolors.TwoSlopeNorm(vmin=-fixed_range, vcenter=0, vmax=fixed_range)

        data_cyclic, lon_cyclic = add_cyclic_point(data.values, coord=data.lon)

        im = ax.contourf(
            lon_cyclic, data.lat, data_cyclic,
            levels=color_levels, cmap=new_cmap, norm=norm,
            extend=extend,
            transform=ccrs.PlateCarree()
        )
        ims.append(im)

        ax.set_title(titles[i], fontsize=12, pad=4)

        # remove per-panel colorbar if using shared
        if not shared_colorbar:
            cbar = fig.colorbar(im, ax=ax, orientation="horizontal",
                                fraction=0.046, pad=0.08, ticks=tick_levels)
            cbar.set_label(colorbar_label, fontsize=10)
            cbar.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
            cbar.ax.tick_params(labelsize=8)

    for ax in axs[nplots:]:
        ax.set_visible(False)

    # shared colorbar
    if shared_colorbar and ims:
        # Add colorbar axes manually: [left, bottom, width, height] in figure fraction
        cbar_ax = fig.add_axes([0.25, 0.04, 0.5, 0.05])  # ← adjust width (0.5) to make it longer/shorter
        cbar = fig.colorbar(
            ims[0],
            cax=cbar_ax,
            orientation="horizontal",
            ticks=tick_levels
        )
        cbar.set_label(colorbar_label, fontsize=12)
        cbar.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
        cbar.ax.tick_params(labelsize=10)

    plt.show()

def _circular_boundary():
    """Standard cartopy recipe: a circle inscribed in the unit square of
    axes coordinates, for clipping a polar-stereo panel to a round shape."""
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    return mpath.Path(verts * radius + center)
 
 
def _broadcast(value, n):
    """None/str -> [value]*n ; list/tuple -> passed through (must be len n)."""
    if value is None:
        return [None] * n
    if isinstance(value, str):
        return [value] * n
    if len(value) != n:
        raise ValueError(f"Expected {n} entries, got {len(value)}: {value}")
    return list(value)
 
 
def plot_storyline_map(map_data, extents, levels, colormaps, titles,
                        colorbar_label='Colorbar Label',
                        white_pct=0.05,
                        projection=None,
                        circular_boundary=False,
                        season_label=None,
                        panel_labels=None,
                        panel_letters=('a', 'b', 'd', 'e', 'c'),
                        show_gridlines=False,
                        show_gridlabels=None,
                        n_lon_ticks=None,
                        n_lat_ticks=None,
                        panel_width=2.6,
                        panel_height=None,
                        col_gap=0.5,
                        row_gap=0.55,
                        margin=0.3,
                        top_margin=0.35,
                        bottom_margin=0.9,
                        font_family='serif',
                        font_size=10,
                        dpi=300):
    """
    Parameters
    ----------
    white_pct : float
        Fraction of the total colorbar range to render as white around zero.
        Default 0.05 (±5% of range); 0.0 disables the white band.
    projection : cartopy.crs.Projection or None
        Axes projection for every panel. Defaults to ccrs.PlateCarree()
        (rectangular sector map). Pass e.g. ccrs.NorthPolarStereo() for a
        hemispheric view; pair with circular_boundary=True for a round panel.
    circular_boundary : bool
        Clip each panel to a circle instead of the default rectangle.
        Meaningful only with a polar projection.
    season_label : str, list of 5 str, or None
        Upper-left corner text outside each panel (e.g. "JJA", "MJJASO").
        A single string is applied to all 5 panels.
    panel_labels : list of 5 str, or None
        Upper-right corner text outside each panel (e.g. "Storyline 1",
        "Storyline 2", "Multi-model mean", "Storyline 3", "Storyline 4"),
        in the same [(0,0),(0,2),(2,0),(2,2),(1,1)] panel order as `titles`.
    panel_letters : sequence of 5 str, or None
        Boxed letter annotation inside the top-left of each panel. None
        omits it.
    show_gridlines : bool
        Draw the dashed lat/lon gridlines. Default False.
    show_gridlabels : bool
        Draw the degree tick labels (e.g. "100E", "10N"). Default True -
        independent of show_gridlines, so labels can stay while lines go.
    wspace, hspace : float
        GridSpec spacing between panels. hspace defaults to a small
        positive value rather than 0 to leave room for season_label/
        panel_labels sitting above each row; set to 0.0 if you don't use
        those and want panels touching vertically too.
    """
    if projection is None:
        projection = ccrs.PlateCarree()
    if show_gridlabels is None:
        show_gridlabels = not circular_boundary
    if panel_height is None:
        panel_height = panel_width
    boundary = _circular_boundary() if circular_boundary else None
 
    n_panels = 5
    single_season = isinstance(season_label, str)
    season_labels = [None] * n_panels if single_season else _broadcast(season_label, n_panels)
    corner_labels = _broadcast(panel_labels, n_panels)
    letters = list(panel_letters) if panel_letters is not None else [None] * n_panels
 
    # --- Figure size computed FROM the panel geometry, in inches
    fig_w_in = 2 * margin + 2 * panel_width + col_gap
    fig_h_in = top_margin + 3 * panel_height + 2 * row_gap + bottom_margin
 
    # --- Panel rects, in inches, then converted to figure fractions.
    #     x-fractions divide by fig_w_in, y-fractions by fig_h_in.
    left_x_in   = margin
    right_x_in  = fig_w_in - margin - panel_width
    center_x_in = fig_w_in / 2 - panel_width / 2
 
    row_top_y_in = fig_h_in - top_margin - panel_height
    row_mid_y_in = row_top_y_in - row_gap - panel_height
    row_bot_y_in = row_mid_y_in - row_gap - panel_height   # == bottom_margin, by construction
 
    def rect(x_in, y_in):
        return (x_in / fig_w_in, y_in / fig_h_in,
                panel_width / fig_w_in, panel_height / fig_h_in)
 
    # Order matches [corner1, corner2, corner3, corner4, mean] and the
    # default panel_letters ('a','b','d','e','c').
    rects = [
        rect(left_x_in,   row_top_y_in),   # a, top-left
        rect(right_x_in,  row_top_y_in),   # b, top-right
        rect(left_x_in,   row_bot_y_in),   # d, bottom-left
        rect(right_x_in,  row_bot_y_in),   # e, bottom-right
        rect(center_x_in, row_mid_y_in),   # c, center
    ]
 
    with plt.rc_context({'font.family': font_family, 'font.size': font_size}):
        fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
 
        if single_season:
            fig.text(margin / fig_w_in, 1 - (top_margin * 0.4) / fig_h_in,
                      season_label, ha='left', va='center', fontsize=font_size)
 
        im = None
        for i, r in enumerate(rects):
            ax = fig.add_axes(r, projection=projection)
            ax.set_extent(extents[i], crs=ccrs.PlateCarree())
            if boundary is not None:
                ax.set_boundary(boundary, transform=ax.transAxes)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
            ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=0.5)
 
            if show_gridlines or show_gridlabels:
                gl = ax.gridlines(draw_labels=show_gridlabels,
                                   linewidth=0.5, color='gray', alpha=0.5,
                                   linestyle='--')
                gl.xlines = show_gridlines
                gl.ylines = show_gridlines
                for attr, val in (('top_labels', False), ('right_labels', False)):
                    try:
                        setattr(gl, attr, val)
                    except AttributeError:
                        pass
                if n_lon_ticks is not None:
                    gl.xlocator = mticker.MaxNLocator(n_lon_ticks)
                if n_lat_ticks is not None:
                    gl.ylocator = mticker.MaxNLocator(n_lat_ticks)
 
            if season_labels[i] is not None:
                ax.text(0.0, 1.02, season_labels[i], transform=ax.transAxes,
                        ha='left', va='bottom', fontsize=font_size + 1)
            if corner_labels[i] is not None:
                ax.text(1.0, 1.02, corner_labels[i], transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=font_size + 1)
            ax.text(0.5, 1.14, titles[i], transform=ax.transAxes,
                    ha='center', va='bottom', fontsize=font_size + 1)
 
            if letters[i] is not None:
                ax.text(0.035, 0.92, f"{letters[i]})", transform=ax.transAxes,
                        ha='left', va='top', fontsize=font_size + 1,
                        fontweight='bold',
                        bbox=dict(boxstyle='square,pad=0.25', facecolor='white',
                                  edgecolor='black', linewidth=0.8),
                        zorder=10)
 
            data        = map_data[i]
            color_levels = levels[i]
            plot_range   = max(abs(color_levels[0]), abs(color_levels[-1]))
            tick_levels  = color_levels[::2]
 
            original_cmap = plt.get_cmap(colormaps[i])
            shifted_cmap  = original_cmap(np.linspace(0, 1, len(color_levels)))
 
            if white_pct > 0.0:
                slot_width   = (2 * plot_range) / len(color_levels)
                white_half   = white_pct * plot_range
                n_white_half = max(1, round(white_half / slot_width))
 
                mid_index    = len(color_levels) // 2
                lo = max(0,                  mid_index - n_white_half)
                hi = min(len(color_levels),  mid_index + n_white_half)
                shifted_cmap[lo:hi] = [1, 1, 1, 1]
 
            new_cmap = mcolors.ListedColormap(shifted_cmap)
            norm     = mcolors.TwoSlopeNorm(
                vmin=-plot_range, vcenter=0, vmax=plot_range
            )
 
            data_cyclic, lon_cyclic = add_cyclic_point(
                data.values, coord=data.lon
            )
            im = ax.contourf(
                lon_cyclic, data.lat, data_cyclic,
                levels=color_levels, cmap=new_cmap, norm=norm,
                extend='both', transform=ccrs.PlateCarree()
            )
 
        if im:
            cbar_w_in = 2 * panel_width + col_gap
            cbar_h_in = 0.15
            cbar_y_in = bottom_margin * 0.35
            cbar_ax = fig.add_axes([
                (fig_w_in / 2 - cbar_w_in / 2) / fig_w_in,
                cbar_y_in / fig_h_in,
                cbar_w_in / fig_w_in,
                cbar_h_in / fig_h_in,
            ])
            cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal',
                                 ticks=tick_levels)
            cbar.ax.xaxis.set_major_formatter(
                FuncFormatter(lambda x, _: f'{x:.1f}')
            )
            cbar.set_label(colorbar_label)
 
        plt.show()
        return fig

def plot_map(data, levels, extent, cmap, title, colorbar_label='Colorbar Label'):
    """
    Plot data with an automatically chosen projection based on the extent.

    Projection selection:
    - Polar (lat_max > 70 or lat_min < -70): NorthPolarStereo / SouthPolarStereo
    - Wide longitude span (> 300°) or global: Robinson
    - Regional box: PlateCarree
    """
    lon_min, lon_max, lat_min, lat_max = extent
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min

    # Choose projection
    if lat_min >= 20:
        proj = ccrs.NorthPolarStereo(central_longitude=(lon_min + lon_max) / 2)
    elif lat_max <= -20:
        proj = ccrs.SouthPolarStereo(central_longitude=(lon_min + lon_max) / 2)
    elif lon_span >= 300 or (lon_span >= 150 and lat_span >= 120):
        proj = ccrs.Robinson(central_longitude=(lon_min + lon_max) / 2)
    else:
        proj = ccrs.PlateCarree(central_longitude=(lon_min + lon_max) / 2)

    fig = plt.figure(figsize=(10, 6))
    ax  = fig.add_subplot(111, projection=proj)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    data_cyclic, lon_cyclic = add_cyclic_point(data.values, coord=data.lon)
    im = ax.contourf(lon_cyclic, data.lat, data_cyclic,
                     levels=levels, cmap=cmap,
                     transform=ccrs.PlateCarree(), extend='both')

    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
    ax.gridlines(draw_labels=True, linewidth=0.4, color='grey',
                 linestyle='--', x_inline=False, y_inline=False)

    cbar = fig.colorbar(im, ax=ax, orientation='horizontal',
                        pad=0.06, shrink=0.7, fraction=0.05)
    cbar.set_label(colorbar_label, fontsize=9)

    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    # plt.show()
    return fig


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


def plot_ellipse(models,x,y,corr='no',x_label='Eastern Pacific Warming [K K$^{-1}$]',y_label='Central Pacific Warming [K K$^{-1}$]', xerr=None,yerr=None):
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
    min_y = np.min(y) - 0.2*np.abs(np.max(y) - np.min(y))
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
    for i, (px, py, t, l) in enumerate(zip(x, y, markers, models)):
        ex = float(xerr[i]) if xerr is not None else 0
        ey = float(yerr[i]) if yerr is not None else 0
        if xerr is not None or yerr is not None:
            ax.errorbar(px, py,
                        xerr=ex if xerr is not None else None,
                        yerr=ey if yerr is not None else None,
                        fmt='', ecolor='gray', elinewidth=0.5,
                        capsize=2, capthick=0.5, zorder=1)
        ax.scatter(px, py, marker=t, label=l, zorder=2)

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
        ts1 = np.sqrt(((1-r**2)/(2*(1-r)))*chi)  # same direction
        ts2 = np.sqrt(((1-r**2)/(2*(1+r)))*chi)  # opposing direction
        # Same-direction corners: A+M+ and A-M-
        ax.plot(mean_x + ts1*np.std(x), mean_y + ts1*np.std(y),
                'ro', alpha=0.6, markersize=10, label='A+M+')
        ax.plot(mean_x - ts1*np.std(x), mean_y - ts1*np.std(y),
                'ro', alpha=0.6, markersize=10, label='A-M-')
        # Opposing corners: A+M- and A-M+
        ax.plot(mean_x + ts2*np.std(x), mean_y - ts2*np.std(y),
                'o', color='orange', alpha=0.6, markersize=10, label='A+M-')
        ax.plot(mean_x - ts2*np.std(x), mean_y + ts2*np.std(y),
                'go', alpha=0.6, markersize=10, label='A-M+')
    elif corr == 'ma':
        r = np.corrcoef(x,y)[0,1]; chi = (1.26**2)*2
        ts1 = np.sqrt(((1-r**2)/(2*(1-r)))*chi)  # same direction
        ts2 = np.sqrt(((1-r**2)/(2*(1+r)))*chi)  # opposing direction
        ax.plot(mean_x + ts1*np.std(x), mean_y + ts1*np.std(y),
                'ro', alpha=0.6, markersize=10, label='A+M+')
        ax.plot(mean_x - ts1*np.std(x), mean_y - ts1*np.std(y),
                'ro', alpha=0.6, markersize=10, label='A-M-')
        ax.plot(mean_x + ts2*np.std(x), mean_y - ts2*np.std(y),
                'o', color='orange', alpha=0.6, markersize=10, label='A+M-')
        ax.plot(mean_x - ts2*np.std(x), mean_y + ts2*np.std(y),
                'go', alpha=0.6, markersize=10, label='A-M+') 
    elif corr == 'pacific':
        r = np.corrcoef(x,y)[0,1]; chi = (1.26**2)*2
        ts1 = np.sqrt(((1-r**2)/(2*(1-r)))*chi)  # same direction
        ts2 = np.sqrt(((1-r**2)/(2*(1+r)))*chi)  # opposing direction
        ax.plot(mean_x + ts1*np.std(x), mean_y + ts1*np.std(y),
                'ro', alpha=0.6, markersize=10, label='High asym Pacific Warming')
        ax.plot(mean_x - ts1*np.std(x), mean_y - ts1*np.std(y),
                'bo', alpha=0.6, markersize=10, label='Low asym Pacific Warming')
        ax.plot(mean_x + ts2*np.std(x), mean_y - ts2*np.std(y),
                'o', color='orange', alpha=0.6, markersize=10, label='High x, Low y')
        ax.plot(mean_x - ts2*np.std(x), mean_y + ts2*np.std(y),
                'go', alpha=0.6, markersize=10, label='Low x, High y')  
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
    return fig