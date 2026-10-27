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
        # As in Zappa & Shepherd (2017) - single symmetric coefficient r=0 (means independent drivers)
        return float(np.sqrt(c / n_drivers))
    else:
        # As in Mindlin et al. (2020) and Monerie et al. (2023) - correlated drivers, two coefficients
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
        # Data-driven quantiles - different value per driver
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


def _map_aspect(projection, extent):
    """Height/width of a map with this projection and extent, as cartopy draws it."""
    fig = plt.figure()
    try:
        ax = fig.add_axes([0, 0, 1, 1], projection=projection)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        return abs(y1 - y0) / abs(x1 - x0)
    finally:
        plt.close(fig)


def plot_storyline_map(map_data, extents, levels, colormaps, titles,
                        colorbar_label='Colorbar Label',
                        white_pct=0.05,
                        projection=None,
                        circular_boundary=False,
                        season_label=None,
                        panel_labels=None,
                        panel_letters=('a', 'b', 'd', 'e', 'c'),
                        letter_position='outside',
                        driver_labels=None,
                        show_gridlines=False,
                        show_gridlabels=None,
                        n_lon_ticks=None,
                        n_lat_ticks=None,
                        panel_width=2.6,
                        panel_height=None,
                        col_gap=None,
                        row_gap=None,
                        margin=None,
                        top_margin=None,
                        bottom_margin=None,
                        font_family='serif',
                        font_size=10,
                        dpi=300,
                        show=True):
    """
    Plot four storylines around the multi-model mean in a diamond layout.

    The layout encodes the two storyline drivers: the left column has the low
    value of the first driver and the right column the high value; the top row
    has the high value of the second driver and the bottom row the low value.
    The multi-model mean sits in the middle.

    Parameters
    ----------
    map_data : sequence of 5 xarray.DataArray
        In the order returned by :func:`storyline_evaluation`:
        [high d1 low d0, high d1 high d0, low d1 low d0, low d1 high d0, MEM],
        drawn top-left, top-right, bottom-left, bottom-right, centre.
    extents : sequence of 5 [lon_min, lon_max, lat_min, lat_max]
    levels : sequence of 5 arrays of contour levels
    colormaps : sequence of 5 colormap names
    titles : sequence of 5 str
        Panel titles, shown after the panel letter above each panel.
    colorbar_label : str
    white_pct : float
        Fraction of the colour range around zero drawn white. 0 disables it.
    projection : cartopy.crs.Projection, optional
        Default ``PlateCarree()``. Use e.g. ``NorthPolarStereo()`` with
        ``circular_boundary=True`` for round hemispheric panels.
    circular_boundary : bool
        Clip each panel to a circle (for polar projections).
    season_label : str or sequence of 5 str, optional
        A single string is written once, top-left of the figure; a list adds
        it to each panel's header.
    panel_labels : sequence of 5 str, optional
        Right-aligned header text per panel, e.g. "Storyline 1", ...,
        "Multi-model mean".
    panel_letters : sequence of 5 str or None
        Bold panel letters, default a, b, d, e, c (reading order of the
        diamond). None omits them.
    letter_position : {'outside', 'inside'}
        'outside' (default) puts the letter in the header line above the
        panel, so it never covers data, for both rectangular and circular
        panels. 'inside' keeps the boxed letter in the top-left corner of the
        map (the previous behaviour).
    driver_labels : (str, str), optional
        Names of the first and second storyline drivers, e.g.
        ``("Tropical warming", "Polar vortex")``. Draws arrows showing in which
        direction each driver increases, in the empty space beside the MEM
        panel.
    show_gridlines, show_gridlabels : bool
        Dashed lat/lon lines; degree labels on the left and bottom edges
        (default: labels on, except for circular panels).
    n_lon_ticks, n_lat_ticks : int, optional
        Approximate number of degree labels. By default the number of
        longitude labels scales with ``panel_width`` so they do not overlap.
    panel_width : float
        Width of each map in inches.
    panel_height : float, optional
        Height of each map in inches. Default: computed from the projection
        and extent, so no space is left empty around the map.
    col_gap, row_gap, margin, top_margin, bottom_margin : float, optional
        Spacing in inches. Defaults are computed from ``font_size`` and from
        whether degree labels are drawn.
    font_family, font_size :
        Text style.
    dpi : int
        Figure resolution; also used when saving with ``fig.savefig``.
        The layout is defined in inches, so dpi changes only the pixel
        density, not the arrangement.
    show : bool
        Call ``plt.show()``. Set False to adjust or save the figure first.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if projection is None:
        projection = ccrs.PlateCarree()
    if show_gridlabels is None:
        show_gridlabels = not circular_boundary
    if letter_position not in ('outside', 'inside'):
        raise ValueError("letter_position must be 'outside' or 'inside'")
    boundary = _circular_boundary() if circular_boundary else None

    n_panels = 5
    single_season = isinstance(season_label, str)
    season_labels = [None] * n_panels if single_season else _broadcast(season_label, n_panels)
    corner_labels = _broadcast(panel_labels, n_panels)
    letters = list(panel_letters) if panel_letters is not None else [None] * n_panels

    # --- Geometry, in inches -------------------------------------------------
    if panel_height is None:
        aspect = 1.0 if circular_boundary else max(
            _map_aspect(projection, e) for e in extents)
        panel_height = panel_width * aspect

    line_h  = 1.5 * (font_size + 1) / 72          # one line of header text
    lat_w   = 0.45 * font_size / 10 if show_gridlabels else 0.0   # "60°N" left of a map
    lon_h   = 0.25 * font_size / 10 if show_gridlabels else 0.0   # "25°E" below a map
    cbar_block = 0.15 + 2.6 * font_size / 72 + 0.1                # bar, ticks, label

    margin        = (lat_w + 0.08) if margin is None else margin
    col_gap       = (lat_w + 0.2) if col_gap is None else col_gap
    row_gap       = (lon_h + line_h + 0.08) if row_gap is None else row_gap
    top_margin    = (line_h * (2 if single_season else 1) + 0.05) if top_margin is None else top_margin
    bottom_margin = (lon_h + cbar_block + 0.15) if bottom_margin is None else bottom_margin

    fig_w_in = margin + 2 * panel_width + col_gap + 0.08
    fig_h_in = top_margin + 3 * panel_height + 2 * row_gap + bottom_margin

    left_x   = margin
    right_x  = margin + panel_width + col_gap
    center_x = (left_x + right_x) / 2
    top_y    = fig_h_in - top_margin - panel_height
    mid_y    = top_y - row_gap - panel_height
    bot_y    = mid_y - row_gap - panel_height

    def rect(x_in, y_in):
        return (x_in / fig_w_in, y_in / fig_h_in,
                panel_width / fig_w_in, panel_height / fig_h_in)

    # Order matches storyline_evaluation and the default letters a, b, d, e, c.
    rects = [rect(left_x, top_y), rect(right_x, top_y),
             rect(left_x, bot_y), rect(right_x, bot_y),
             rect(center_x, mid_y)]

    with plt.rc_context({'font.family': font_family, 'font.size': font_size}):
        fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)

        if single_season:
            fig.text(left_x / fig_w_in, 1 - 0.05 / fig_h_in, season_label,
                     ha='left', va='top', fontsize=font_size + 1, fontweight='bold')

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
                for attr in ('top_labels', 'right_labels'):
                    try:
                        setattr(gl, attr, False)
                    except AttributeError:
                        pass
                if n_lon_ticks is None and not circular_boundary:
                    # about one label per 0.65 in, so labels never run together
                    n_lon_ticks = max(3, round(panel_width / 0.65))
                if n_lon_ticks is not None:
                    gl.xlocator = mticker.MaxNLocator(n_lon_ticks)
                if n_lat_ticks is not None:
                    gl.ylocator = mticker.MaxNLocator(n_lat_ticks)
                # cartopy draws these labels at render time, outside rc_context
                label_style = {'family': font_family, 'size': font_size - 1}
                gl.xlabel_style = label_style
                gl.ylabel_style = label_style

            # --- Header: one line above the panel, never on the data -------
            header_y = 1.0 + 0.04 / panel_height
            left_anchor = None
            if letters[i] is not None and letter_position == 'outside':
                left_anchor = ax.text(0.0, header_y, f"{letters[i]})",
                                      transform=ax.transAxes, ha='left',
                                      va='bottom', fontsize=font_size + 1,
                                      fontweight='bold')
            if left_anchor is not None:
                ax.annotate(titles[i], xy=(1, 0), xycoords=left_anchor,
                            xytext=(4, 0), textcoords='offset points',
                            ha='left', va='bottom', fontsize=font_size + 1)
            else:
                ax.text(0.0, header_y, titles[i], transform=ax.transAxes,
                        ha='left', va='bottom', fontsize=font_size + 1)
            right_text = " · ".join(t for t in (season_labels[i], corner_labels[i]) if t)
            if right_text:
                ax.text(1.0, header_y, right_text, transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=font_size,
                        color='0.3')

            if letters[i] is not None and letter_position == 'inside':
                ax.text(0.035, 0.96, f"{letters[i]})", transform=ax.transAxes,
                        ha='left', va='top', fontsize=font_size + 1,
                        fontweight='bold',
                        bbox=dict(boxstyle='square,pad=0.25', facecolor='white',
                                  edgecolor='black', linewidth=0.8),
                        zorder=10)

            data         = map_data[i]
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

        # --- Driver arrows in the empty space beside the MEM panel ----------
        if driver_labels is not None:
            x_label, y_label = driver_labels
            arrow = dict(arrowstyle='-|>', color='0.35', lw=1.2,
                         shrinkA=0, shrinkB=0)
            flank_w = center_x - left_x - 0.1          # free width left of MEM
            pad = 0.1
            # second driver increases upwards (top row = high): left flank
            ax_x = (left_x + flank_w * 0.55) / fig_w_in
            fig.add_artist(mpl.patches.FancyArrowPatch(
                (ax_x, (mid_y + pad) / fig_h_in),
                (ax_x, (mid_y + panel_height - pad) / fig_h_in),
                transform=fig.transFigure, mutation_scale=12, **arrow))
            fig.text(ax_x - 0.08 / fig_w_in, (mid_y + panel_height / 2) / fig_h_in,
                     y_label, rotation=90, ha='right', va='center',
                     fontsize=font_size, color='0.25')
            # first driver increases to the right (right column = high): right flank
            rx0 = (center_x + panel_width + 0.15) / fig_w_in
            rx1 = (right_x + panel_width - 0.15) / fig_w_in
            ry = (mid_y + panel_height / 2) / fig_h_in
            fig.add_artist(mpl.patches.FancyArrowPatch(
                (rx0, ry), (rx1, ry), transform=fig.transFigure,
                mutation_scale=12, **arrow))
            fig.text((rx0 + rx1) / 2, ry + 0.06 / fig_h_in, x_label,
                     ha='center', va='bottom', fontsize=font_size, color='0.25')

        if im:
            cbar_w_in = 2 * panel_width + col_gap
            cbar_h_in = 0.15
            cbar_y_in = bottom_margin - lon_h - 0.15 - cbar_h_in
            cbar_ax = fig.add_axes([
                left_x / fig_w_in,
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

        if show:
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
