from storypy.utils import np, xr, plt, ccrs, gridspec, cfeature, pd
from shapely.geometry.polygon import Polygon
from cartopy.util import add_cyclic_point
import matplotlib as mpl
import matplotlib.ticker as mticker

def create_arc(lon_min, lon_max, lat_min, lat_max, n_points=100):
    lons = np.linspace(lon_min, lon_max, n_points)
    lats1 = np.full(n_points, lat_min)
    lats2 = np.full(n_points, lat_max)
    lons_combined = np.concatenate([lons, lons[::-1]])
    lats_combined = np.concatenate([lats1, lats2[::-1]])
    return Polygon(zip(lons_combined, lats_combined))

# Plotting function with stippling
def plot_function(target_change, p_values, positives_model, negatives_model, region_extents,
                  sig_level=0.05, sig=1, map_extent=None, projection='platecarree',
                  central_longitude=0):
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
 
    if central_longitude == 0:
        target_change = target_change.sel(
            lon=slice(extent[0], extent[1]),
            lat=slice(extent[2], extent[3])
        )
    else:
        target_change = target_change.sel(
            lat=slice(extent[2], extent[3])
        )

    if central_longitude == 0:
        extent = [
            float(target_change['lon'].min()), float(target_change['lon'].max()),
            float(target_change['lat'].min()), float(target_change['lat'].max())
        ]
 
    print(f"extent used: {extent}")

    if central_longitude != 0:
        target_data, target_lon = add_cyclic_point(
            target_change.values,
            coord=target_change['lon'].values
        )
        target_change = xr.DataArray(
            target_data,
            dims=target_change.dims,
            coords={
                'lat': target_change['lat'],
                'lon': target_lon,
            },
            attrs=target_change.attrs,
        )
 
    # Align p_values
    if p_values is not None:
        if central_longitude == 0:
            p_values = p_values.sel(
                lon=slice(extent[0], extent[1]),
                lat=slice(extent[2], extent[3])
            )
        else:
            p_values = p_values.sel(
                lat=slice(extent[2], extent[3])
            )
 
    # Align beta masks
    if positives_model is not None and negatives_model is not None:
        if central_longitude == 0:
            positives_model_da = positives_model['pr'].sel(
                lon=slice(extent[0], extent[1]),
                lat=slice(extent[2], extent[3])
            )
        else:
            positives_model_da = positives_model['pr'].sel(
                lat=slice(extent[2], extent[3])
            )
        if central_longitude == 0:
            negatives_model_da = negatives_model['pr'].sel(
                lon=slice(extent[0], extent[1]),
                lat=slice(extent[2], extent[3])
            )
        else:
            negatives_model_da = negatives_model['pr'].sel(
                lat=slice(extent[2], extent[3])
            )
 
    # Plot
    if projection == 'robinson':
        proj = ccrs.Robinson(central_longitude=central_longitude)
        figsize = (18, 10)
    elif projection == 'platecarree':
        proj = ccrs.PlateCarree(central_longitude=central_longitude)
        figsize = (15, 10)
    else:
        raise ValueError(f"projection must be 'robinson' or 'platecarree', "
                        f"got '{projection}'")
 
    fig, ax = plt.subplots(figsize=figsize,
                        subplot_kw={'projection': proj},
                        constrained_layout=True)
 
    # Robinson shows full globe — no set_extent needed
    if projection == 'platecarree' and map_extent is not None:
        ax.set_extent(map_extent, crs=ccrs.PlateCarree(central_longitude=central_longitude))
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
 
        # Beta mask — models agree on sign ------------------------------------------------------------------
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
 
        # Large but non-robust: gamma only → open circles
        large_nonrobust = gamma_mask.values
        if large_nonrobust.any():
            ax.scatter(
                lon2d[large_nonrobust], lat2d[large_nonrobust],
                s=40, facecolors='none', edgecolors='black', linewidths=0.8,
                transform=ccrs.PlateCarree(), zorder=4,
            )
            legend_handles.append(
                mlines.Line2D([0], [0], marker='o', color='black',
                              linestyle='None', markersize=6,
                              markerfacecolor='none',
                              label='Large signal (γ > 1)')
            )

        # Robust: beta AND gamma → filled dots
        robust = beta_mask.values
        if robust.any():
            ax.scatter(
                lon2d[robust], lat2d[robust],
                color='black', s=10,
                transform=ccrs.PlateCarree(), zorder=5,
            )
            legend_handles.append(
                mlines.Line2D([0], [0], marker='.', color='black',
                              linestyle='None', markersize=6,
                              label='Robust (β ≥ 90%)')
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

# The following plot_precipitation_change function is deprecated and will be removed in future versions. Please use plot_anomaly_series instead as it is more refined for good visuals.
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
 
# Okabe-Ito colourblind-safe palette (10 colours, cycling for many models)
_OKABE_ITO = [
    '#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7',
    '#56B4E9', '#F0E442', '#000000', '#999999', '#E69F00',
]
 
# RC params for consistent figure styling
_RC = {
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':         9,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   8,
    'axes.linewidth':    0.8,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size':  3,
    'ytick.major.size':  3,
    'grid.linewidth':    0.4,
    'grid.color':        '#DDDDDD',
    'grid.alpha':        1.0,
    'axes.grid':         True,
    'axes.grid.axis':    'y',
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.05,
}
 
 
def plot_anomaly_series(
    target_change,
    region_extents,
    years,
    member_series     = None,
    var_name          = 'pr',
    unit              = r'm s$^{-1}$ K$^{-1}$',
    region_labels     = None,
    baseline_period   = (1960, 1990),
    mid_period        = (2040, 2070),
    eoc_period        = (2070, 2100),
    rolling_window    = 30,
    show_members      = True,
    box_width_ratio   = 0.30,
    figsize           = None,
    panel_labels      = None,
):
    """
    Plot regional precipitation anomaly time series with box-and-whisker
    end-of-century panels.
 
    Parameters
    ----------
    target_change : list of xr.DataArray
        One DataArray per model, each with a ``time`` dimension and
        spatial dimensions ``lat`` / ``lon``.
    region_extents : list of tuple
        Each tuple is ``(lat_min, lat_max, lon_min, lon_max)``.
    years : array-like
        Year values corresponding to the time axis.
    var_name : str
        Variable name for the y-axis label. Default ``'pr'``.
    region_labels : list of str, optional
        Panel titles. Defaults to ``['Region 1', 'Region 2', ...]``.
    baseline_period : tuple of int
        Start and end year for the baseline mean. Default ``(1960, 1990)``.
    mid_period : tuple of int
        Start and end year for the mid-century box plot. Default
        ``(2040, 2070)``.
    eoc_period : tuple of int
        Start and end year for the end-of-century box plot. Default
        ``(2070, 2100)``.
    rolling_window : int
        Window length in years for the rolling mean. Default 30.
    show_members : bool
        Whether to plot individual ensemble members as thin grey lines.
        Default ``True``.
    box_width_ratio : float
        Width of the box plot panel relative to the time series panel.
        Default 0.22.
    figsize : tuple, optional
        Figure size in inches. Auto-scaled if ``None``.
    panel_labels : list of str, optional
        Bold panel labels e.g. ``['(a)', '(b)']``. Default: none.
 
    Returns
    -------
    matplotlib.figure.Figure
    """
    mpl.rcParams.update(_RC)
 
    years        = np.asarray(years)
    num_regions  = len(region_extents)
    region_labels = region_labels or [chr(ord('A') + i) for i in range(num_regions)]
    panel_labels  = panel_labels  or [None] * num_regions
 
    # Width ratios: [ts_panel, box_panel, ts_panel, box_panel, ...]
    width_ratios = []
    for _ in range(num_regions):
        width_ratios += [1.0, box_width_ratio]
    ncols = num_regions * 2
 
    if figsize is None:
        figsize = (5.5 * num_regions + 1.5 * num_regions * box_width_ratio, 3.8)
 
    fig, axes = plt.subplots(
        1, ncols,
        figsize=figsize,
        gridspec_kw={'width_ratios': width_ratios, 'wspace': 0.05},
    )
    # axes layout: axes[0]=ts1, axes[1]=box1, axes[2]=ts2, axes[3]=box2, ...
 
    colours = [_OKABE_ITO[i % len(_OKABE_ITO)] for i in range(len(target_change))]
 
    for i, (region_extent, region_label) in enumerate(
        zip(region_extents, region_labels)
    ):
        ax_ts  = axes[i * 2]
        ax_box = axes[i * 2 + 1]
 
        lat_min, lat_max, lon_min, lon_max = region_extent
 
        def _to_da(obj, var):
            if isinstance(obj, xr.Dataset):
                return obj[var] if var in obj \
                       else obj[list(obj.data_vars)[0]]
            if isinstance(obj, xr.DataArray):
                return obj
            return obj
 
        target_change_da = [_to_da(m, var_name) for m in target_change]
 
        # --- Extract per-model regional anomaly time series ---
        # model_series stores (time_values, rm_values) tuples
        model_series   = []
        eoc_values     = []
 
        for model_da in target_change_da:
            region = model_da.sel(
                lat=slice(lat_min, lat_max),
                lon=slice(lon_min, lon_max),
            )
            # Spatial mean
            lat_w  = np.cos(np.deg2rad(region['lat']))
            region = region.weighted(lat_w).mean(('lat', 'lon')).squeeze()
 
            # Baseline subtraction
            baseline = region.sel(
                time=slice(*baseline_period)
            ).mean(dim='time')
            anomaly = region - baseline
 
            # 30-yr rolling mean — store with actual time coordinate
            rm = (anomaly
                  .rolling(time=rolling_window, center=True, min_periods=1)
                  .mean())
            model_series.append((rm.time.values, rm.squeeze().values))
 
            # # Baseline mean (should be ~0 by construction)
            # base_mean = float(
            #     anomaly.sel(time=slice(*baseline_period)).mean(dim='time')
            # )
            # base_values.append(base_mean)
 
            # # Mid-century mean
            # mid_mean = float(
            #     anomaly.sel(time=slice(*mid_period)).mean(dim='time')
            # )
            # mid_values.append(mid_mean)
 
            # End-of-century mean
            eoc_mean = float(
                anomaly.sel(time=slice(*eoc_period)).mean(dim='time')
            )
            eoc_values.append(eoc_mean)
 
        # --- Layer 1: model means (dark grey, drawn first) ---
        for t_vals, rm_vals in model_series:
            ax_ts.plot(t_vals, rm_vals,
                       color='#333333', linewidth=1.0,
                       alpha=0.85, zorder=3)
 
        # --- Layer 2: individual member lines (light grey, on top of model lines) ---
        if member_series is not None:
            for mem_da in member_series:
                if isinstance(mem_da, xr.Dataset):
                    mem_da = mem_da[list(mem_da.data_vars)[0]]
                region = mem_da.sel(
                    lat=slice(lat_min, lat_max),
                    lon=slice(lon_min, lon_max),
                )
                lat_w    = np.cos(np.deg2rad(region['lat']))
                region   = region.weighted(lat_w).mean(('lat', 'lon')).squeeze()
                baseline = region.sel(time=slice(*baseline_period)).mean('time')
                anomaly  = region - baseline
                rm = (anomaly
                      .rolling(time=rolling_window, center=True, min_periods=1)
                      .mean())
                ax_ts.plot(rm.time.values, rm.squeeze().values,
                           color='#DDDDDD', linewidth=0.3,
                           alpha=0.6, zorder=2)
 
        # --- Layer 3: MEM (thick red, on top of everything) ---
        # Use common time axis from first model
        t_common  = model_series[0][0]
        mem_stack = []
        for t_vals, rm_vals in model_series:
            if len(rm_vals) == len(t_common):
                mem_stack.append(rm_vals)
        mem_vals = np.nanmean(np.stack(mem_stack, axis=0), axis=0)
        ax_ts.plot(t_common, mem_vals, color='#D55E00', linewidth=2.5,
                   zorder=5, label='Multi-model Mean')
 
        ax_ts.axhline(0, color='black', linewidth=0.6,
                      linestyle='--', zorder=4)
 
        # Axis formatting
        ax_ts.set_title(region_label, pad=6)
        ax_ts.set_xlabel('Year', labelpad=4)
        if i == 0:
            ax_ts.set_ylabel(f'{var_name} anomaly ({unit})', labelpad=4)
        ax_ts.set_xlim(float(model_series[0][0][0]), float(model_series[0][0][-1]))
        ax_ts.xaxis.set_major_locator(mticker.MultipleLocator(20))
        ax_ts.tick_params(axis='x', labelrotation=0)
 
        leg = ax_ts.legend(loc='upper left', fontsize=7, frameon=True,
                           framealpha=0.9, edgecolor='#BBBBBB',
                           handlelength=1.2)
        leg.get_frame().set_linewidth(0.4)
 
        if panel_labels[i]:
            ax_ts.text(-0.10, 1.05, panel_labels[i],
                       transform=ax_ts.transAxes,
                       fontsize=11, fontweight='bold',
                       va='top', ha='left')
 
        # --- End-of-century box plot only ---
        eoc_arr = np.array(eoc_values)
 
        ax_box.boxplot(
            eoc_arr,
            positions    = [1],
            vert         = True,
            widths       = 0.3,
            patch_artist = True,
            notch        = False,
            showfliers   = True,
            boxprops     = dict(facecolor='#A6CEE3', edgecolor='#1F78B4',
                                linewidth=1.0),
            medianprops  = dict(color='#08306B', linewidth=1.8),
            whiskerprops = dict(color='#1F78B4', linewidth=1.2, linestyle='-'),
            capprops     = dict(color='#1F78B4', linewidth=1.2),
            flierprops   = dict(marker='o', markersize=3,
                                markerfacecolor='#1F78B4',
                                markeredgecolor='none', alpha=0.5),
        )
        # MEM dot only — no per-model scatter
        ax_box.scatter([1], [float(np.nanmean(eoc_arr))],
                    color='#08306B', s=25, zorder=6, linewidths=0)
 
        ax_box.axhline(0, color='black', linewidth=0.6, linestyle='--', zorder=3)
 
        eoc_label = f"{str(eoc_period[0])[2:]}–{str(eoc_period[1])[2:]}"
        ax_box.set_xticks([1])
        ax_box.set_xticklabels([eoc_label], fontsize=8)
        ax_box.set_xlim(0.4, 1.6)

        ax_box.sharey(ax_ts)
        ax_box.tick_params(axis='y', which='both',
                           labelleft=False, labelright=False,
                           left=False, right=False)
        ax_box.set_yticklabels([])
        ax_box.spines['left'].set_visible(False)
        ax_box.grid(axis='y', linewidth=0.4, color='#DDDDDD')
 
    # fig.tight_layout(pad=0.5)
    return fig