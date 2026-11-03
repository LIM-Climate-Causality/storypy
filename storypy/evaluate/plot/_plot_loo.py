"""
storypy.evaluate.plot_loo
==========================

Visualisation functions for leave-one-out (LOO) sensitivity analysis results
produced by :func:`storypy.compute.leave_one_out.run_loo_sensitivity`.

Functions
---------
plot_loo_influence_heatmap  - ranked model influence per driver (heatmap)
plot_loo_spaghetti          - LOO coefficient spread over a target region
plot_loo_violin             - distribution of regional coefficients per driver
plot_loo_std_map            - spatial map of LOO std per driver
plot_loo_dashboard          - all four plots in one figure

Example
-------
>>> from storypy.compute.leave_one_out import run_loo_sensitivity
>>> from storypy.evaluate.plot_loo import plot_loo_dashboard
>>>
>>> results = run_loo_sensitivity('./output', 'pr')
>>> fig = plot_loo_dashboard(
...     results,
...     region=(30, 45, -10, 40),   # Mediterranean
...     region_name='Mediterranean',
... )
>>> fig.savefig('loo_sensitivity.png', dpi=150, bbox_inches='tight')
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ---------------------------------------------------------------------------
# Helper - extract regional mean per LOO iteration
# ---------------------------------------------------------------------------

def _regional_mean(ds: xr.Dataset, region: tuple) -> pd.DataFrame:
    """
    Compute spatial mean over a region for each LOO iteration and driver.

    Parameters
    ----------
    ds : xr.Dataset
        LOO coefficients with dims (left_out_model, lat, lon).
    region : tuple
        (lat_min, lat_max, lon_min, lon_max).

    Returns
    -------
    pd.DataFrame
        Index = left_out_model, columns = driver names.
    """
    lat_min, lat_max, lon_min, lon_max = region
    ds_reg = ds.sel(lat=slice(lat_min, lat_max),
                    lon=slice(lon_min, lon_max))
    mean   = ds_reg.mean(dim=['lat', 'lon'])
    return mean.to_dataframe().drop(columns='left_out_model', errors='ignore')


def _influence_table(loo_coefs: xr.Dataset, region: tuple) -> pd.DataFrame:
    """
    Rank models by their influence (deviation from LOO mean) per driver.

    Returns
    -------
    pd.DataFrame
        Index = left_out_model, columns = drivers, values = mean absolute
        deviation from the LOO mean coefficient in that region.
    """
    reg_df   = _regional_mean(loo_coefs, region)
    loo_mean = reg_df.mean(axis=0)
    influence = reg_df.subtract(loo_mean, axis=1).abs()
    return influence.sort_values('MEM', ascending=False)


# ---------------------------------------------------------------------------
# Plot 1 - Heatmap of model influence per driver
# ---------------------------------------------------------------------------

def plot_loo_influence_heatmap(
    results:      dict,
    region:       tuple,
    region_name:  str  = '',
    drivers:      list = None,
    ax:           plt.Axes = None,
    cmap:         str  = 'YlOrRd',
) -> plt.Figure:
    """
    Heatmap of LOO model influence per driver for a given region.

    Rows = models (sorted by influence on MEM, most influential first).
    Columns = regression drivers.
    Colour = mean absolute deviation of the regional coefficient from the
    LOO mean when that model is dropped.

    Parameters
    ----------
    results : dict
        Output from :func:`run_loo_sensitivity`.
    region : tuple
        (lat_min, lat_max, lon_min, lon_max).
    region_name : str
        Label used in the title.
    drivers : list of str, optional
        Subset of drivers to show. Default: all.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Creates a new figure if None.
    cmap : str
        Colormap. Default 'YlOrRd'.

    Returns
    -------
    matplotlib.figure.Figure
    """
    influence = _influence_table(results['coefficients'], region)
    if drivers is not None:
        influence = influence[[d for d in drivers if d in influence.columns]]

    fig_created = ax is None
    if fig_created:
        fig, ax = plt.subplots(figsize=(max(6, len(influence.columns) * 1.2),
                                        max(5, len(influence) * 0.4)))
    else:
        fig = ax.figure

    im = ax.imshow(influence.values, aspect='auto', cmap=cmap)
    plt.colorbar(im, ax=ax, label='Mean absolute deviation')

    ax.set_xticks(range(len(influence.columns)))
    ax.set_xticklabels(influence.columns, rotation=30, ha='right', fontsize=10)
    ax.set_yticks(range(len(influence.index)))
    ax.set_yticklabels(influence.index, fontsize=8)

    # Annotate cells with values
    for i in range(len(influence.index)):
        for j in range(len(influence.columns)):
            val = influence.iloc[i, j]
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=7,
                    color='white' if val > influence.values.max() * 0.6
                    else 'black')

    title = 'LOO model influence'
    if region_name:
        title += f' - {region_name}'
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Driver', fontsize=10)
    ax.set_ylabel('Dropped model (ranked by MEM influence)', fontsize=10)

    if fig_created:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 2 - Spaghetti plot of LOO coefficient spread
# ---------------------------------------------------------------------------

def plot_loo_spaghetti(
    results:      dict,
    region:       tuple,
    region_name:  str  = '',
    drivers:      list = None,
    highlight:    list = None,
    ax:           plt.Axes = None,
) -> plt.Figure:
    """
    Spaghetti plot of regional-mean LOO coefficients per iteration.

    Each line represents one LOO iteration (one dropped model). The full
    ensemble mean (no model dropped) is shown as a dashed black reference.
    Models named in ``highlight`` are drawn in red.

    Parameters
    ----------
    results : dict
        Output from :func:`run_loo_sensitivity`.
    region : tuple
        (lat_min, lat_max, lon_min, lon_max).
    region_name : str
        Label used in the title.
    drivers : list of str, optional
        Subset of drivers to plot on the x-axis. Default: all.
    highlight : list of str, optional
        Model names to highlight in red (e.g. most influential models).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
    """
    reg_df  = _regional_mean(results['coefficients'], region)
    if drivers is not None:
        reg_df = reg_df[[d for d in drivers if d in reg_df.columns]]

    driver_names  = reg_df.columns.tolist()
    x             = np.arange(len(driver_names))
    full_ens_mean = reg_df.mean(axis=0)

    fig_created = ax is None
    if fig_created:
        fig, ax = plt.subplots(figsize=(max(6, len(driver_names) * 1.5), 5))
    else:
        fig = ax.figure

    highlight = highlight or []

    for model, row in reg_df.iterrows():
        colour    = 'red' if model in highlight else '#a8c8e8'
        linewidth = 1.8   if model in highlight else 0.7
        zorder    = 5     if model in highlight else 2
        alpha     = 1.0   if model in highlight else 0.6
        ax.plot(x, row.values, color=colour, linewidth=linewidth,
                alpha=alpha, zorder=zorder,
                label=model if model in highlight else None)

    # Full ensemble mean reference line
    ax.plot(x, full_ens_mean.values, color='black', linewidth=2.0,
            linestyle='--', zorder=6, label='LOO mean')

    # Shaded ± 1 std band
    loo_std = reg_df.std(axis=0)
    ax.fill_between(x,
                    full_ens_mean - loo_std,
                    full_ens_mean + loo_std,
                    alpha=0.15, color='grey', label='±1 std')

    ax.set_xticks(x)
    ax.set_xticklabels(driver_names, fontsize=10)
    ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
    ax.set_ylabel('Regional mean coefficient', fontsize=10)

    title = 'LOO coefficient spread'
    if region_name:
        title += f' - {region_name}'
    ax.set_title(title, fontsize=12, fontweight='bold')

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8, loc='best')

    if fig_created:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 3 - Violin / box plot per driver
# ---------------------------------------------------------------------------

def plot_loo_violin(
    results:      dict,
    region:       tuple,
    region_name:  str  = '',
    drivers:      list = None,
    n_label:      int  = 3,
    ax:           plt.Axes = None,
) -> plt.Figure:
    """
    Violin plot of regional LOO coefficient distribution per driver.

    Each violin shows the distribution of the regional-mean coefficient
    across all LOO iterations for that driver. The ``n_label`` most
    influential models (largest deviation from mean) are labelled on
    the plot as outlier points.

    Parameters
    ----------
    results : dict
        Output from :func:`run_loo_sensitivity`.
    region : tuple
        (lat_min, lat_max, lon_min, lon_max).
    region_name : str
        Label used in the title.
    drivers : list of str, optional
        Subset of drivers to show. Default: all.
    n_label : int
        Number of outlier models to label per driver. Default 3.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
    """
    reg_df = _regional_mean(results['coefficients'], region)
    if drivers is not None:
        reg_df = reg_df[[d for d in drivers if d in reg_df.columns]]

    driver_names = reg_df.columns.tolist()

    fig_created = ax is None
    if fig_created:
        fig, ax = plt.subplots(figsize=(max(6, len(driver_names) * 1.5), 5))
    else:
        fig = ax.figure

    data_list = [reg_df[d].values for d in driver_names]
    parts     = ax.violinplot(data_list, positions=range(len(driver_names)),
                              showmeans=True, showmedians=False,
                              showextrema=True)

    # Style violins
    for pc in parts['bodies']:
        pc.set_facecolor('#a8c8e8')
        pc.set_edgecolor('#2166ac')
        pc.set_alpha(0.7)

    # Overlay individual points and label outliers
    loo_mean = reg_df.mean(axis=0)
    for j, driver in enumerate(driver_names):
        col      = reg_df[driver]
        deviations = (col - loo_mean[driver]).abs().sort_values(ascending=False)
        top_models = deviations.index[:n_label].tolist()

        for model, val in col.items():
            is_top = model in top_models
            ax.scatter(j, val,
                       color='red' if is_top else 'steelblue',
                       s=25 if is_top else 10,
                       zorder=5 if is_top else 3,
                       alpha=1.0 if is_top else 0.5)
            if is_top:
                ax.annotate(
                    model,
                    xy=(j, val),
                    xytext=(8, 0),
                    textcoords='offset points',
                    fontsize=6,
                    color='red',
                    va='center',
                )

    ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
    ax.set_xticks(range(len(driver_names)))
    ax.set_xticklabels(driver_names, fontsize=10)
    ax.set_ylabel('Regional mean coefficient', fontsize=10)

    title = f'LOO coefficient distribution (top {n_label} influential models labelled)'
    if region_name:
        title += f'\n{region_name}'
    ax.set_title(title, fontsize=11, fontweight='bold')

    if fig_created:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 4 - Spatial map of LOO std
# ---------------------------------------------------------------------------

def plot_loo_std_map(
    results:      dict,
    drivers:      list = None,
    region_box:   tuple = None,
    projection:   str   = 'platecarree',
    map_extent:   list  = None,
    cmap:         str   = 'Reds',
) -> plt.Figure:
    """
    Spatial map of LOO standard deviation per driver.

    High std at a gridpoint means the regression coefficient there is
    sensitive to which model is dropped - i.e. the result is driven by
    a small number of models.

    Parameters
    ----------
    results : dict
        Output from :func:`run_loo_sensitivity`.
    drivers : list of str, optional
        Subset of drivers to map. Default: all.
    region_box : tuple, optional
        (lat_min, lat_max, lon_min, lon_max) to draw a box on the map.
    projection : str
        ``'platecarree'`` or ``'robinson'``. Default ``'platecarree'``.
    map_extent : list, optional
        [lon_min, lon_max, lat_min, lat_max] to zoom the map.
        Default: global.
    cmap : str
        Colormap. Default ``'Reds'``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    loo_std  = results['std']
    if drivers is not None:
        loo_std = loo_std[drivers]

    driver_names = list(loo_std.data_vars)
    n_drivers    = len(driver_names)

    proj = ccrs.Robinson() if projection == 'robinson' else ccrs.PlateCarree()

    fig, axes = plt.subplots(
        1, n_drivers,
        figsize=(5 * n_drivers, 4),
        subplot_kw={'projection': proj},
    )
    if n_drivers == 1:
        axes = [axes]

    for ax, driver in zip(axes, driver_names):
        da  = loo_std[driver]
        vmax = float(np.nanpercentile(np.abs(da.values), 95))

        da.plot.contourf(
            ax        = ax,
            transform = ccrs.PlateCarree(),
            cmap      = cmap,
            levels    = 10,
            vmin      = 0,
            vmax      = vmax,
            add_colorbar = True,
            cbar_kwargs  = {'shrink': 0.7, 'label': 'LOO std'},
        )

        ax.add_feature(cfeature.COASTLINE.with_scale('50m'),
                       edgecolor='black', linewidth=0.6)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'),
                       linestyle='--', edgecolor='grey', linewidth=0.4)

        if map_extent is not None:
            ax.set_extent(map_extent, crs=ccrs.PlateCarree())

        # Draw target region box
        if region_box is not None:
            lat_min, lat_max, lon_min, lon_max = region_box
            ax.plot(
                [lon_min, lon_max, lon_max, lon_min, lon_min],
                [lat_min, lat_min, lat_max, lat_max, lat_min],
                transform=ccrs.PlateCarree(),
                color='blue', linewidth=1.5,
            )

        ax.set_title(f'LOO std - {driver}', fontsize=11, fontweight='bold')

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Dashboard - all four plots
# ---------------------------------------------------------------------------

def plot_loo_dashboard(
    results:      dict,
    region:       tuple,
    region_name:  str   = '',
    drivers:      list  = None,
    map_extent:   list  = None,
    n_label:      int   = 3,
    highlight:    list  = None,
) -> plt.Figure:
    """
    Four-panel LOO sensitivity dashboard.

    Combines all four diagnostic plots into a single figure:

    - Top-left:  heatmap of model influence per driver
    - Top-right: spaghetti plot of coefficient spread
    - Bottom-left: violin plot with labelled outlier models
    - Bottom-right: spatial std map for MEM coefficient

    Parameters
    ----------
    results : dict
        Output from :func:`run_loo_sensitivity`.
    region : tuple
        (lat_min, lat_max, lon_min, lon_max) target region.
    region_name : str
        Region label for titles.
    drivers : list of str, optional
        Subset of drivers. Default: all.
    map_extent : list, optional
        [lon_min, lon_max, lat_min, lat_max] for the std map.
    n_label : int
        Number of outlier models to label in the violin plot.
    highlight : list of str, optional
        Models to highlight in the spaghetti plot.

    Returns
    -------
    matplotlib.figure.Figure

    Example
    -------
    >>> fig = plot_loo_dashboard(
    ...     results,
    ...     region      = (30, 45, -10, 40),
    ...     region_name = 'Mediterranean',
    ...     map_extent  = [-30, 60, 15, 70],
    ...     n_label     = 3,
    ... )
    >>> fig.savefig('loo_dashboard.png', dpi=150, bbox_inches='tight')
    """
    fig = plt.figure(figsize=(18, 12))

    # --- Top left: heatmap ---
    ax_heat = fig.add_subplot(2, 2, 1)
    plot_loo_influence_heatmap(
        results, region, region_name=region_name,
        drivers=drivers, ax=ax_heat,
    )

    # --- Top right: spaghetti ---
    ax_spag = fig.add_subplot(2, 2, 2)
    plot_loo_spaghetti(
        results, region, region_name=region_name,
        drivers=drivers, highlight=highlight, ax=ax_spag,
    )

    # --- Bottom left: violin ---
    ax_viol = fig.add_subplot(2, 2, 3)
    plot_loo_violin(
        results, region, region_name=region_name,
        drivers=drivers, n_label=n_label, ax=ax_viol,
    )

    # --- Bottom right: spatial std map (MEM only) ---
    ax_map = fig.add_subplot(2, 2, 4,
                             projection=ccrs.PlateCarree())
    mem_std = results['std']['MEM']
    vmax    = float(np.nanpercentile(np.abs(mem_std.values), 95))
    mem_std.plot.contourf(
        ax           = ax_map,
        transform    = ccrs.PlateCarree(),
        cmap         = 'Reds',
        levels       = 10,
        vmin         = 0,
        vmax         = vmax,
        add_colorbar = True,
        cbar_kwargs  = {'shrink': 0.7, 'label': 'LOO std'},
    )
    ax_map.add_feature(cfeature.COASTLINE.with_scale('50m'),
                       edgecolor='black', linewidth=0.6)
    ax_map.add_feature(cfeature.BORDERS.with_scale('50m'),
                       linestyle='--', edgecolor='grey', linewidth=0.4)
    if map_extent is not None:
        ax_map.set_extent(map_extent, crs=ccrs.PlateCarree())

    # Draw target region box
    lat_min, lat_max, lon_min, lon_max = region
    ax_map.plot(
        [lon_min, lon_max, lon_max, lon_min, lon_min],
        [lat_min, lat_min, lat_max, lat_max, lat_min],
        transform=ccrs.PlateCarree(),
        color='blue', linewidth=1.5,
    )
    ax_map.set_title('LOO std - MEM', fontsize=11, fontweight='bold')

    fig.suptitle(
        f'LOO sensitivity dashboard'
        + (f' - {region_name}' if region_name else ''),
        fontsize=14, fontweight='bold', y=1.01,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 5 - Sign consistency map
# ---------------------------------------------------------------------------

def plot_loo_sign_consistency(
    results:     dict,
    drivers:     list  = None,
    region_box:  tuple = None,
    map_extent:  list  = None,
) -> plt.Figure:
    """
    Map of sign consistency across LOO iterations per driver.

    At each gridpoint, computes the fraction of LOO iterations where the
    coefficient has the same sign as the LOO mean. A value of 1.0 means
    all iterations agree on sign - the result is sign-robust. Values < 0.8
    indicate the sign is sensitive to individual models and should be
    interpreted cautiously.

    This is a key metric for paper-quality robustness assessment since a
    coefficient that flips sign when one model is dropped is physically
    unreliable regardless of its magnitude.

    Parameters
    ----------
    results : dict
        Output from :func:`run_loo_sensitivity`.
    drivers : list of str, optional
        Subset of drivers to map. Default: all.
    region_box : tuple, optional
        (lat_min, lat_max, lon_min, lon_max) to draw a box on the map.
    map_extent : list, optional
        [lon_min, lon_max, lat_min, lat_max] to zoom the map.

    Returns
    -------
    matplotlib.figure.Figure
    """
    loo_coefs    = results['coefficients']
    loo_mean     = results['mean']
    if drivers is None:
        drivers = list(loo_coefs.data_vars)

    n_drivers = len(drivers)
    fig, axes = plt.subplots(
        1, n_drivers,
        figsize=(5 * n_drivers, 4),
        subplot_kw={'projection': ccrs.PlateCarree()},
    )
    if n_drivers == 1:
        axes = [axes]

    cmap = plt.cm.RdYlGn
    cmap.set_under('grey')

    for ax, driver in zip(axes, drivers):
        stack     = loo_coefs[driver]   # (left_out_model, lat, lon)
        mean_sign = np.sign(loo_mean[driver].values)

        # Fraction of iterations with same sign as mean
        same_sign = (np.sign(stack.values) == mean_sign[np.newaxis, :, :])
        frac      = same_sign.mean(axis=0)   # (lat, lon)

        frac_da = xr.DataArray(
            frac,
            dims=['lat', 'lon'],
            coords={'lat': loo_mean['lat'], 'lon': loo_mean['lon']},
        )

        frac_da.plot.contourf(
            ax           = ax,
            transform    = ccrs.PlateCarree(),
            cmap         = cmap,
            levels       = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
            vmin         = 0.5,
            vmax         = 1.0,
            add_colorbar = True,
            cbar_kwargs  = {'shrink': 0.7,
                            'label': 'Sign consistency (fraction of LOO)'},
            extend       = 'min',
        )
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'),
                       edgecolor='black', linewidth=0.6)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'),
                       linestyle='--', edgecolor='grey', linewidth=0.4)
        if map_extent is not None:
            ax.set_extent(map_extent, crs=ccrs.PlateCarree())
        if region_box is not None:
            lat_min, lat_max, lon_min, lon_max = region_box
            ax.plot(
                [lon_min, lon_max, lon_max, lon_min, lon_min],
                [lat_min, lat_min, lat_max, lat_max, lat_min],
                transform=ccrs.PlateCarree(),
                color='blue', linewidth=1.5,
            )
        ax.set_title(f'Sign consistency - {driver}',
                     fontsize=11, fontweight='bold')

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 6 - LOO deviation from full-ensemble per model (bar chart)
# ---------------------------------------------------------------------------

def plot_loo_deviation_bars(
    results:      dict,
    region:       tuple,
    region_name:  str  = '',
    drivers:      list = None,
    n_show:       int  = 10,
) -> plt.Figure:
    """
    Horizontal bar chart of per-model deviation from the full-ensemble
    coefficient for each driver, showing only the top ``n_show`` most
    influential models.

    The deviation is expressed as a percentage of the full-ensemble
    regional mean, making it directly comparable across drivers with
    different coefficient magnitudes.

    Parameters
    ----------
    results : dict
        Output from :func:`run_loo_sensitivity`.
    region : tuple
        (lat_min, lat_max, lon_min, lon_max).
    region_name : str
        Label used in the title.
    drivers : list of str, optional
        Subset of drivers. Default: all.
    n_show : int
        Number of top influential models to show per driver. Default 10.

    Returns
    -------
    matplotlib.figure.Figure
    """
    reg_df   = _regional_mean(results['coefficients'], region)
    loo_mean = reg_df.mean(axis=0)

    if drivers is None:
        drivers = reg_df.columns.tolist()

    n_drivers = len(drivers)
    fig, axes = plt.subplots(
        1, n_drivers,
        figsize=(5 * n_drivers, max(4, n_show * 0.35)),
        sharey=False,
    )
    if n_drivers == 1:
        axes = [axes]

    for ax, driver in zip(axes, drivers):
        col      = reg_df[driver]
        mean_val = loo_mean[driver]

        # Percentage deviation from LOO mean
        if abs(mean_val) > 1e-8:
            pct_dev = ((col - mean_val) / abs(mean_val) * 100)
        else:
            pct_dev = col - mean_val

        # Top n_show by absolute deviation
        top = pct_dev.abs().sort_values(ascending=False).head(n_show)
        top_dev = pct_dev.loc[top.index].sort_values()

        colours = ['#d73027' if v > 0 else '#4575b4'
                   for v in top_dev.values]
        ax.barh(top_dev.index, top_dev.values, color=colours, edgecolor='white',
                linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.8)

        # Add value labels
        for model, val in top_dev.items():
            ax.text(val + (0.2 if val >= 0 else -0.2), model,
                    f'{val:+.1f}%', va='center', fontsize=7,
                    ha='left' if val >= 0 else 'right')

        ax.set_xlabel('Deviation from LOO mean (%)', fontsize=9)
        ax.set_title(f'{driver}', fontsize=11, fontweight='bold')
        ax.tick_params(axis='y', labelsize=8)

    title = f'Per-model LOO deviation (top {n_show} most influential)'
    if region_name:
        title += f' - {region_name}'
    fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Table 1 - Quantitative robustness summary (paper-ready)
# ---------------------------------------------------------------------------

def loo_robustness_table(
    results:     dict,
    region:      tuple,
    region_name: str = '',
) -> pd.DataFrame:
    """
    Compute a quantitative robustness summary table suitable for inclusion
    in a paper.

    Metrics per driver:

    - **full_mean** - regional mean coefficient using all models (LOO mean)
    - **loo_std** - std of regional coefficient across LOO iterations
    - **cv_pct** - coefficient of variation as percentage (std / |mean| × 100)
    - **sign_consistency** - fraction of LOO iterations with same sign as mean
    - **max_deviation_pct** - largest single-model deviation as % of mean
    - **most_influential** - model whose dropout causes the largest deviation

    Parameters
    ----------
    results : dict
        Output from :func:`run_loo_sensitivity`.
    region : tuple
        (lat_min, lat_max, lon_min, lon_max).
    region_name : str
        Label printed in the table header.

    Returns
    -------
    pd.DataFrame
        One row per driver. Print with ``print(df.to_string())`` or export
        with ``df.to_latex()`` for direct inclusion in a paper.

    Example
    -------
    >>> df = loo_robustness_table(results, region=(30, 45, -10, 40),
    ...                           region_name='Mediterranean')
    >>> print(df.to_string())
    >>> df.to_csv('loo_robustness_mediterranean.csv')
    >>> print(df.to_latex(float_format='%.3f'))   # paste into paper
    """
    reg_df   = _regional_mean(results['coefficients'], region)
    loo_mean = results['mean']

    rows = []
    for driver in reg_df.columns:
        col      = reg_df[driver]
        mean_val = float(col.mean())

        # Sign consistency across LOO iterations
        mean_sign     = np.sign(mean_val)
        sign_frac     = float((np.sign(col) == mean_sign).mean())

        # CV
        cv_pct = (float(col.std()) / abs(mean_val) * 100
                  if abs(mean_val) > 1e-8 else np.nan)

        # Largest single-model deviation
        deviation     = (col - mean_val).abs()
        most_inf      = deviation.idxmax()
        max_dev_pct   = (float(deviation.max()) / abs(mean_val) * 100
                         if abs(mean_val) > 1e-8 else np.nan)

        # Spatial sign consistency from full map
        stack     = results['coefficients'][driver]
        map_mean  = results['mean'][driver]
        sign_map  = (np.sign(stack.values) == np.sign(map_mean.values)[np.newaxis])
        spatial_sign_consistency = float(sign_map.mean())

        rows.append({
            'driver':                   driver,
            'full_mean':                round(mean_val, 5),
            'loo_std':                  round(float(col.std()), 5),
            'cv_%':                     round(cv_pct, 1),
            'sign_consistency':         round(sign_frac, 3),
            'spatial_sign_consist.':    round(spatial_sign_consistency, 3),
            'max_deviation_%':          round(max_dev_pct, 1),
            'most_influential_model':   most_inf,
        })

    df = pd.DataFrame(rows).set_index('driver')

    header = 'LOO robustness summary'
    if region_name:
        header += f' - {region_name}'
    print(f'\n{header}')
    print('=' * len(header))
    print(df.to_string())
    print()
    return df


# ---------------------------------------------------------------------------
# Plot 7 - Scatter: full-ensemble vs LOO coefficient per model
# ---------------------------------------------------------------------------

def plot_loo_scatter(
    results:      dict,
    region:       tuple,
    region_name:  str  = '',
    drivers:      list = None,
    annotate:     bool = True,
    n_annotate:   int  = 5,
) -> plt.Figure:
    """
    Scatter plot of full-ensemble regional coefficient vs each LOO iteration.

    Each point represents one LOO iteration (one dropped model). The x-axis
    shows the full-ensemble mean coefficient (all models included), and the
    y-axis shows the coefficient when that model is dropped. Points lying
    on the 1:1 line indicate no influence; points far from it identify
    influential models.

    Parameters
    ----------
    results : dict
        Output from :func:`run_loo_sensitivity`.
    region : tuple
        (lat_min, lat_max, lon_min, lon_max).
    region_name : str
        Label used in the title.
    drivers : list of str, optional
        Subset of drivers to plot. Default: all.
    annotate : bool
        Whether to label the most influential points. Default True.
    n_annotate : int
        Number of most influential models to label. Default 5.

    Returns
    -------
    matplotlib.figure.Figure
    """
    reg_df       = _regional_mean(results['coefficients'], region)
    full_mean    = reg_df.mean(axis=0)   # LOO mean as proxy for full ensemble

    if drivers is None:
        drivers = reg_df.columns.tolist()

    n_drivers = len(drivers)
    ncols     = min(n_drivers, 4)
    nrows     = int(np.ceil(n_drivers / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.5 * ncols, 4.5 * nrows),
        squeeze=False,
    )

    for idx, driver in enumerate(drivers):
        ax       = axes[idx // ncols][idx % ncols]
        col      = reg_df[driver]
        ref_val  = full_mean[driver]

        # Deviation from full ensemble
        deviation = (col - ref_val).abs()
        top_idx   = deviation.nlargest(n_annotate).index

        # Colour by deviation magnitude
        norm   = mcolors.Normalize(vmin=0, vmax=float(deviation.max()))
        cmap   = plt.cm.YlOrRd
        colours = [cmap(norm(deviation[m])) for m in col.index]

        sc = ax.scatter(
            [ref_val] * len(col),   # x: full ensemble value (same for all)
            col.values,             # y: LOO value
            c=[norm(deviation[m]) for m in col.index],
            cmap=cmap,
            s=60, zorder=4, edgecolors='grey', linewidths=0.4,
        )
        plt.colorbar(sc, ax=ax, label='|deviation| from mean')

        # 1:1 reference line
        all_vals = list(col.values) + [ref_val]
        vmin_    = min(all_vals) * 1.15
        vmax_    = max(all_vals) * 1.15
        ref_line = np.linspace(vmin_, vmax_, 100)
        ax.plot(ref_line, ref_line, 'k--', linewidth=1.0,
                label='1:1 line', zorder=2)

        # Tolerance band ± 5% of ref_val
        tol = abs(ref_val) * 0.05
        ax.fill_between(ref_line, ref_line - tol, ref_line + tol,
                        alpha=0.12, color='grey', label='±5% band')

        # Annotate most influential models
        if annotate:
            for model in top_idx:
                ax.annotate(
                    model,
                    xy=(ref_val, col[model]),
                    xytext=(6, 0),
                    textcoords='offset points',
                    fontsize=7, color='darkred',
                    va='center',
                    arrowprops=dict(arrowstyle='-', color='darkred',
                                   lw=0.5),
                )

        ax.set_xlabel('Full-ensemble mean coefficient', fontsize=9)
        ax.set_ylabel('LOO coefficient (model dropped)', fontsize=9)
        ax.set_title(f'{driver}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=7)

    # Hide unused axes
    for idx in range(n_drivers, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    title = 'Full-ensemble vs LOO coefficient'
    if region_name:
        title += f' - {region_name}'
    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Gridpoint LOO analysis
# ---------------------------------------------------------------------------

def run_loo_gridpoint(
    results:   dict,
    lat:       float,
    lon:       float,
    drivers:   list = None,
) -> pd.DataFrame:
    """
    Extract and summarise LOO coefficients at a single gridpoint.

    Selects the nearest gridpoint to (lat, lon) from the LOO coefficient
    stack and returns a DataFrame with one row per LOO iteration (dropped
    model) and one column per driver, plus summary statistics appended at
    the bottom.

    Parameters
    ----------
    results : dict
        Output from :func:`run_loo_sensitivity`.
    lat : float
        Target latitude in degrees.
    lon : float
        Target longitude in degrees (−180 to 180).
    drivers : list of str, optional
        Subset of drivers. Default: all.

    Returns
    -------
    pd.DataFrame
        Index: dropped model names + ['MEAN', 'STD', 'CV_%',
        'SIGN_CONSISTENCY']. Columns: driver names.

    Example
    -------
    >>> # Check robustness at Rome (41.9°N, 12.5°E)
    >>> df = run_loo_gridpoint(results, lat=41.9, lon=12.5)
    >>> print(df)
    """
    loo_coefs = results['coefficients']
    if drivers is not None:
        loo_coefs = loo_coefs[drivers]

    # Nearest gridpoint selection
    gp = loo_coefs.sel(lat=lat, lon=lon, method='nearest')
    actual_lat = float(gp['lat'])
    actual_lon = float(gp['lon'])
    print(f'Selected gridpoint: lat={actual_lat:.2f}°, lon={actual_lon:.2f}°')

    driver_names = list(loo_coefs.data_vars)
    models       = gp['left_out_model'].values.tolist()

    # Build DataFrame of LOO values
    rows = {}
    for model in models:
        rows[model] = {
            d: float(gp[d].sel(left_out_model=model))
            for d in driver_names
        }
    df = pd.DataFrame(rows).T
    df.index.name = 'left_out_model'

    # Append summary statistics
    mean_row  = df.mean(axis=0).rename('MEAN')
    std_row   = df.std(axis=0).rename('STD')
    cv_row    = (std_row / mean_row.abs() * 100).rename('CV_%')
    sign_cons = (df.apply(lambda col: (np.sign(col) == np.sign(col.mean())).mean())
                 .rename('SIGN_CONSISTENCY'))

    summary = pd.DataFrame([mean_row, std_row, cv_row, sign_cons])
    df_full = pd.concat([df, summary])

    print(f'\nLOO coefficients at ({actual_lat:.2f}°N, {actual_lon:.2f}°E):')
    print(df_full.to_string(float_format='{:.5f}'.format))
    return df_full


def plot_loo_gridpoint(
    results:     dict,
    lat:         float,
    lon:         float,
    drivers:     list = None,
    n_annotate:  int  = 3,
) -> plt.Figure:
    """
    Bar chart of LOO coefficients at a single gridpoint, one panel per driver.

    Each bar represents one LOO iteration. Bars are coloured by deviation
    from the gridpoint LOO mean. The dashed line shows the LOO mean.
    The ``n_annotate`` most influential models are labelled.

    Parameters
    ----------
    results : dict
        Output from :func:`run_loo_sensitivity`.
    lat : float
        Target latitude in degrees.
    lon : float
        Target longitude in degrees (−180 to 180).
    drivers : list of str, optional
        Subset of drivers. Default: all.
    n_annotate : int
        Number of most influential models to label. Default 3.

    Returns
    -------
    matplotlib.figure.Figure

    Example
    -------
    >>> # Robustness at Rome
    >>> fig = plot_loo_gridpoint(results, lat=41.9, lon=12.5)
    >>> fig.savefig('loo_rome.png', dpi=150, bbox_inches='tight')
    """
    loo_coefs = results['coefficients']
    if drivers is not None:
        loo_coefs = loo_coefs[[d for d in drivers
                                if d in loo_coefs.data_vars]]

    # Nearest gridpoint
    gp         = loo_coefs.sel(lat=lat, lon=lon, method='nearest')
    actual_lat = float(gp['lat'])
    actual_lon = float(gp['lon'])
    driver_names = list(loo_coefs.data_vars)
    models       = gp['left_out_model'].values.tolist()

    n_drivers = len(driver_names)
    fig, axes = plt.subplots(
        1, n_drivers,
        figsize=(5 * n_drivers, max(5, len(models) * 0.22)),
        sharey=False,
    )
    if n_drivers == 1:
        axes = [axes]

    for ax, driver in zip(axes, driver_names):
        vals     = np.array([float(gp[driver].sel(left_out_model=m))
                             for m in models])
        mean_val = float(np.mean(vals))
        deviations = np.abs(vals - mean_val)
        top_idx  = np.argsort(deviations)[-n_annotate:]

        norm    = mcolors.Normalize(vmin=0, vmax=deviations.max())
        cmap    = plt.cm.YlOrRd
        colours = [cmap(norm(d)) for d in deviations]

        bars = ax.barh(models, vals, color=colours,
                       edgecolor='white', linewidth=0.4)
        ax.axvline(mean_val, color='black', linewidth=1.2,
                   linestyle='--', label=f'LOO mean = {mean_val:.4f}')

        # Label most influential
        for i in top_idx:
            ax.annotate(
                models[i],
                xy=(vals[i], i),
                xytext=(4 if vals[i] >= mean_val else -4, 0),
                textcoords='offset points',
                fontsize=7, color='darkred',
                ha='left' if vals[i] >= mean_val else 'right',
                va='center',
            )

        # CV and sign consistency as subtitle
        cv  = float(np.std(vals) / abs(mean_val) * 100) if abs(mean_val) > 1e-8 else np.nan
        sc  = float(np.mean(np.sign(vals) == np.sign(mean_val)))
        ax.set_title(
            f'{driver}\nCV={cv:.1f}%  sign_consist.={sc:.3f}',
            fontsize=10, fontweight='bold',
        )
        ax.set_xlabel('LOO coefficient', fontsize=9)
        ax.tick_params(axis='y', labelsize=7)
        ax.legend(fontsize=7)

    fig.suptitle(
        f'LOO coefficients at gridpoint '
        f'({actual_lat:.2f}°N, {actual_lon:.2f}°E)',
        fontsize=12, fontweight='bold',
    )
    fig.tight_layout()
    return fig