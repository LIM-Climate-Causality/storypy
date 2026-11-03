"""
storypy.compute._jet_index
===========================

Jet-latitude / jet-speed target builder - the scalar analogue of
ESMValProcessor.process_var() for the eddy-driven jet, built to drop
straight into storypy.compute.compute_regression / storyline_evaluation
the same way target_pr.nc or target_u850.nc do.

Why this is a separate function rather than reusing process_var()
-------------------------------------------------------------------
process_var() differences the *spatial* U850 pattern (future minus
baseline, gridpoint by gridpoint) and normalizes by GW - that's right
for a variable like precipitation, where "change at this gridpoint" is
the quantity of interest. The jet doesn't work that way: you have to
find the jet's latitude in the *actual* wind field for each period
first, and only then take the future-minus-baseline change in that
latitude. Running the parabolic jet-finder on the difference field
instead would locate the latitude of peak *change* in wind, not the
latitude of the jet - a different, less interpretable quantity.

Verified against the real storypy.preprocess._diagnostics:

- seasonal_data_months(data, months) groups by (calendar-aware) year,
  averages within each year, and renames the grouped coordinate to
  'time' as a plain integer-year axis - not a datetime index. So
  .sel(time=slice(int(y0), int(y1))).mean('time') is exactly right,
  and matches what clim_change() itself does internally.
- clim_change() calls adjust_longitudes() as its very first step,
  before any box/region selection - so this module does the same
  before applying lon_bounds, rather than assuming a -180/180 grid.

Multi-member variant_selection (this revision)
------------------------------------------------
Matches ESMValProcessor._combine_and_save()'s grouping exactly: ESMValTool
metadata is grouped by 'dataset' first, then by 'alias' within it - alias
is the per-member identifier (e.g. "ACCESS-CM2_r1i1p1f1"). The earlier
draft of this module looped over both levels but only ever recorded the
outer `dataset` name against each member's result, so a multi-member
model like CanESM5 (50 members in Richard's ghosh_2023 run) would end up
with 50 rows all labelled "CanESM5" in target_jetlat.nc instead of one
averaged row - silently breaking the 'model' coordinate's uniqueness
that compute_regression()/SpatialRegression rely on.

This revision computes jetlat/jetspeed per *alias* (member) first, then
reduces to one value per base model using the same
`base = alias.split('_r')[0]` grouping and 'mean' | 'first' | 'last'
variant_selection strategy as _combine_and_save(), and attaches the same
n_members / member_ids coordinates target_pr.nc / target_u850.nc carry -
so target_jetlat.nc's model list and member provenance now match the
rest of the pipeline's target files exactly, not just in name.

Output format
-------------
target_jetlat.nc / target_jetspeed.nc: a single DataArray named
'jetlat' / 'jetspeed' with dims ('model',), plus 'n_members' and
'member_ids' coords on 'model' - same convention as target_pr.nc.
compute_regression()/SpatialRegression don't require spatial dims:
xr.apply_ufunc(..., input_core_dims=[["model"]], ...) reduces over
'model' regardless of what other dims (or none) remain, and
SpatialRegression.perform_regression()'s variable-name rename logic
falls into its generic `else` branch for any name it doesn't
special-case (ua/sst/tas/pr) - so naming the DataArray 'jetlat' to
match target_variable=["jetlat"] is all that's needed for it to work
unmodified.
"""

import os
from collections import defaultdict

import numpy as np
import xarray as xr

from storypy.preprocess._diagnostics import seasonal_data_months, adjust_longitudes
from storypy._compat import group_metadata


def eddy_driven_jet_index(u850, lat_bounds=(30, 70), lon_bounds=None,
                           lat_name="lat", lon_name="lon"):
    """
    Woollings et al. (2010)-style eddy-driven jet latitude/speed index.

    Parameters
    ----------
    u850 : xarray.DataArray
        850 hPa zonal wind for a single period-mean field (dims must
        include lat_name, and lon_name if lon_bounds is given).
    lat_bounds : tuple
        Latitude search window, restricted away from the subtropical
        jet. NH *summer* eddy-driven jet sits further poleward than in
        winter - check this against your own climatology rather than
        reusing DJF bounds.
    lon_bounds : tuple or None
        Longitude sector to average first, in -180/180 convention,
        e.g. (-60, 0) for Atlantic. None = whole-hemisphere zonal mean.
        Longitudes are normalized to -180/180 via adjust_longitudes()
        by the caller before this runs.
    lat_name, lon_name : str
        Dimension names in `u850`.

    Returns
    -------
    jet_lat, jet_speed : float
        Sub-grid jet latitude (deg) and speed (m/s).
    """
    u = u850.sel({lat_name: slice(*lat_bounds)})
    if lon_bounds is not None:
        u = u.sel({lon_name: slice(*lon_bounds)})
    if lon_name in u.dims:
        u = u.mean(lon_name)
    u = u.load()

    lat = u[lat_name]
    imax = int(u.argmax(lat_name, skipna=True).values)
    i0 = min(max(imax, 1), lat.size - 2)

    y0 = float(u.isel({lat_name: i0 - 1}))
    y1 = float(u.isel({lat_name: i0}))
    y2 = float(u.isel({lat_name: i0 + 1}))
    x0 = float(lat.isel({lat_name: i0 - 1}))
    x1 = float(lat.isel({lat_name: i0}))
    x2 = float(lat.isel({lat_name: i0 + 1}))

    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2**2 * (y0 - y1) + x1**2 * (y2 - y0) + x0**2 * (y1 - y2)) / denom
    c = (x1 * x2 * (x1 - x2) * y0 + x2 * x0 * (x2 - x0) * y1
         + x0 * x1 * (x0 - x1) * y2) / denom

    jet_lat = -b / (2 * a)
    jet_speed = a * jet_lat**2 + b * jet_lat + c
    return jet_lat, jet_speed


def _jet_index_per_member(config, main_config, lat_bounds, lon_bounds):
    """
    Compute jetlat/jetspeed for every (dataset, alias) member individually.

    Returns
    -------
    dict[str, dict[str, float]]
        {'jetlat': {alias: value, ...}, 'jetspeed': {alias: value, ...}}
        keyed by the ESMValTool *alias* (member-level name), not the
        base dataset name - grouping/averaging across members happens
        one level up, in compute_jet_target(), mirroring
        ESMValProcessor._combine_and_save().
    """
    meta_dataset = group_metadata(config["input_data"].values(), "dataset")
    period1, period2 = main_config["period1"], main_config["period2"]
    season = list(main_config["season"])

    jetlat_by_alias = {}
    jetspeed_by_alias = {}

    for dataset, alias_list in meta_dataset.items():
        meta_group = group_metadata(alias_list, "alias")
        for alias, alias_members in meta_group.items():
            try:
                u850_files = {
                    m['variable_group']: xr.open_dataset(m['filename'])[m['short_name']]
                    for m in alias_members if m['variable_group'] == 'u850'
                }
                gw_files = {
                    m['variable_group']: xr.open_dataset(m['filename'])[m['short_name']]
                    for m in alias_members if m['variable_group'] == 'gw'
                }
                if 'u850' not in u850_files or 'gw' not in gw_files:
                    print(f"No u850/gw data for alias '{alias}' in dataset '{dataset}'; skipping.")
                    continue

                # Same order as clim_change(): normalize longitudes first.
                u850_da = adjust_longitudes(u850_files['u850'], lon_dim='lon')

                u_seas = seasonal_data_months(u850_da, season)
                u_ref = u_seas.sel(time=slice(int(period1[0]), int(period1[1]))).mean('time')
                u_fut = u_seas.sel(time=slice(int(period2[0]), int(period2[1]))).mean('time')

                lat_ref, spd_ref = eddy_driven_jet_index(u_ref, lat_bounds, lon_bounds)
                lat_fut, spd_fut = eddy_driven_jet_index(u_fut, lat_bounds, lon_bounds)

                gw_seas = seasonal_data_months(gw_files['gw'], season)
                gw_ref = gw_seas.sel(time=slice(int(period1[0]), int(period1[1]))).mean('time')
                gw_fut = gw_seas.sel(time=slice(int(period2[0]), int(period2[1]))).mean('time')
                gw_scalar = float(gw_fut.values) - float(gw_ref.values)

                if gw_scalar == 0.0 or np.isnan(gw_scalar):
                    print(f"Warning: GW scalar is {gw_scalar} for alias '{alias}'; skipping.")
                    continue

                jetlat_by_alias[alias] = (lat_fut - lat_ref) / gw_scalar
                jetspeed_by_alias[alias] = (spd_fut - spd_ref) / gw_scalar

            except KeyError as e:
                print(f"KeyError: {e}. Skipping alias {alias} in dataset {dataset}.")
                continue
            except Exception as e:
                print(f"Error processing alias '{alias}' in dataset '{dataset}': {e}")
                continue

    return {'jetlat': jetlat_by_alias, 'jetspeed': jetspeed_by_alias}


def _reduce_variants(values_by_alias, variant_selection):
    """
    Reduce member-level {alias: value} to one value per base model,
    using exactly the grouping/selection logic of
    ESMValProcessor._combine_and_save(): base model = alias split on
    '_r' (strips the realization suffix, e.g. "ACCESS-CM2_r1i1p1f1"
    -> "ACCESS-CM2"); 'mean' | 'first' | 'last' across members, sorted
    by alias name within each group for determinism.

    Returns
    -------
    (models, values, n_members_list, member_ids_list) - parallel lists,
    sorted by base model name (matching _combine_and_save's
    `sorted(groups.items())`).
    """
    if variant_selection not in ('mean', 'first', 'last'):
        raise ValueError(
            f"variant_selection must be 'mean', 'first', or 'last', "
            f"got '{variant_selection}'"
        )

    groups = defaultdict(list)
    for alias, value in values_by_alias.items():
        base = alias.split('_r')[0]
        groups[base].append((alias, value))

    models, values, n_members_list, member_ids_list = [], [], [], []

    for base_model, members in sorted(groups.items()):
        members_sorted = sorted(members, key=lambda x: x[0])

        if variant_selection == 'last':
            chosen_value = members_sorted[-1][1]
            member_aliases = [members_sorted[-1][0]]
        elif variant_selection == 'first':
            chosen_value = members_sorted[0][1]
            member_aliases = [members_sorted[0][0]]
        else:  # 'mean'
            chosen_value = float(np.mean([v for _, v in members_sorted]))
            member_aliases = [a for a, _ in members_sorted]

        models.append(base_model)
        values.append(chosen_value)
        n_members_list.append(len(member_aliases))
        member_ids_list.append('|'.join(member_aliases))

    return models, values, n_members_list, member_ids_list


def compute_jet_target(config, main_config, lat_bounds=(30, 70), lon_bounds=None,
                        variant_selection='mean'):
    """
    Build target_jetlat.nc and target_jetspeed.nc from ESMValTool U850 +
    GW metadata, in the same 'model'-indexed format (including
    n_members/member_ids coords) that compute_regression() and
    storyline_evaluation() already consume for target_pr.nc/target_u850.nc.

    Parameters
    ----------
    config : dict
        Parsed ESMValTool config from parse_config() - same object you
        pass into ESMValProcessor. Must contain a 'u850' variable_group
        (raw JJA-mean zonal wind, NOT the already-differenced target)
        and a 'gw' variable_group, exactly as process_var() requires.
    main_config : dict
        Needs 'work_dir', 'period1', 'period2', 'season' (list of
        months, e.g. [6,7,8] for JJA).
    lat_bounds, lon_bounds :
        Passed to eddy_driven_jet_index() - see basin-split decision in
        the study design doc before picking lon_bounds.
    variant_selection : str
        'mean' (default, matches Zappa & Shepherd 2017 and
        ESMValProcessor's default), 'first', or 'last' - how to combine
        multiple ensemble members of the same model.

    Returns
    -------
    (jetlat_by_model, jetspeed_by_model, models) : lists, aligned by index.
    """
    per_member = _jet_index_per_member(config, main_config, lat_bounds, lon_bounds)

    out_dir = main_config['work_dir']
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for name in ('jetlat', 'jetspeed'):
        models, values, n_members_list, member_ids_list = _reduce_variants(
            per_member[name], variant_selection
        )
        da = xr.DataArray(values, dims='model', coords={'model': models}, name=name)
        da = da.assign_coords(
            n_members=('model', n_members_list),
            member_ids=('model', member_ids_list),
        )
        da.to_dataset(name=name).to_netcdf(os.path.join(out_dir, f'target_{name}.nc'))
        print(f"Saved {len(models)} models to {os.path.join(out_dir, f'target_{name}.nc')} "
              f"(variant_selection='{variant_selection}')")
        results[name] = (models, values)

    jetlat_models, jetlat_values = results['jetlat']
    _, jetspeed_values = results['jetspeed']
    return jetlat_values, jetspeed_values, jetlat_models