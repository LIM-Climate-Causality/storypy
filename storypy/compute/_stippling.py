"""
storypy.compute._stippling
==========================

Significance testing for spatial precipitation change maps.

Provides :class:`StipplingComputer` with two methods:

* :meth:`beta_significance` -- model agreement test (Zappa & Shepherd 2017
  beta criterion): flags gridpoints where >X% of models agree on sign of
  change. Works directly from ``target_<var>.nc``.

* :meth:`gamma_significance` -- signal-to-noise test (Zappa & Shepherd 2017
  gamma criterion): flags gridpoints where the forced signal exceeds internal
  variability. Requires piControl NetCDF files from a separate ESMValTool run.

Example
-------
>>> from storypy.compute._stippling import StipplingComputer
>>>
>>> # Beta -- works from target_pr.nc
>>> sc = StipplingComputer(work_dir='./output', target_variable='pr')
>>> pos, neg = sc.beta_significance(agreement_threshold=0.9)
>>>
>>> # Gamma -- needs piControl ESMValTool config
>>> from storypy.preprocess import parse_config
>>> picontrol_cfg = parse_config('/path/to/picontrol_run/settings.yml')
>>> sc_gamma = StipplingComputer(
...     work_dir          = './output',
...     target_variable   = 'pr',
...     picontrol_config  = picontrol_cfg,
...     variable_group    = 'pr_picontrol',
...     season_months     = [12, 1, 2],
...     variant_selection = 'mean',
... )
>>> gamma = sc_gamma.gamma_significance(rolling_window=30)
"""

import os
import numpy as np
import xarray as xr
from collections import defaultdict


class StipplingComputer:
    """
    Compute significance masks for spatial precipitation change maps.

    Parameters
    ----------
    work_dir : str
        Directory containing ``target_<var>.nc`` and where
        ``stippling/`` output will be written.
    target_variable : str
        Variable name, e.g. ``'pr'`` or ``'ua'``. Default ``'pr'``.
    picontrol_config : dict, optional
        ESMValTool config dict from :func:`storypy.preprocess.parse_config`
        pointing at a piControl diagnostic run. Required for
        :meth:`gamma_significance`. Mirrors the ``config`` argument used
        in the original ``gamma_significance_test.py`` script.
    variable_group : str
        Variable group name in the ESMValTool piControl metadata.
        e.g. ``'pr_picontrol'``. Default ``'pr_picontrol'``.
    season_months : list[int] or None
        Months to select before computing rolling statistics.
        e.g. ``[12, 1, 2]`` for DJF, ``[11, 12, 1, 2, 3]`` for NDJFM.
        Default ``None`` (use all months).
    variant_selection : str
        How to handle multi-member piControl runs. One of:

        - ``'mean'``  : average gamma across all variants (default,
                        matches Zappa & Shepherd 2017).
        - ``'first'`` : use first variant alphabetically.
        - ``'last'``  : use last variant alphabetically.
    """

    def __init__(
        self,
        work_dir:          str,
        target_variable:   str  = 'pr',
        picontrol_config:  dict = None,
        variable_group:    str  = 'pr_picontrol',
        season_months:     list = None,
        variant_selection: str  = 'mean',
    ):
        self.work_dir          = work_dir
        self.var               = target_variable
        self.picontrol_config  = picontrol_config
        self.variable_group    = variable_group
        self.season_months     = season_months
        self.variant_selection = variant_selection

        # # Load target dataset
        # target_path = os.path.join(work_dir, f'target_{target_variable}.nc')
        # if not os.path.exists(target_path):
        #     raise FileNotFoundError(
        #         f"Target file not found: {target_path}. "
        #         "Run ESMValProcessor.process_var() first."
        #     )
        # self.target_ds = xr.open_dataset(target_path)

        # Prepare output directory
        out_dir = os.path.join(work_dir, 'stippling')
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir

    # ------------------------------------------------------------------
    # Beta significance -- model agreement on sign of change
    # ------------------------------------------------------------------
    def beta_significance(
        self,
        agreement_threshold: float = 0.9,
    ):
        """
        Compute model agreement stippling mask (beta criterion).

        At each gridpoint counts how many models show positive vs negative
        change. Flags points where ``agreement_threshold`` fraction of
        models agree on sign.

        This replicates the logic of ``beta_significance_test.py`` but
        works directly from ``target_<var>.nc`` rather than ESMValTool
        metadata, so no additional ESMValTool run is needed.

        Parameters
        ----------
        agreement_threshold : float
            Fraction of models required to agree on sign. Default 0.9
            (90%), matching the original script.

        Returns
        -------
        (xarray.DataArray, xarray.DataArray)
            Tuple ``(positives, negatives)`` -- binary 2D arrays where 1
            means the threshold is exceeded. Also saved as NetCDF files
            under ``{work_dir}/stippling/``:

            - ``number_of_models_positive_trend_CMIP6.nc``
            - ``number_of_models_negative_trend_CMIP6.nc``

        Notes
        -----
        The target DataArray must have a ``model`` dimension. Each slice
        is the climatological change already normalised by GW, averaged
        across ensemble members per model (as produced by
        :class:`storypy.preprocess.ESMValProcessor`).
        """
        target_path = os.path.join(self.work_dir, f'target_{self.var}.nc')
        if not os.path.exists(target_path):
            raise FileNotFoundError(
                f"Target file not found: {target_path}. "
                "Run ESMValProcessor.process_var() first."
            )
        target_ds = xr.open_dataset(target_path)
        da = target_ds[self.var]   # dims: (model, lat, lon)

        n_models  = da.sizes['model']
        threshold = n_models * agreement_threshold

        # Count models with positive / negative change at each gridpoint
        positives = (da > 0).sum(dim='model')
        negatives = (da < 0).sum(dim='model')

        # Binary mask: 1 where threshold is exceeded
        pos_mask = xr.where(positives >= threshold, 1, 0)
        neg_mask = xr.where(negatives >= threshold, 1, 0)

        # Assign spatial coordinates from first model slice
        template = da.isel(model=0).drop_vars('model')
        pos_out  = pos_mask.assign_coords(template.coords)
        neg_out  = neg_mask.assign_coords(template.coords)
        pos_out.name = 'pr'
        neg_out.name = 'pr'

        # Save
        pos_out.to_dataset(name='pr').to_netcdf(
            os.path.join(self.out_dir,
                         'number_of_models_positive_trend_CMIP6.nc')
        )
        neg_out.to_dataset(name='pr').to_netcdf(
            os.path.join(self.out_dir,
                         'number_of_models_negative_trend_CMIP6.nc')
        )
        print(
            f"Beta significance saved to {self.out_dir} "
            f"(threshold={agreement_threshold:.0%}, n_models={n_models})"
        )
        return pos_out, neg_out

    # ------------------------------------------------------------------
    # Internal helper -- load piControl data via ESMValTool metadata
    # ------------------------------------------------------------------
    def _load_picontrol(self):
        """
        Load piControl DataArrays grouped by base model name.

        Reads ESMValTool metadata from ``picontrol_config`` exactly as
        the original ``gamma_significance_test.py`` script did:

            meta_dataset = group_metadata(..., "dataset")
            meta         = group_metadata(..., "alias")

        Returns
        -------
        dict[str, list[tuple[str, xarray.DataArray]]]
            Maps base model name to a list of ``(alias, DataArray)``
            tuples, one per available variant.
        """
        from esmvaltool.diag_scripts.shared import group_metadata

        if self.picontrol_config is None:
            raise ValueError(
                "picontrol_config must be provided for gamma_significance. "
                "Pass the ESMValTool config dict obtained from "
                "parse_config('/path/to/picontrol_settings.yml')."
            )

        # Mirror original script: group by both dataset and alias
        meta_dataset = group_metadata(
            self.picontrol_config["input_data"].values(), "dataset"
        )
        meta = group_metadata(
            self.picontrol_config["input_data"].values(), "alias"
        )

        result = defaultdict(list)

        for dataset, dataset_list in meta_dataset.items():
            for alias, alias_list in meta.items():
                for m in alias_list:
                    if (m["dataset"] == dataset and
                            m["variable_group"] == self.variable_group):
                        da = xr.open_dataset(m["filename"])[m["short_name"]]
                        result[dataset].append((alias, da))

        if not result:
            raise RuntimeError(
                f"No piControl data found for variable_group="
                f"'{self.variable_group}'. Check that the recipe "
                f"variable group name matches."
            )

        return result

    # ------------------------------------------------------------------
    # Gamma significance -- signal-to-noise ratio from piControl
    # ------------------------------------------------------------------
    def gamma_significance(
        self,
        rolling_window: int = 30,
    ):
        """
        Compute signal-to-noise stippling mask (gamma criterion).
 
        Saves two NetCDF files under ``{work_dir}/stippling/``:
 
        - ``gamma_forced_CMIP6.nc`` : median signal-to-noise ratio across
        models (f = signal / noise).
        - ``summatory_denom_CMIP6.nc`` : sum of ``1 / noise^2`` across all
        models, used as the denominator weight in the Zappa & Shepherd
        significance formula.
 
        Parameters
        ----------
        rolling_window : int
            Window length in years. Default 30.
 
        Returns
        -------
        (xarray.DataArray, xarray.DataArray)
            Tuple ``(gamma_median, summatory_denom)``.
        """
        model_data = self._load_picontrol()
 
        # Load forced signal from target_pr.nc
        target_path = os.path.join(self.work_dir, f'target_{self.var}.nc')
        if not os.path.exists(target_path):
            raise FileNotFoundError(
                f"Target file not found: {target_path}. "
                "Run ESMValProcessor.process_var() first."
            )
        target_ds     = xr.open_dataset(target_path)
        forced_signal = target_ds[self.var].mean(dim='model')
 
        gamma_list       = []
        summatory_list   = []   # accumulates 1/noise^2 per model
 
        for model, members in sorted(model_data.items()):
 
            members_sorted = sorted(members, key=lambda x: x[0])
 
            if self.variant_selection == 'first':
                selected = [members_sorted[0][1]]
            elif self.variant_selection == 'last':
                selected = [members_sorted[-1][1]]
            else:
                selected = [da for _, da in members_sorted]
 
            model_gammas    = []
            model_inv_noise = []   # 1/noise^2 per variant
 
            for da in selected:
 
                if (self.season_months is not None and
                        da.sizes.get('time', 0) > 100):
                    from storypy.preprocess._diagnostics import (
                        seasonal_data_months
                    )
                    da = seasonal_data_months(da, self.season_months)
 
                if self.var == 'pr':
                    da = da * 86400
 
                if da.sizes.get('time', 0) < rolling_window * 2:
                    print(
                        f"  Skipping {model}: only "
                        f"{da.sizes.get('time', 0)} time steps "
                        f"(need at least {rolling_window * 2})"
                    )
                    continue
 
                # Internal variability (noise): std of non-overlapping piControl rolling means
                rolling_mean = (
                    da.rolling(time=rolling_window, center=True)
                    .mean()
                    .dropna('time')
                )
                non_overlap = rolling_mean.isel(
                    time=slice(0, None, rolling_window)
                )
                noise = non_overlap.std(dim='time')
 
                # Wrap noise lons to -180/180 before interp so it aligns
                # with forced_signal (which is already in -180/180 convention)
                if noise['lon'].values.max() > 180:
                    noise = noise.assign_coords(
                        lon=((noise['lon'].values + 180) % 360 - 180)
                    )
                    noise = noise.sortby('lon')
 
                # Forced signal: MEM change from target_pr.nc interpolated to piControl grid
                signal = forced_signal.interp_like(noise)
 
                # Gamma = |forced signal| / piControl noise
                gamma = xr.where(noise > 0, np.abs(signal) / noise, np.nan)
                # 1/noise^2 — denominator weight (original: summatory_denom)
                inv_noise2 = xr.where(noise > 0, 1.0 / (noise ** 2), np.nan)
 
                model_gammas.append(gamma)
                model_inv_noise.append(inv_noise2)
 
            if not model_gammas:
                print(f"  No valid gamma for {model}, skipping.")
                continue
 
            # Average across variants for this model
            if len(model_gammas) == 1:
                model_gamma     = model_gammas[0]
                model_inv_noise2 = model_inv_noise[0]
            else:
                model_gamma      = (xr.concat(model_gammas, dim='variant')
                                    .mean(dim='variant'))
                model_inv_noise2 = (xr.concat(model_inv_noise, dim='variant')
                                    .mean(dim='variant'))
 
            gamma_list.append(model_gamma)
            summatory_list.append(model_inv_noise2)
            print(
                f"  Gamma computed for {model} "
                f"({len(model_gammas)} variant(s))"
            )
 
        if not gamma_list:
            raise RuntimeError(
                "No gamma values were computed. Check that piControl "
                "files were loaded correctly and have sufficient time steps."
            )
 
        # Median gamma across models
        gamma_stack  = xr.concat(gamma_list, dim='model')
        gamma_median = gamma_stack.median(dim='model')
        gamma_median.name = 'gamma'
 
        # Sum of 1/noise^2 across models — matches original summatory_denom
        summatory_stack = xr.concat(summatory_list, dim='model')
        summatory_denom = summatory_stack.sum(dim='model')
        summatory_denom.name = 'summatory_denom'
 
        gamma_median = gamma_median.assign_coords(
            lon=(gamma_median.lon.values + 180) % 360 - 180
        )
        gamma_median = gamma_median.sortby('lon')
 
        summatory_denom = summatory_denom.assign_coords(
            lon=(summatory_denom.lon.values + 180) % 360 - 180
        )
        summatory_denom = summatory_denom.sortby('lon')
 
        # Save both outputs
        gamma_path = os.path.join(self.out_dir, 'gamma_forced_CMIP6.nc')
        denom_path = os.path.join(self.out_dir, 'summatory_denom_CMIP6.nc')
 
        gamma_median.to_netcdf(gamma_path)
        summatory_denom.to_netcdf(denom_path)
 
        print(
            f"Gamma significance saved to {self.out_dir} "
            f"({len(gamma_list)} models):\n"
            f"  {gamma_path}\n"
            f"  {denom_path}"
        )
        return gamma_median, summatory_denom