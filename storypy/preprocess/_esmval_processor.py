import os
from storypy.utils import np, xr

from ._diagnostics import clim_change, seasonal_data_months, test_mean_significance
from storypy.evaluate.plot import plot_precipitation_change, plot_function

class ESMValProcessor:
    def __init__(self, config, user_config):
        """
        Initialize the processor with esmvaltool config and user-defined config.
        """
        self.config = config
        self.user_config = user_config
        xr.set_options(keep_attrs=True)
        self._compute_bounding_box()
        # Unpack frequently used config values
        uc = self.user_config
        self.region_method = uc["region_method"]
        self.box = uc["box"]
        self.period1 = uc["period1"]
        self.period2 = uc["period2"]
        self.region_id = uc["region_id"]
        self.season = uc["season"]
        self.region_extents = uc["region_extents"]
        self.var_names = uc["var_name"]
        self.titles = uc.get("titles", [])
        # Prepare storage
        self.ensemble_changes = {var: [] for var in self.var_names}
        self.time_series_changes = {var: [] for var in self.var_names}
        self.all_model_names = set()
        # Prepare output dirs
        os.makedirs(self.config["plot_dir"], exist_ok=True)

    def _compute_bounding_box(self):
        # Calculate and expand bounding box by 5 degrees
        extents = self.user_config['region_extents']
        all_lat_min = min(r[0] for r in extents)
        all_lat_max = max(r[1] for r in extents)
        all_lon_min = min(r[2] for r in extents)
        all_lon_max = max(r[3] for r in extents)
        self.user_config['box'] = {
            'lat_min': max(all_lat_min - 5, -90),
            'lat_max': min(all_lat_max + 5, 90),
            'lon_min': all_lon_min - 5 if all_lon_min - 5 >= -180 else -180,
            'lon_max': all_lon_max + 5 if all_lon_max + 5 <= 180 else 180
        }

    def _load_metadata(self):
        # Group esmvaltool metadata by dataset and alias
        meta_ds = group_metadata(self.config["input_data"].values(), "dataset")
        meta_alias = group_metadata(self.config["input_data"].values(), "alias")
        return meta_ds, meta_alias

    def _process_data(self):
        meta_dataset, _ = self._load_metadata()
        # Loop through each dataset grouping
        for dataset, alias_list in meta_dataset.items():
            meta_group = group_metadata(alias_list, "alias")
            for alias, alias_members in meta_group.items():
                self.all_model_names.add(alias)
                for var in self.var_names:
                    try:
                        # load target and gw DataArrays
                        target_data = {
                            m['variable_group']: xr.open_dataset(m['filename'])[m['short_name']]
                            for m in alias_members
                            if m['dataset'] == dataset and m['variable_group'] == var
                        }
                        target_gw = {
                            m['variable_group']: xr.open_dataset(m['filename'])[m['short_name']]
                            for m in alias_members
                            if m['dataset'] == dataset and m['variable_group'] == 'gw'
                        }
                        # seasonal subset & unit conversion
                        da = seasonal_data_months(target_data[var], list(self.season))
                        if var == 'pr':
                            da = da * 86400
                        # climatological changes
                        change = clim_change(da, period1=self.period1, period2=self.period2,
                                             region_method=self.region_method, box=self.box,
                                             region_id=self.region_id, season=self.season,
                                             preserve_time_series=False)
                        change_ts = clim_change(da, period1=self.period1, period2=self.period2,
                                                region_method=self.region_method, box=self.box,
                                                region_id=self.region_id, season=self.season,
                                                preserve_time_series=True)
                        # process gw
                        gw_seas = seasonal_data_months(target_gw['gw'], list(self.season))
                        has_spatial = ('lat' in gw_seas.dims and 'lon' in gw_seas.dims)
                        if has_spatial:
                            gw_change = clim_change(gw_seas, period1=self.period1, period2=self.period2,
                                                    region_method=self.region_method, box=self.box,
                                                    region_id=self.region_id, season=self.season,
                                                    preserve_time_series=False)
                        else:
                            gw_change = clim_change(gw_seas, period1=self.period1, period2=self.period2,
                                                    season=self.season, preserve_time_series=False)
                        if not has_spatial:
                            gw_change = gw_change.expand_dims({'lat': change['lat'], 'lon': change['lon']})
                        # append normalized changes
                        self.ensemble_changes[var].append(change / gw_change)
                        self.time_series_changes[var].append(change_ts / gw_change)
                    except KeyError as e:
                        print(f"KeyError: {e}. Skipping alias {alias} in dataset {dataset}.")
                        continue

    def _combine_and_save(self):
        # Combine ensemble changes into Dataset and save
        combined = xr.Dataset({
            var: xr.concat(self.ensemble_changes[var], dim='model')
            for var in self.var_names
        })
        # assign sorted model coordinate
        models = sorted(self.all_model_names)
        for var in self.var_names:
            combined[var] = combined[var].assign_coords(model=('model', models))
        out_file = os.path.join(self.user_config['work_dir'], 'combined_changes_esmval.nc')
        combined.to_netcdf(out_file)
        print(f"Saved all model ensemble changes to {out_file}")
        return combined

    def _plot_spatial(self, combined):
        # plot spatial maps for each variable
        for var in self.var_names:
            if self.ensemble_changes[var]:
                arr = xr.concat(self.ensemble_changes[var], dim='model')
                mean = arr.mean(dim='model')
                if 'model' in arr.dims and arr.sizes['model'] > 1:
                    pval = xr.apply_ufunc(
                        test_mean_significance, arr,
                        input_core_dims=[['model']], output_core_dims=[[]],
                        vectorize=True, dask='parallelized'
                    )
                else:
                    print("Only one model available; cannot compute p-values.")
                    pval = None
                pos = xr.open_dataset(os.path.join(self.config['work_dir'], 'stippling',
                                                   'number_of_models_positive_trend_CMIP6.nc'))
                neg = xr.open_dataset(os.path.join(self.config['work_dir'], 'stippling',
                                                   'number_of_models_negative_trend_CMIP6.nc'))
                fig = plot_function(mean, pval, pos, neg, self.region_extents)
                if fig:
                    fig.savefig(os.path.join(self.config['plot_dir'], f"beta_forced_{var}_plot.png"))

    def _plot_timeseries(self):
        years = np.arange(1950, 2100)
        for var in self.var_names:
            fig = plot_precipitation_change(
                self.time_series_changes[var], region_extents=self.region_extents,
                years=years, var_name=var
            )
            if fig:
                fig.savefig(os.path.join(self.user_config['plot_dir'], f"time_series_plot_{var}.png"))

    def run(self):
        """
        Execute the full processing workflow.
        """
        self._process_data()
        combined = self._combine_and_save()
        self._plot_spatial(combined)
        self._plot_timeseries()