import os
import fnmatch
from storypy.utils import np, xr

from ._diagnostics import clim_change, seasonal_data_months
from storypy.evaluate.plot import plot_precipitation_change, plot_function

class DirectProcessor:
    def __init__(self, user_config, driver_config=None):
        """
        Initialize processor for direct CMIP6 data processing without esmvaltool metadata.

        driver_config usage:
        - pass a dict with keys to customize remote‐driver computation separately from main processing.
        - Supported keys:
          * 'var_name': list of variables to compute drivers for (defaults to user_config['var_name']).
          * 'box': spatial bounding box dict with lat_min, lat_max, lon_min, lon_max (defaults to main box).
          * 'work_dir': path to directory where driver .nc outputs will be saved (defaults to main work_dir).
        - Any key not provided falls back to the corresponding setting in user_config.

        Parameters:
        - user_config: dict for main processing (variables, periods, region, directories).
        - driver_config: optional dict for remote‐driver computation; keys can override
          var_name, box, work_dir (for driver outputs), etc.
        """
        # main configuration
        self.uc = user_config
        xr.set_options(keep_attrs=True)
        self._compute_bounding_box()
        # unpack config
        uc = self.uc
        self.var_names = uc['var_name']
        self.data_dir = uc['data_dir']
        self.work_dir = uc['work_dir']
        self.plot_dir = uc['plot_dir']
        self.exp_name = uc['exp_name']
        self.freq = uc['freq']
        self.grid = uc['grid']
        self.region_method = uc['region_method']
        self.box = uc['box']
        self.period1 = uc['period1']
        self.period2 = uc['period2']
        self.region_id = uc['region_id']
        self.season = uc['season']
        self.region_extents = uc['region_extents']
        # driver configuration (fall back to user_config values)
        self.dc = driver_config or {}
        self.driver_vars = self.dc.get('var_name', self.var_names)
        self.driver_short_names = self.dc.get('short_name', self.driver_vars)

        if len(self.driver_vars) != len(self.driver_short_names):
            raise ValueError("Length of 'var_name' and 'short_name' must match in driver_config.")
        self.driver_period1 = self.dc.get('period1', self.period1)
        self.driver_period2 = self.dc.get('period2', self.period2)
        self.driver_box = self.dc.get('box', self.box)
        self.driver_work_dir = self.dc.get('work_dir', self.work_dir)
        self.driver_season = self.dc.get('season', self.season)
        self.driver_region_extents = self.dc.get('region_extents', self.region_extents)
        # storage
        self.ensemble_changes = {var: [] for var in self.var_names}
        self.time_series_changes = {var: [] for var in self.var_names}
        self.driver_data = {var: [] for var in self.driver_vars}

    def _compute_bounding_box(self):
        ext = self.uc['region_extents']
        lat_min = min(r[0] for r in ext)
        lat_max = max(r[1] for r in ext)
        lon_min = min(r[2] for r in ext)
        lon_max = max(r[3] for r in ext)
        self.uc['box'] = {
            'lat_min': max(lat_min - 5, -90),
            'lat_max': min(lat_max + 5, 90),
            'lon_min': lon_min - 5 if lon_min - 5 >= -180 else -180,
            'lon_max': lon_max + 5 if lon_max + 5 <= 180 else 180
        }

    def _find_common_models(self):
        model_sets = {}
        for var in self.var_names:
            path = os.path.join(self.data_dir, var, self.freq, self.grid)
            if not os.path.exists(path):
                print(f"Directory not found for variable {var}: {path}")
                continue
            files = os.listdir(path)
            models = {f.split('_')[2] for f in files if f.endswith('.nc')}
            model_sets[var] = models
        if model_sets:
            common = set.intersection(*model_sets.values())
            print("Common models across all variables:", common)
            return sorted(common)
        return []

    def _process_var(self, var, common_models):
        path = os.path.join(self.data_dir, var, self.freq, self.grid)
        files = os.listdir(path)
        for m in common_models:
            ens_hist, ens_scen, ens_hist_gw, ens_scen_gw = [], [], [], []
            for f in files:
                if m not in f: continue
                tas_name = 'tas_' + '_'.join(f.split('_')[1:])
                if fnmatch.fnmatch(f, f"{var}_*_{m}_*historical*.nc"):
                    ens_hist.append(xr.open_dataset(os.path.join(path, f)))
                    ens_hist_gw.append(xr.open_dataset(os.path.join(self.data_dir, 'tas', self.freq, self.grid, tas_name)))
                elif fnmatch.fnmatch(f, f"{var}_*_{m}_*{self.exp_name}*.nc"):
                    ens_scen.append(xr.open_dataset(os.path.join(path, f)))
                    ens_scen_gw.append(xr.open_dataset(os.path.join(self.data_dir, 'tas', self.freq, self.grid, tas_name)))
            if not ens_hist or not ens_scen: continue
            hist = xr.concat(ens_hist, dim='ensemble'); scen = xr.concat(ens_scen, dim='ensemble')
            self.target = {var: xr.concat([hist.mean('ensemble'), scen.mean('ensemble')], dim='time')}
            if ens_hist_gw and ens_scen_gw:
                gh = xr.concat(ens_hist_gw, dim='ensemble'); gs = xr.concat(ens_scen_gw, dim='ensemble')
                gw = xr.concat([gh.mean('ensemble'), gs.mean('ensemble')], dim='time').mean(('lat','lon'))
                self.target['gw'] = gw
            try:
                tv = seasonal_data_months(self.target[var], list(self.season))
                if var=='pr': tv*=86400
                ch = clim_change(tv, period1=self.period1, period2=self.period2,
                                 region_method=self.region_method, box=self.box,
                                 region_id=self.region_id, season=self.season,
                                 preserve_time_series=False)
                ts = clim_change(tv, period1=self.period1, period2=self.period2,
                                 region_method=self.region_method, box=self.box,
                                 region_id=self.region_id, season=self.season,
                                 preserve_time_series=True)
                sg = seasonal_data_months(self.target['gw'], list(self.season))
                has_sp = set(('lat','lon')).issubset(sg.dims)
                if has_sp:
                    gwc = clim_change(sg, period1=self.period1, period2=self.period2,
                                     region_method=self.region_method, box=self.box,
                                     region_id=self.region_id, season=self.season,
                                     preserve_time_series=False)
                else:
                    gwc = clim_change(sg, period1=self.period1, period2=self.period2,
                                     season=self.season, preserve_time_series=False)
                if not has_sp:
                    gwc=gwc.expand_dims({'lat':ch['lat'],'lon':ch['lon']})
                gv=gwc['tas']
                norm = (ch/gv).expand_dims({'model':[m]})
                norm_ts=(ts/gv).expand_dims({'model':[m]})
                self.ensemble_changes[var].append(norm)
                self.time_series_changes[var].append(norm_ts)
            except KeyError as e:
                print(f"KeyError: {e}. Skipping variable {var}.")
    
    def _process_driver_var(self, variable_name, common_models):
        path = os.path.join(self.data_dir, variable_name, self.freq, self.grid)
        files = os.listdir(path)

        driver_period1 = self.driver_period1
        driver_period2 = self.driver_period2
        driver_season = self.driver_season
        driver_box = self.driver_box
        driver_region_extents = self.driver_region_extents

        short_name_map = dict(zip(self.driver_vars, self.driver_short_names))
        short_name = short_name_map.get(variable_name, variable_name)

        for model in common_models:
            ens_hist = []
            ens_scen = []

            for f in files:
                if fnmatch.fnmatch(f, f"{variable_name}_*_{model}_*historical*.nc"):
                    ens_hist.append(xr.open_dataset(os.path.join(path, f)))
                elif fnmatch.fnmatch(f, f"{variable_name}_*_{model}_*{self.exp_name}*.nc"):
                    ens_scen.append(xr.open_dataset(os.path.join(path, f)))

            if not ens_hist or not ens_scen:
                continue

            hist = xr.concat(ens_hist, dim='ensemble')
            scen = xr.concat(ens_scen, dim='ensemble')

            combined = xr.concat([hist.mean('ensemble'), scen.mean('ensemble')], dim='time')

            try:
                seasonal = seasonal_data_months(combined, list(driver_season))
                if variable_name == 'pr':
                    seasonal *= 86400
                delta = clim_change(seasonal,
                                    period1=driver_period1,
                                    period2=driver_period2,
                                    region_method=self.region_method,
                                    box=driver_box,
                                    region_id=self.region_id,
                                    season=driver_season,
                                    preserve_time_series=False)
                da = delta[variable_name]
                if 'model' not in da.dims:
                    da = da.expand_dims({'model': [model]})
                da.name = short_name
                if short_name not in self.driver_data:
                    self.driver_data[short_name] = []
                self.driver_data[short_name].append(da)
            except Exception as e:
                print(f"Clim change failed for {variable_name}, model {model}: {e}")
                empty = seasonal.isel(time=0).mean(('lat', 'lon'))
                da = xr.full_like(empty, np.nan)
                if 'model' not in da.dims:
                    da = da.expand_dims({'model': [model]})
                da.name = short_name
                if short_name not in self.driver_data:
                    self.driver_data[short_name] = []
                self.driver_data[short_name].append(da)

    def _combine_and_save(self):
        combined = xr.Dataset({
            var: xr.concat([ds.to_dataarray() for ds in self.ensemble_changes[var]], dim='model', coords='minimal', compat='override')
            for var in self.var_names
        })
        for var in self.var_names:
            combined[var] = combined[var].assign_coords(model=np.unique(combined[var]['model']))
        out = os.path.join(self.work_dir, 'combined_changes.nc')
        combined.to_netcdf(out)
        print(f"Saved all model ensemble changes to {out}")
        return combined

    
    def _combine_and_save_drivers(self):
        # combine and write driver data
        os.makedirs(self.driver_work_dir, exist_ok=True)
        for var in self.driver_short_names:
            if self.driver_data[var]:
                da_all = xr.concat(self.driver_data[var], dim='model', coords='minimal', compat='override')
                ens_mean = da_all.mean(dim='model')
                ds_out = xr.Dataset({
                    f"{var}": da_all,
                    f"{var}_mean": ens_mean
                })
                outpath = os.path.join(self.driver_work_dir, f'remote_driver_{var}.nc')
                ds_out.to_netcdf(outpath)
                print(f"Saved remote driver output for {var} to {outpath}")
    
    def _plot_timeseries(self):
        yrs = np.arange(1950, 2100)
        for var in self.var_names:
            fig = plot_precipitation_change(
                self.time_series_changes[var],
                region_extents=self.region_extents,
                years=yrs,
                var_name=var
            )
            if fig:
                os.makedirs(self.plot_dir, exist_ok=True)
                fig.savefig(os.path.join(self.plot_dir, f"time_series_plot_{var}.png"))

    # def run(self):
    #     common = self._find_common_models()
    #     # main change processing
    #     for var in self.var_names:
    #         path = os.path.join(self.data_dir, var, self.freq, self.grid)
    #         if not os.path.exists(path):
    #             continue
    #         self._process_var(var, common)
    #     combined = self._combine_and_save()
    #     # driver processing
    #     for var in self.driver_vars:
    #         self._process_driver_var(var, common)
    #     self._combine_and_save_drivers()
    #     # plot time series
    #     self._plot_timeseries()
    # main change processing
    def process_var(self):
        common = self._find_common_models()
        for var in self.var_names:
            path = os.path.join(self.data_dir, var, self.freq, self.grid)
            if not os.path.exists(path):
                continue
            self._process_var(var, common)
        combined = self._combine_and_save()
        # plot time series
        self._plot_timeseries()
    # driver processing
    def process_driver(self):
        common = self._find_common_models()
        for var in self.driver_vars:
            self._process_driver_var(var, common)
        self._combine_and_save_drivers()