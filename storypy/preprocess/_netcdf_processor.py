import os
import warnings
import fnmatch
from storypy.utils import np, xr

from ._diagnostics import clim_change, seasonal_data_months
from storypy.evaluate.plot import plot_precipitation_change, plot_function

class ModelDataPreprocessor:
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
            files = [f for f in os.listdir(path) if f.endswith('.nc')]
            # Extract full model+member name (e.g., CNRM-ESM2-1_r15i1p1f2)
            model_members = {f.split('_')[2] + '_' + f.split('_')[3] for f in files}
            model_sets[var] = model_members

        if model_sets:
            common_model_members = set.intersection(*model_sets.values())
            print("Common model+members across all variables:", common_model_members)
            # Extract only model names (without member) for output
            common_models = sorted({mm.split('_')[0] for mm in common_model_members})
            return common_models

        return []
    
    def _process_var(self, var, common_models):
        path = os.path.join(self.data_dir, var, self.freq, self.grid)
        files = os.listdir(path)

        for m in common_models:
            ens_hist, ens_scen, ens_hist_gw, ens_scen_gw = [], [], [], []

            for f in files:
                if m not in f:
                    continue

                tas_name = 'tas_' + '_'.join(f.split('_')[1:])
                var_file_path = os.path.join(path, f)
                tas_file_path = os.path.join(self.data_dir, 'tas', self.freq, self.grid, tas_name)

                if fnmatch.fnmatch(f, f"{var}_*_{m}_*historical*.nc"):
                    try:
                        ens_hist.append(xr.open_dataset(var_file_path))
                    except FileNotFoundError:
                        print(f"Missing historical file skipped: {var_file_path}")
                        continue

                    try:
                        ens_hist_gw.append(xr.open_dataset(tas_file_path))
                    except FileNotFoundError:
                        print(f"Missing tas file for historical skipped: {tas_file_path}")
                        continue

                elif fnmatch.fnmatch(f, f"{var}_*_{m}_*{self.exp_name}*.nc"):
                    try:
                        ens_scen.append(xr.open_dataset(var_file_path))
                    except FileNotFoundError:
                        print(f"Missing scenario file skipped: {var_file_path}")
                        continue

                    try:
                        ens_scen_gw.append(xr.open_dataset(tas_file_path))
                    except FileNotFoundError:
                        print(f"Missing tas file for scenario skipped: {tas_file_path}")
                        continue

            if not ens_hist or not ens_scen:
                continue  # Skip this model if key data is missing

            hist = xr.concat(ens_hist, dim='ensemble')
            scen = xr.concat(ens_scen, dim='ensemble')
            self.target = {var: xr.concat([hist.mean('ensemble'), scen.mean('ensemble')], dim='time')}

            if ens_hist_gw and ens_scen_gw:
                gh = xr.concat(ens_hist_gw, dim='ensemble')
                gs = xr.concat(ens_scen_gw, dim='ensemble')
                gw = xr.concat([gh.mean('ensemble'), gs.mean('ensemble')], dim='time').mean(('lat', 'lon'))
                self.target['gw'] = gw

            try:
                tv = seasonal_data_months(self.target[var], list(self.season))
                if var == 'pr':
                    tv *= 86400

                ch = clim_change(tv, period1=self.period1, period2=self.period2,
                                region_method=self.region_method, box=self.box,
                                region_id=self.region_id, season=self.season,
                                preserve_time_series=False)

                ts = clim_change(tv, period1=self.period1, period2=self.period2,
                                region_method=self.region_method, box=self.box,
                                region_id=self.region_id, season=self.season,
                                preserve_time_series=True)

                sg = seasonal_data_months(self.target['gw'], list(self.season))
                has_sp = set(('lat', 'lon')).issubset(sg.dims)

                if has_sp:
                    gwc = clim_change(sg, period1=self.period1, period2=self.period2,
                                    region_method=self.region_method, box=self.box,
                                    region_id=self.region_id, season=self.season,
                                    preserve_time_series=False)
                else:
                    gwc = clim_change(sg, period1=self.period1, period2=self.period2,
                                    season=self.season, preserve_time_series=False)

                if not has_sp:
                    gwc = gwc.expand_dims({'lat': ch['lat'], 'lon': ch['lon']})

                gv = gwc['tas']
                norm = (ch / gv).expand_dims({'model': [m]})
                norm_ts = (ts / gv).expand_dims({'model': [m]})

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
        combined = xr.Dataset()

        for var in self.var_names:
            if not self.ensemble_changes[var]:
                print(f"Warning: No data found for variable '{var}', skipping.")
                continue

            arrays = []
            model_names = []

            for ds in self.ensemble_changes[var]:
                da = ds[var].squeeze(drop=True)

                # Try to extract real model name
                try:
                    model_name = ds.attrs.get('model_id') or ds.coords['model'].values.item()
                except Exception:
                    model_name = f"model_{len(model_names)}"

                arrays.append(da)
                model_names.append(model_name)

            data = xr.concat(arrays, dim='model', coords='minimal', compat='override')
            data = data.assign_coords(model=('model', model_names))
            combined[var] = data

        out = os.path.join(self.work_dir, 'target.nc')
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

    # # main change processing
    def process_var(self):
        self.common_models = self._find_common_models()
        for var in self.var_names:
            path = os.path.join(self.data_dir, var, self.freq, self.grid)
            if not os.path.exists(path):
                continue
            self._process_var(var, self.common_models)
        combined = self._combine_and_save()
        # plot time series
        self._plot_timeseries()

    # driver processing
    def process_driver(self):
        common = self._find_common_models()
        for var in self.driver_vars:
            self._process_driver_var(var, common)
        self._combine_and_save_drivers()