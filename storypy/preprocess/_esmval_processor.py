"""
storypy.preprocess._esmval_processor
====================================

Module for preprocessing CMIP6 data and associated drivers
using the ESMValTool configuration system. The `ESMValProcessor`
class provides a high-level interface to read, process, and
aggregate ensemble data directly from ESMValTool-compatible
metadata.

Typical workflow
----------------
1. Parse a user-defined ESMValTool configuration file with :func:`parse_config`.
2. Initialize an :class:`ESMValProcessor` instance with configuration and
   user settings.
3. Call :meth:`process_var` to process target variables.
4. Call :meth:`process_driver` to process and aggregate driver variables.

Example
-------
>>> from storypy.preprocess._esmval_processor import parse_config, ESMValProcessor
>>> cfg = parse_config("config.yml")
>>> user_cfg = {
...     "work_dir": "./output",
...     "plot_dir": "./output/plots",
...     "region_method": "box",
...     "box": {"lat_min": -10, "lat_max": 10, "lon_min": 0, "lon_max": 50},
...     "period1": [1950, 1980],
...     "period2": [1990, 2020],
...     "region_id": "Tropical_Africa",
...     "season": ["DJF"],
...     "region_extents": [[-10, 10, 0, 50]],
...     "var_name": ["pr"],
... }
>>> processor = ESMValProcessor(cfg, user_cfg)
>>> processor.process_var()      # process target variable(s)
>>> processor.process_driver()   # process driver variables
"""

import os
import warnings
from storypy.utils import np, xr

from ._diagnostics import clim_change, seasonal_data_months, test_mean_significance
# from storypy.evaluate.plot import plot_precipitation_change, plot_function
from esmvaltool.diag_scripts.shared import run_diagnostic, get_cfg, group_metadata
from esmvaltool.diag_scripts.shared._base import _get_input_data_files

def parse_config(file):
    """
    Parse an ESMValTool configuration file.

    Reads and expands an ESMValTool YAML configuration to include
    resolved paths to input data files.

    Parameters
    ----------
    file : str or path-like
        Path to the ESMValTool configuration YAML file.

    Returns
    -------
    dict
        Parsed configuration dictionary containing ``input_data``
        and other ESMValTool runtime settings.

    Examples
    --------
    >>> from storypy.preprocess._esmval_processor import parse_config
    >>> cfg = parse_config("config.yml")
    >>> list(cfg.keys())
    ['input_data', 'run_dir', 'output_dir', ...]
    """
    config = get_cfg(file)           
    config['input_data'] = _get_input_data_files(config)
    return config

class ESMValProcessor:
    """
    High-level processor for CMIP6 data using ESMValTool metadata.

    This class handles loading, preprocessing, computing climatological
    changes, and generating plots for climate variables and remote drivers.
    It is designed to operate directly on the configuration and metadata
    produced by ESMValTool diagnostics.

    Parameters
    ----------
    config : dict
        Parsed ESMValTool configuration dictionary (see :func:`parse_config`).
    user_config : dict
        User-defined configuration dictionary controlling variable names,
        time periods, spatial regions, output paths, and plotting options.
    driver_config : dict, optional
        Optional configuration dictionary for processing driver variables.
        Keys can override ``var_name``, ``period1``, ``period2``,
        ``box``, or ``work_dir``.

    Attributes
    ----------
    ensemble_changes : dict
        Stores computed climatological changes for each variable across models.
    time_series_changes : dict
        Stores time series of changes for each variable.
    driver_data : dict
        Processed driver variables grouped by model.
    all_model_names : set
        All dataset/alias names encountered during processing.

    Notes
    -----
    The workflow generally proceeds as follows:

    1. Initialize the processor with ESMValTool and user configurations.
    2. Call :meth:`process_var` to compute climatological changes for target variables.
    3. Call :meth:`process_driver` to compute and aggregate driver variables.
    4. The results are saved to NetCDF files and corresponding plots.

    Example
    -------
    >>> cfg = parse_config("config.yml")
    >>> user_cfg = {...}  # user-defined settings
    >>> proc = ESMValProcessor(cfg, user_cfg)
    >>> proc.process_var()
    >>> proc.process_driver()
    """
    def __init__(self, config, user_config, driver_config=None):
        """
        Initialize the ESMValProcessor instance.

        Sets up configuration attributes, creates output directories,
        and prepares internal containers for storing processed data.

        Parameters
        ----------
        config : dict
            ESMValTool configuration from :func:`parse_config`.
        user_config : dict
            User-defined processing options.
        driver_config : dict, optional
            Overrides for driver variable settings.
        """
        self.config = config
        self.user_config = user_config
        xr.set_options(keep_attrs=True)
        self._compute_bounding_box()
        # Unpack frequently used config values
        uc = self.user_config
        # self.data_dir = uc['data_dir']
        self.work_dir = uc['work_dir']
        self.plot_dir = uc['plot_dir']
        self.region_method = uc["region_method"]
        self.box = uc["box"]
        self.period1 = uc["period1"]
        self.period2 = uc["period2"]
        self.region_id = uc["region_id"]
        self.season = uc["season"]
        self.region_extents = uc["region_extents"]
        self.var_names = uc["var_name"]
        self.titles = uc.get("titles", [])
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

        # Prepare storage
        self.ensemble_changes = {var: [] for var in self.var_names}
        self.time_series_changes = {var: [] for var in self.var_names}
        self.driver_data = {var: [] for var in self.driver_vars}
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

    def _process_drivers(self):
        meta_dataset, _ = self._load_metadata()

        for dataset, alias_list in meta_dataset.items():
            meta_group = group_metadata(alias_list, "alias")

            for alias, alias_members in meta_group.items():
                self.all_model_names.add(alias)

                for var in self.driver_vars:
                    try:
                        # Load the driver variable
                        driver_data = {
                            m['variable_group']: xr.open_dataset(m['filename'])[m['short_name']]
                            for m in alias_members
                            if m['dataset'] == dataset and m['variable_group'] == var
                        }

                        if not driver_data:
                            print(f"No data found for driver '{var}' in alias '{alias}' of dataset '{dataset}'.")
                            continue

                        da = seasonal_data_months(driver_data[var], list(self.driver_season))

                        if var == 'pr':
                            da = da * 86400  # Convert from kg m-2 s-1 to mm/day

                        delta = clim_change(
                            da,
                            period1=self.driver_period1,
                            period2=self.driver_period2,
                            region_method=self.region_method,
                            box=self.driver_box,
                            region_id=self.region_id,
                            season=self.driver_season,
                            preserve_time_series=False
                        )

                        # Add model dimension if needed
                        if 'model' not in delta.dims:
                            delta = delta.expand_dims(model=[alias])

                        self.driver_data[var].append(delta)

                    except Exception as e:
                        print(f"Error processing driver '{var}' for alias '{alias}' in dataset '{dataset}': {e}")
    
    def _combine_and_save(self):
        # Combine ensemble changes grouped by base model name (e.g., UKESM1-0-LL)
        from collections import defaultdict

        grouped_means = {}

        for var in self.var_names:
            # Build a list of (model_id, DataArray)
            model_data = list(zip(self.all_model_names, self.ensemble_changes[var]))
            base_model_groups = defaultdict(list)

            for model_id, da in model_data:
                base_model = model_id.split('_')[0]  # e.g., UKESM1-0-LL
                base_model_groups[base_model].append(da)

            # Compute mean over ensemble members for each base model
            mean_per_model = []
            model_names = []

            for base_model, members in base_model_groups.items():
                if len(members) == 1:
                    mean_da = members[0]
                else:
                    mean_da = xr.concat(members, dim='ensemble').mean(dim='ensemble')

                mean_da = mean_da.expand_dims(model=[base_model])
                mean_per_model.append(mean_da)
                model_names.append(base_model)

            # Concatenate all base model means along the 'model' dimension
            grouped_means[var] = xr.concat(mean_per_model, dim='model', coords='minimal', compat='override')

        # Create final Dataset
        combined_ds = xr.Dataset(grouped_means)

        # Save to file
        out_file = os.path.join(self.user_config['work_dir'], f'target_{var}.nc')
        combined_ds.to_netcdf(out_file)
        print(f"Saved ensemble mean data to {out_file}")

        return combined_ds
    
    def _combine_drivers_and_save(self):
        """
        Combine processed driver variables across models and save as NetCDF.
        """
        if not self.driver_data:
            print("No driver data to combine.")
            return None

        combined_driver_ds = xr.Dataset({
            var: xr.concat(self.driver_data[var], dim='model')
            for var in self.driver_vars
        })

        # Assign sorted model names
        models = sorted(self.all_model_names)
        for var in self.driver_vars:
            combined_driver_ds[var] = combined_driver_ds[var].assign_coords(model=('model', models))

        out_file = os.path.join(self.user_config['work_dir'], f'driver_{var}.nc')
        combined_driver_ds.to_netcdf(out_file)
        print(f"Saved all driver variable changes to {out_file}")
        return combined_driver_ds

    def _plot_spatial(self, combined):
        from storypy.evaluate.plot import plot_function
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
        from storypy.evaluate.plot import plot_precipitation_change
        years = np.arange(1950, 2100)
        for var in self.var_names:
            fig = plot_precipitation_change(
                self.time_series_changes[var], region_extents=self.region_extents,
                years=years, var_name=var
            )
            if fig:
                fig.savefig(os.path.join(self.user_config['plot_dir'], f"time_series_plot_{var}.png"))

    def process_var(self):
        """
        Process target variables and compute climatological changes.

        This method loops through all datasets and variables defined in
        ``user_config``, computes ensemble-mean climatological changes
        between the two defined time periods, saves the results as
        NetCDF.

        Returns
        -------
        xarray.Dataset
            Combined dataset containing climatological changes for all variables.

        See Also
        --------
        process_driver : process driver (remote forcing) variables.
        """
        self._process_data()
        combined = self._combine_and_save()
        self._plot_spatial(combined)
        return

    def process_driver(self):
        """
        Process and combine driver (remote forcing) variables.

        This method processes all driver variables specified in
        ``driver_config`` or ``user_config``. It computes climatological
        changes, aggregates across models, saves the results as NetCDF,
        and returns the combined dataset.

        Returns
        -------
        xarray.Dataset
            Combined dataset containing climatological changes for all drivers.
        """
        self._process_drivers()
        combined_drivers = self._combine_drivers_and_save()
        return combined_drivers