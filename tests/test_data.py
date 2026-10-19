"""
tests/test_data.py
==================

Tests for storypy.data:
  - load_change_field
  - load_driver_field
  - list_targets
  - read_regression
  - read_drivers / read_scaled_drivers / read_scaled_standardized_drivers
"""

import pytest
import xarray as xr
import pandas as pd
from storypy.data import (
    load_change_field,
    load_driver_field,
    list_targets,
    read_regression,
    read_drivers,
    read_scaled_drivers,
    read_scaled_standardized_drivers,
)


class TestLoadChangeField:

    def test_returns_dataset(self):
        ds = load_change_field('zs17', 'pr', 'NDJFM')
        assert isinstance(ds, xr.Dataset)

    def test_has_lat_lon_dims(self):
        ds = load_change_field('zs17', 'pr', 'NDJFM')
        assert 'lat' in ds.dims
        assert 'lon' in ds.dims

    def test_lat_slice_applied(self):
        """Default lat_slice=(-88, 88) should exclude the poles."""
        ds = load_change_field('zs17', 'pr', 'NDJFM')
        assert float(ds['lat'].min()) >= -88.0
        assert float(ds['lat'].max()) <=  88.0

    def test_u850_variable(self):
        ds = load_change_field('zs17', 'u850', 'NDJFM')
        assert isinstance(ds, xr.Dataset)

    def test_mindlin_study(self):
        ds = load_change_field('mindlin20', 'pr', 'DJF')
        assert isinstance(ds, xr.Dataset)

    def test_invalid_study_raises(self):
        with pytest.raises(FileNotFoundError):
            load_change_field('nonexistent_study', 'pr', 'NDJFM')

    def test_invalid_season_raises(self):
        with pytest.raises(FileNotFoundError):
            load_change_field('zs17', 'pr', 'INVALID')


class TestLoadDriverField:

    def test_returns_dataset(self):
        ds = load_driver_field('zs17')
        assert isinstance(ds, xr.Dataset)

    def test_invalid_study_raises(self):
        with pytest.raises(FileNotFoundError):
            load_driver_field('nonexistent')


class TestListTargets:

    def test_returns_list(self):
        result = list_targets()
        assert isinstance(result, list)

    def test_all_nc_files(self):
        result = list_targets()
        assert all(f.endswith('.nc') for f in result)

    def test_known_files_present(self):
        result = list_targets()
        assert 'zs17_target_pr_NDJFM.nc' in result

    def test_sorted(self):
        result = list_targets()
        assert result == sorted(result)


class TestReadRegression:

    def test_default_diagnostic(self):
        ds = read_regression('pr')
        assert isinstance(ds, xr.Dataset)

    def test_all_diagnostics_loadable(self):
        for diag in [
            'regression_coefficients',
            'regression_coefficients_pvalues',
            'regression_coefficients_relative_importance',
            'R2',
        ]:
            ds = read_regression('pr', diagnostic=diag)
            assert isinstance(ds, xr.Dataset)

    def test_ua_variable(self):
        ds = read_regression('ua')
        assert isinstance(ds, xr.Dataset)

    def test_invalid_diagnostic_raises(self):
        with pytest.raises(ValueError, match='Unknown diagnostic'):
            read_regression('pr', diagnostic='invalid_diag')

    def test_has_spatial_dims(self):
        ds = read_regression('pr')
        assert 'lat' in ds.dims
        assert 'lon' in ds.dims


class TestReadDriverCSVs:

    def test_read_drivers_returns_dataframe(self):
        df = read_drivers()
        assert isinstance(df, pd.DataFrame)

    def test_read_scaled_drivers_returns_dataframe(self):
        df = read_scaled_drivers()
        assert isinstance(df, pd.DataFrame)

    def test_read_scaled_standardized_drivers_returns_dataframe(self):
        df = read_scaled_standardized_drivers()
        assert isinstance(df, pd.DataFrame)

    def test_drivers_have_model_index(self):
        df = read_drivers()
        assert df.index.name == 'model' or pd.api.types.is_string_dtype(df.index) or df.index.dtype == object

    def test_scaled_std_has_same_columns_as_raw(self):
        """Scaled standardized drivers should have same columns as raw drivers."""
        raw = read_drivers()
        std = read_scaled_standardized_drivers()
        assert list(raw.columns) == list(std.columns)
