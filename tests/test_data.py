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
        assert df.index.name == 'model' or df.index.dtype == object

    def test_scaled_std_has_same_columns_as_raw(self):
        """Scaled standardized drivers should have same columns as raw drivers."""
        raw = read_drivers()
        std = read_scaled_standardized_drivers()
        assert list(raw.columns) == list(std.columns)

import os
import pandas as pd
import pandas.testing as pdt

def compare_driver_csvs(dir1, dir2, filename="drivers.csv", atol=0.01, rtol=0.01):
    """
    Compare drivers.csv files from two directories using only common models and drivers.
    
    Parameters:
        dir1 (str): Path to logic_1 output directory
        dir2 (str): Path to logic_2 output directory
        filename (str): CSV file name to compare (default: "drivers.csv")
        atol (float): Absolute tolerance
        rtol (float): Relative tolerance
    
    Returns:
        bool: True if values match within tolerance, False otherwise
    """
    file1 = os.path.join(dir1, filename)
    file2 = os.path.join(dir2, filename)

    df1 = pd.read_csv(file1, index_col=0)
    df2 = pd.read_csv(file2, index_col=0)

    # Find common models (index) and drivers (columns)
    common_models = df1.index.intersection(df2.index)
    common_columns = df1.columns.intersection(df2.columns)

    df1_common = df1.loc[common_models, common_columns].sort_index().sort_index(axis=1)
    df2_common = df2.loc[common_models, common_columns].sort_index().sort_index(axis=1)

    try:
        pd.testing.assert_frame_equal(df1_common, df2_common, atol=atol, rtol=rtol, check_dtype=False)
        print("✅ Common models and drivers match within tolerance.")
        return True
    except AssertionError as e:
        print("❌ Differences found in common model values:")
        print(e)
        return False

esmvaltool_dir = "/climca/people/ralawode/esmvaltool_output/test_recipe_20250514_132506/work/storyline_analysis/remote_drivers/remote_drivers"
directnetcdf_dir = "/climca/people/storylinetool/test_user/driver_outputs/remote_drivers"

compare_driver_csvs(esmvaltool_dir, directnetcdf_dir, filename="drivers.csv")