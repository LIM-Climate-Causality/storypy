"""Unit tests for storypy.compute.compute_regression."""

import numpy as np
import pytest
import xarray as xr

from storypy.compute import compute_regression


@pytest.fixture
def work_dir(tmp_path, planted_regression):
    target, drivers, _ = planted_regression
    target.to_dataset().to_netcdf(tmp_path / "target_pr.nc")
    (tmp_path / "remote_drivers").mkdir()
    drivers.iloc[::-1].to_csv(tmp_path / "remote_drivers" / "scaled_standardized_drivers.csv")
    return tmp_path


def test_returns_written_paths(work_dir, planted_regression):
    """Used to return [None]."""
    _, _, coefs = planted_regression
    files = compute_regression({"work_dir": str(work_dir), "target_variable": ["pr"]})
    assert len(files) == 4 and all(f.endswith(".nc") for f in files)
    ds = xr.open_dataset(work_dir / "regression_output" / "pr" / "regression_coefficients.nc")
    # Rows of the CSV were reversed on purpose: alignment is by model name.
    np.testing.assert_allclose(ds["ta"].values, coefs["ta"], atol=0.02)
    assert set(ds.data_vars) == {"MEM", "ta", "pw"}


def test_single_driver(work_dir):
    """Crashed with one driver (apply_ufunc got a 1-tuple)."""
    compute_regression({"work_dir": str(work_dir), "target_variable": ["pr"], "drivers": "ta"})
    ri = xr.open_dataset(work_dir / "regression_output" / "pr" /
                         "regression_coefficients_relative_importance.nc")
    assert list(ri.data_vars) == ["ta"] and float(ri["ta"].mean()) > 0


@pytest.mark.parametrize("cfg, err", [
    ({"target_variable": []}, ValueError),
    ({"target_variable": ["pr", "ua"]}, ValueError),
    ({"target_variable": ["ua"]}, FileNotFoundError),
    ({"target_variable": ["pr"], "drivers": ["nope"]}, ValueError),
])
def test_errors(work_dir, cfg, err):
    with pytest.raises(err):
        compute_regression({"work_dir": str(work_dir), **cfg})


def test_no_common_models(work_dir):
    with xr.open_dataset(work_dir / "target_pr.nc") as f:
        ds = f.load()
    ds.assign_coords(model=[f"X{i}" for i in range(ds.sizes["model"])]) \
        .to_netcdf(work_dir / "target_pr.nc")
    with pytest.raises(ValueError, match="No common models"):
        compute_regression({"work_dir": str(work_dir), "target_variable": ["pr"]})
