"""Unit tests for storypy.compute._compute_driver."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from storypy.compute import compute_drivers, driver_indices, stand_numpy, stand_pandas
from storypy.compute._compute_driver import (
    EXCLUDE_VARS,
    _build_drivers_dataset,
    _collect_scalar_drivers,
    _select_one_file_per_var,
)


def test_standardisation():
    s = pd.Series([1.0, 2.0, 3.0])
    np.testing.assert_allclose(stand_pandas(s), stand_numpy(s.values))
    assert stand_pandas(pd.Series([2.0, 2.0])).isna().all()


def test_select_one_file_prefers_ensemble_mean():
    meta = [{"dataset": "A", "variable_group": "x", "alias": "A_r1"},
            {"dataset": "A", "variable_group": "x", "alias": "EnsembleMean"}]
    assert _select_one_file_per_var(meta, "A", "x")["alias"] == "EnsembleMean"
    assert _select_one_file_per_var(meta[:1], "A", "x")["alias"] == "A_r1"
    assert _select_one_file_per_var(meta, "B", "x") is None


def test_build_drivers_dataset_fills_missing_with_nan():
    ds = _build_drivers_dataset([{"gw": 1.0, "a": 2.0}, {"gw": 3.0}], ["m1", "m2"])
    assert np.isnan(ds["a"].sel(model="m2"))


def _write_combined(tmp_path, **variables):
    models = ["m1", "m2", "m3", "m4"]
    ds = xr.Dataset({k: ("model", np.asarray(v, float)) for k, v in variables.items()},
                    coords={"model": models})
    ds.to_netcdf(tmp_path / "drivers.nc")


def test_compute_drivers_from_combined_file(tmp_path):
    _write_combined(tmp_path, gw=[2, 4, 2, 4], a=[2, 4, 6, 8], b=[1, 1, 3, 2])
    raw, scaled, std = compute_drivers({"work_dir": str(tmp_path), "var_name": ["a", "b"],
                                        "short_name": ["A", "B"],
                                        "gradient_drivers": [("A_minus_B", "A", "B")]})
    assert list(raw.columns) == ["A", "B", "A_minus_B"]
    np.testing.assert_allclose(scaled["A"], [1, 1, 3, 2])
    np.testing.assert_allclose(std.mean(), 0, atol=1e-12)
    for f in ["drivers.csv", "scaled_drivers.csv", "scaled_standardized_drivers.csv",
              "drivers_errors.csv", "scaled_drivers_errors.csv"]:
        assert (tmp_path / "remote_drivers" / f).exists()


def test_compute_drivers_errors(tmp_path):
    with pytest.raises(ValueError, match="must match"):
        compute_drivers({"work_dir": str(tmp_path), "var_name": ["a"], "short_name": []})
    with pytest.raises(FileNotFoundError):
        compute_drivers({"work_dir": str(tmp_path), "var_name": ["a"]})
    _write_combined(tmp_path, gw=[1, 1, 1, 1], a=[1, 2, 3, 4])
    with pytest.raises(ValueError, match="missing required driver"):
        compute_drivers({"work_dir": str(tmp_path), "var_name": ["zzz"]})


def test_compute_drivers_from_per_driver_files(tmp_path):
    models = ["m1", "m2", "m3"]
    xr.Dataset({"gw": ("model", [2.0, 2.0, 4.0])}, coords={"model": models}) \
        .to_netcdf(tmp_path / "remote_driver_gw.nc")
    field = np.ones((3, 2, 2)) * np.array([1.0, 2.0, 3.0])[:, None, None]
    xr.Dataset({"A": (("model", "lat", "lon"), field),
                "A_std": (("model", "lat", "lon"), field * 0.1)},
               coords={"model": models, "lat": [0, 10], "lon": [0, 10]}) \
        .to_netcdf(tmp_path / "remote_driver_A.nc")
    raw, scaled, _ = compute_drivers({"work_dir": str(tmp_path), "var_name": ["psl"],
                                      "short_name": ["A"]})
    np.testing.assert_allclose(raw["A"], [1, 2, 3])
    np.testing.assert_allclose(scaled["A"], [0.5, 1, 0.75])
    err = pd.read_csv(tmp_path / "remote_drivers" / "drivers_errors.csv", index_col=0)
    np.testing.assert_allclose(err["A"], [0.1, 0.2, 0.3])


@pytest.mark.parametrize("selection, m1_value", [("mean", 1.05), ("first", 1.0), ("last", 1.1)])
def test_driver_indices_variant_selection(tmp_path, esmval_metadata, selection, m1_value):
    input_data, _ = esmval_metadata
    raw, scaled, std = driver_indices({"input_data": input_data, "work_dir": str(tmp_path),
                                       "variant_selection": selection})
    assert set(raw.columns) == {"sst_east", "jet"}   # 'pr' (excluded) and 'gw' are not drivers
    assert raw.loc["M1", "sst_east"] == pytest.approx(m1_value)


def test_driver_indices_scaling_and_spread(tmp_path, esmval_metadata):
    input_data, expected = esmval_metadata
    raw, scaled, _ = driver_indices({"input_data": input_data, "work_dir": str(tmp_path)})
    for m, exp in expected.items():
        assert scaled.loc[m, "sst_east"] == pytest.approx(exp["sst_east"] / exp["gw"])
    err = pd.read_csv(tmp_path / "remote_drivers" / "drivers_errors.csv", index_col=0)
    assert err.loc["M1", "sst_east"] == pytest.approx(0.05)
    assert err.loc["M5", "sst_east"] == 0


def test_driver_indices_configurable_exclusions(tmp_path, esmval_metadata):
    """EXCLUDE_VARS was hard-coded; config can now replace or extend it."""
    input_data, _ = esmval_metadata
    cfg = {"input_data": input_data, "work_dir": str(tmp_path)}
    raw, *_ = driver_indices({**cfg, "extra_exclude_vars": ["jet"]})
    assert list(raw.columns) == ["sst_east"]
    raw, *_ = driver_indices({**cfg, "exclude_vars": []})
    assert set(raw.columns) == {"sst_east", "jet", "pr"}
    raw, *_ = driver_indices({**cfg, "exclude_vars": ["pr", "jet"]})
    assert list(raw.columns) == ["sst_east"]
    assert "pr" in EXCLUDE_VARS            # default unchanged


def test_collect_scalar_drivers_raw_monthly(tmp_path):
    """season_months path: seasonal mean, anomaly vs ref period, future mean."""
    time = pd.date_range("1960-01", "2099-12", freq="MS")
    # Step from Dec 2069, so the whole DJF-2070 season is already "future".
    values = np.where(time >= "2069-12-01", 3.0, 1.0)
    meta = {"A": []}
    for group in ("gw", "x"):
        f = tmp_path / f"{group}.nc"
        xr.Dataset({group: ("time", values * (2 if group == "x" else 1))},
                   coords={"time": time}).to_netcdf(f)
        meta["A"].append({"filename": str(f), "dataset": "A", "alias": "A_r1",
                          "variable_group": group, "short_name": group})
    rd, models = _collect_scalar_drivers(meta, str(tmp_path), season_months=[12, 1, 2])
    assert models == ["A"]
    assert rd[0]["gw"] == pytest.approx(2.0) and rd[0]["x"] == pytest.approx(4.0)
    with pytest.raises(ValueError):
        _collect_scalar_drivers(meta, str(tmp_path), variant_selection="median")
