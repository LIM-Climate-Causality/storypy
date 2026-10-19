"""
End-to-end: raw CMIP-style NetCDF files -> storylines, as a user would run it
(README "Option B"). The synthetic archive has known answers, so the test
checks the science, not only that files appear.
"""

import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from storypy.compute import StipplingComputer, compute_drivers, compute_regression
from storypy.evaluate.plot import bivariate_dist, storyline_evaluation
from storypy.preprocess import ModelDataPreprocessor


@pytest.fixture
def pipeline(main_config, driver_config):
    cfg = main_config

    proc = ModelDataPreprocessor(cfg, driver_config)
    proc.process_var()
    proc.process_driver()

    compute_drivers({"work_dir": driver_config["work_dir"],
                     "var_name": ["psl", "uas"], "short_name": ["dA", "dB"]})
    files = compute_regression({"work_dir": cfg["work_dir"], "target_variable": ["pr"]})
    return cfg, files


def test_recovers_planted_response(pipeline, cmip_archive):
    cfg, files = pipeline
    reg = os.path.join(cfg["work_dir"], "regression_output", "pr")
    with xr.open_dataset(os.path.join(reg, "regression_coefficients.nc")) as coef:
        np.testing.assert_allclose(coef["MEM"].values, cmip_archive.alpha, atol=1e-4)
        np.testing.assert_allclose(coef["dA"].values, cmip_archive.beta, atol=1e-4)
        np.testing.assert_allclose(coef["dB"].values, cmip_archive.gamma, atol=1e-4)
    with xr.open_dataset(os.path.join(reg, "R2.nc")) as r2:
        assert float(r2["pr"].min()) > 0.999
    with xr.open_dataset(
            os.path.join(reg, "regression_coefficients_relative_importance.nc")) as ri:
        total = ri["dA"] + ri["dB"]
        np.testing.assert_allclose(total.values, r2_values(reg), atol=1e-8)


def r2_values(reg):
    with xr.open_dataset(os.path.join(reg, "R2.nc")) as r2:
        return r2["pr"].values


def test_storylines(pipeline, cmip_archive):
    cfg, _ = pipeline
    c = bivariate_dist(0.8, 2)
    sl, labels = storyline_evaluation(cfg, "pr", ["dA", "dB"], gw_level=3.0)
    expected = {
        "high dB high dA": cmip_archive.alpha + c * (cmip_archive.beta + cmip_archive.gamma),
        "low dB low dA": cmip_archive.alpha - c * (cmip_archive.beta + cmip_archive.gamma),
        "MEM": cmip_archive.alpha,
    }
    for label, field in expected.items():
        np.testing.assert_allclose(sl.sel(storyline=label).values, 3.0 * field, atol=1e-3)
    assert len(labels) == 5


def test_significance_and_artifacts(pipeline, cmip_archive):
    cfg, _ = pipeline
    pos, neg = StipplingComputer(cfg["work_dir"], "pr").beta_significance(0.9)
    with xr.open_dataset(os.path.join(cfg["work_dir"], "target_pr.nc")) as t:
        frac_pos = (t["pr"] > 0).mean("model")
    np.testing.assert_array_equal(pos.values, (frac_pos >= 0.9).values.astype(int))
    assert int((pos + neg).max()) <= 1

    assert os.path.exists(os.path.join(cfg["plot_dir"], "time_series_plot_pr.png"))
    drivers = pd.read_csv(os.path.join(cfg["work_dir"], "driver_outputs", "remote_drivers",
                                       "scaled_standardized_drivers.csv"), index_col=0)
    assert sorted(drivers.index) == sorted(cmip_archive.models)


@pytest.mark.network
def test_regression_maps_render(pipeline, tmp_path):
    """Coefficient maps need cartopy coastlines (downloaded on first use)."""
    from storypy.compute import SpatialRegression

    cfg, _ = pipeline
    sr = SpatialRegression()
    sr.regressor_names = ["MEM", "dA", "dB"]
    sr.rd_num = 3
    reg_dir = os.path.join(cfg["work_dir"], "regression_output")
    try:
        sr.plot_regression_coef_map(reg_dir, "pr", str(tmp_path))
        # sr.plot_regression_lmg_map(reg_dir, "pr", str(tmp_path), era5_file=None)
    except OSError as err:  # offline and no cached Natural Earth data
        pytest.skip(f"cartopy data unavailable: {err}")
    assert (tmp_path / "regression_coefficients_unknown_var.png").exists()
    # assert (tmp_path / "regression_coefficients_relative_importance_XXX.png").exists()
