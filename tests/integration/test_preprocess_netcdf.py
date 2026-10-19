"""ModelDataPreprocessor on the synthetic CMIP archive (real files on disk)."""

import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from storypy.compute import compute_drivers
from storypy.preprocess import ModelDataPreprocessor


@pytest.fixture
def processed(main_config, driver_config):
    proc = ModelDataPreprocessor(main_config, driver_config)
    proc.process_var()
    proc.process_driver()
    return proc


def test_target_file(processed, main_config, cmip_archive):
    with xr.open_dataset(f"{main_config['work_dir']}/target_pr.nc") as ds:
        target = ds.load()
    assert sorted(target.model.values) == sorted(cmip_archive.models)
    assert "NO-TAS-MODEL" not in target.model.values
    # n_members counts one entry per model here (informational only, not used
    # anywhere); member_ids lists the actual members.
    assert target.member_ids.sel(model="CanESM5").item() == "CanESM5_r1i1p1f1|CanESM5_r2i1p1f1"
    # The box covers the whole grid; values are the planted pattern [mm/day/K].
    for m in ("ACCESS-CM2", "CESM2", "CESM2-WACCM", "MIROC6"):
        np.testing.assert_allclose(target.pr.sel(model=m).values,
                                   cmip_archive.expected_pattern(m), atol=1e-6)


def test_driver_files_and_compute_drivers(processed, driver_config, cmip_archive):
    for name in ("gw", "dA", "dB"):
        assert os.path.exists(f"{driver_config['work_dir']}/remote_driver_{name}.nc")
    raw, scaled, std = compute_drivers({"work_dir": driver_config["work_dir"],
                                        "var_name": ["psl", "uas"],
                                        "short_name": ["dA", "dB"]})
    a = cmip_archive.a.loc[scaled.index]
    np.testing.assert_allclose(scaled["dA"], a, rtol=1e-6)
    np.testing.assert_allclose(scaled["dB"], cmip_archive.b.loc[scaled.index], rtol=1e-6)
    err = pd.read_csv(f"{driver_config['work_dir']}/remote_drivers/drivers_errors.csv",
                      index_col=0)
    assert (err.loc["CanESM5"] > 0).all() and (err.loc["ACCESS-CM2"] == 0).all()


def test_time_series_plot_written(main_config, cmip_archive):
    cfg = {**main_config, "region_extents": [(-40, 0, -100, 0), (0, 40, 0, 100)],
           "region_labels": ["South", "North"]}
    ModelDataPreprocessor(cfg).process_var()
    assert os.path.exists(os.path.join(cfg["plot_dir"], "time_series_plot_pr.png"))
