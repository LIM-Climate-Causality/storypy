"""Gamma (signal-to-noise) stippling with a synthetic piControl run."""

import numpy as np
import pytest
import xarray as xr

from storypy.compute import StipplingComputer


@pytest.fixture
def setup(tmp_path):
    rng = np.random.default_rng(1)
    lat, lon = [-10.0, 10.0], [0.0, 90.0, 180.0, 270.0]
    models = ["A", "B", "C"]
    # Forced signal: large at lon=0/90, zero at lon=180/270.
    signal = np.zeros((3, 2, 4))
    signal[:, :, :2] = 1.0
    xr.Dataset({"pr": (("model", "lat", "lon"), signal)},
               coords={"model": models, "lat": lat, "lon": [0.0, 90.0, -180.0, -90.0]}) \
        .to_netcdf(tmp_path / "target_pr.nc")

    input_data = {}
    for m in models:
        for variant in ("r1", "r2"):
            # 200 years of seasonal means (already seasonally averaged)
            noise = 0.05 * rng.standard_normal((200, 2, 4))
            f = tmp_path / f"pi_{m}_{variant}.nc"
            xr.Dataset({"pr": (("time", "lat", "lon"), noise / 86400)},
                       coords={"time": np.arange(200), "lat": lat, "lon": lon}).to_netcdf(f)
            input_data[str(f)] = {"filename": str(f), "dataset": m, "alias": f"{m}_{variant}",
                                  "variable_group": "pr_picontrol", "short_name": "pr"}
    return tmp_path, {"input_data": input_data}


@pytest.mark.parametrize("selection", ["mean", "first", "last"])
def test_gamma_separates_signal_from_noise(setup, selection):
    work_dir, cfg = setup
    sc = StipplingComputer(str(work_dir), "pr", picontrol_config=cfg,
                           variant_selection=selection)
    gamma, denom = sc.gamma_significance(rolling_window=30)
    assert gamma.sel(lon=0).min() > 1           # strong forced signal
    assert float(abs(gamma.sel(lon=-180)).max()) == 0.0
    assert (denom > 0).all()
    assert (work_dir / "stippling" / "gamma_forced_CMIP6.nc").exists()


def test_gamma_errors(setup):
    work_dir, cfg = setup
    bad = {"input_data": {k: {**v, "variable_group": "other"} for k, v in cfg["input_data"].items()}}
    with pytest.raises(RuntimeError, match="No piControl data"):
        StipplingComputer(str(work_dir), "pr", picontrol_config=bad).gamma_significance()
    with pytest.raises(RuntimeError, match="No gamma"):
        StipplingComputer(str(work_dir), "pr", picontrol_config=cfg).gamma_significance(
            rolling_window=150)
