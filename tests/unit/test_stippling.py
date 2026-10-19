"""Unit tests for storypy.compute.StipplingComputer (beta criterion)."""

import numpy as np
import pytest
import xarray as xr

from storypy.compute import StipplingComputer


@pytest.fixture
def work_dir(tmp_path):
    data = np.ones((10, 2, 2))
    data[:, 0, 1] = -1          # all models negative
    data[:5, 1, 0] = -1         # split 50/50
    data[0, 1, 1] = -1          # 90% positive
    xr.Dataset({"ua": (("model", "lat", "lon"), data)},
               coords={"model": [f"m{i}" for i in range(10)], "lat": [0, 1], "lon": [0, 1]}) \
        .to_netcdf(tmp_path / "target_ua.nc")
    return tmp_path


def test_beta_significance(work_dir):
    pos, neg = StipplingComputer(str(work_dir), "ua").beta_significance(0.9)
    np.testing.assert_array_equal(pos.values, [[1, 0], [0, 1]])
    np.testing.assert_array_equal(neg.values, [[0, 1], [0, 0]])
    saved = xr.open_dataset(work_dir / "stippling" / "number_of_models_positive_trend_CMIP6.nc")
    assert "pr" in saved          # name expected by plot_function


def test_missing_target_and_config(tmp_path):
    sc = StipplingComputer(str(tmp_path), "pr")
    with pytest.raises(FileNotFoundError):
        sc.beta_significance()
    with pytest.raises(ValueError, match="picontrol_config"):
        sc.gamma_significance()
