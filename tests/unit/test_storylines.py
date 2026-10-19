"""Unit tests for storyline maths in storypy.evaluate.plot._storylines."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.stats import chi2

from storypy.evaluate.plot import bivariate_dist, make_symmetric_colorbar, storyline_evaluation


def test_bivariate_dist_independent_drivers():
    assert bivariate_dist(0.8, 2) == pytest.approx(np.sqrt(chi2.ppf(0.8, 2) / 2))
    assert bivariate_dist(0.8, 2) == pytest.approx(1.2686, abs=1e-4)


def test_bivariate_dist_correlated_drivers():
    i1, i2 = bivariate_dist(0.8, 2, r=0.35)
    assert i1 == pytest.approx(1.474, abs=1e-3)
    assert i2 == pytest.approx(1.023, abs=1e-3)
    assert i1 > bivariate_dist(0.8, 2) > i2


@pytest.mark.parametrize("kwargs", [{"confidence_level": 0}, {"confidence_level": 1.2},
                                    {"r": 1.0}, {"r": -1.5}])
def test_bivariate_dist_rejects_bad_input(kwargs):
    with pytest.raises(ValueError):
        bivariate_dist(**kwargs)


@pytest.fixture
def regression_dir(tmp_path):
    lat, lon = [0.0, 10.0], [0.0, 10.0]
    one = xr.DataArray(np.ones((2, 2)), dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    out = tmp_path / "regression_output" / "pr"
    out.mkdir(parents=True)
    xr.Dataset({"MEM": one * 1.0, "a": one * 0.5, "b": one * -0.25}) \
        .to_netcdf(out / "regression_coefficients.nc")
    return tmp_path


def test_storyline_evaluation_default_coefficient(regression_dir):
    sl, labels = storyline_evaluation({"work_dir": str(regression_dir)}, "pr", ["a", "b"],
                                      storyline_coefficient=1.0, gw_level=2)
    assert labels[-1] == "MEM"
    values = sl.isel(lat=0, lon=0).values
    # [high b low a, high b high a, low b low a, low b high a, MEM] * gw_level
    np.testing.assert_allclose(values, 2 * np.array([0.25, 1.25, 0.75, 1.75, 1.0]))


def test_storyline_evaluation_with_correlation(regression_dir):
    rd = regression_dir / "remote_drivers"
    rd.mkdir()
    pd.DataFrame({"a": [1, 2, 3, 4.0], "b": [1, 3, 2, 4.0]}) \
        .to_csv(rd / "scaled_standardized_drivers.csv")
    sl, _ = storyline_evaluation({"work_dir": str(regression_dir)}, "pr", ["a", "b"],
                                 use_correlation=True)
    assert sl.sizes["storyline"] == 5


def test_make_symmetric_colorbar():
    out = make_symmetric_colorbar(1.0)
    levels = np.asarray(out[0] if isinstance(out, tuple) else out)
    np.testing.assert_allclose(levels, -levels[::-1])
