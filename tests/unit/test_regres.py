"""Unit tests for storypy.compute._regres."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import xarray as xr

from storypy.compute import SpatialRegression
from storypy.compute._regres import filtro, lmg, stand, stand_detr

NAMES = ["MEM", "ta", "pw"]


def fitted(target, drivers, tmp_path, var="pr", names=NAMES):
    """Fit and write to tmp_path/var (perform_regression expects the folder)."""
    (tmp_path / var).mkdir(exist_ok=True)
    sr = SpatialRegression()
    sr.regression_data(target, drivers, names)
    return sr, sr.perform_regression(str(tmp_path), var)


def test_recovers_planted_coefficients(tmp_path, planted_regression):
    target, drivers, coefs = planted_regression
    _, files = fitted(target, drivers, tmp_path)
    ds = xr.open_dataset(files[0])
    for name, expected in coefs.items():
        np.testing.assert_allclose(ds[name].values, expected, atol=0.02)
    assert float(xr.open_dataset(files[3])["pr"].median()) > 0.95


def test_point_methods_match_statsmodels(planted_regression):
    target, drivers, _ = planted_regression
    sr = SpatialRegression()
    sr.regression_data(target, drivers, NAMES)
    x = target.isel(lat=2, lon=3).values
    res = sm.OLS(x, sm.add_constant(drivers.values)).fit()
    np.testing.assert_allclose(sr.linear_regression(x), res.params)
    np.testing.assert_allclose(sr.linear_regression_pvalues(x), res.pvalues)
    assert sr.linear_regression_R2(x) == pytest.approx(res.rsquared)


def test_nan_models_dropped_per_gridpoint(planted_regression):
    target, drivers, _ = planted_regression
    sr = SpatialRegression()
    sr.regression_data(target, drivers, NAMES)
    x = target.isel(lat=0, lon=0).values.copy()
    x[0] = np.nan
    res = sm.OLS(x[1:], sm.add_constant(drivers.values[1:])).fit()
    np.testing.assert_allclose(sr.linear_regression(x), res.params)
    x[:12] = np.nan
    assert np.isnan(sr.linear_regression(x)).all()
    assert np.isnan(sr.linear_regression_relative_importance(x)).all()


def test_relative_importance_sums_to_r2(planted_regression):
    """Was always 0: it called an undefined R object (`robjects`)."""
    target, drivers, _ = planted_regression
    sr = SpatialRegression()
    sr.regression_data(target, drivers, NAMES)
    x = target.isel(lat=1, lon=1).values
    ri = sr.linear_regression_relative_importance(x)
    assert len(ri) == 2 and sum(ri) == pytest.approx(sr.linear_regression_R2(x))
    assert max(ri) > 0


def test_lmg_two_driver_formula(rng):
    X = rng.standard_normal((30, 2))
    y = X @ np.array([1.0, 0.5]) + rng.standard_normal(30)

    def r2(cols):
        return sm.OLS(y, sm.add_constant(X[:, cols])).fit().rsquared

    out = lmg(y, X)
    # For two drivers LMG_1 = (R2(1) + R2(1,2) - R2(2)) / 2
    assert out[0] == pytest.approx((r2([0]) + r2([0, 1]) - r2([1])) / 2)
    assert out.sum() == pytest.approx(r2([0, 1]))


@pytest.mark.parametrize("var", ["pr", "ua", "tas", "sst", "some_new_var"])
def test_perform_regression_any_variable_name(tmp_path, planted_regression, var):
    """The if/elif on ``var`` failed whenever the target's name differed
    from what it expected (e.g. var='tas' with a target named 'pr')."""
    target, drivers, _ = planted_regression
    _, files = fitted(target, drivers, tmp_path, var=var)
    assert set(xr.open_dataset(files[0]).data_vars) == set(NAMES)
    assert set(xr.open_dataset(files[1]).data_vars) == set(NAMES)
    assert set(xr.open_dataset(files[2]).data_vars) == {"ta", "pw"}


def test_single_driver(tmp_path, planted_regression):
    target, drivers, _ = planted_regression
    _, files = fitted(target, drivers[["ta"]], tmp_path, names=["MEM", "ta"])
    assert list(xr.open_dataset(files[2]).data_vars) == ["ta"]


def test_open_saved_maps(tmp_path, planted_regression):
    target, drivers, _ = planted_regression
    sr, _ = fitted(target, drivers, tmp_path)
    maps, pvals, r2 = sr.open_regression_coef(str(tmp_path), "pr")
    assert len(maps) == len(pvals) == 3
    maps, pvals, r2 = sr.open_lmg_coef(str(tmp_path), "pr")
    assert len(maps) == 2


def test_helpers():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert stand(x).mean() == pytest.approx(0) and stand(x).std() == pytest.approx(1)
    assert np.allclose(stand_detr(x), 0)
    da = xr.DataArray(np.arange(30.0), dims="time", coords={"time": np.arange(30)})
    assert filtro(da).sizes["time"] < 30
