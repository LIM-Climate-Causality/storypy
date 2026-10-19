"""Unit tests for storypy.preprocess._diagnostics."""

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from storypy.preprocess import adjust_longitudes, clim_change, seasonal_data_months
from storypy.preprocess._diagnostics import extract_metadata
from storypy.preprocess._diagnostics import test_mean_significance as mean_significance


def monthly(start="2000-01", end="2003-12", values=None):
    time = pd.date_range(start, end, freq="MS")
    vals = np.arange(len(time), dtype=float) if values is None else values
    return xr.DataArray(vals, dims="time", coords={"time": time})


def test_djf_assigns_december_to_next_year():
    da = monthly()
    djf = seasonal_data_months(da, [12, 1, 2])
    # Season 2001 = Dec 2000 (idx 11), Jan 2001 (12), Feb 2001 (13)
    assert float(djf.sel(time=2001)) == pytest.approx(12.0)
    assert djf.time.dtype.kind == "i"


def test_non_wrapping_season_with_december_stays_in_year():
    """Bug fix: OND used to put December into the following year."""
    ond = seasonal_data_months(monthly(), [10, 11, 12])
    assert float(ond.sel(time=2000)) == pytest.approx(10.0)  # months 9,10,11 idx
    assert list(ond.time.values) == [2000, 2001, 2002, 2003]


def test_cftime_calendar():
    time = xr.date_range("2000-01-01", periods=24, freq="MS", calendar="noleap", use_cftime=True)
    da = xr.DataArray(np.ones(24), dims="time", coords={"time": time})
    assert isinstance(time[0], cftime.datetime)
    assert seasonal_data_months(da, [6, 7, 8]).sizes["time"] == 2


@pytest.mark.parametrize("months", [[], [0], [13]])
def test_invalid_months(months):
    with pytest.raises(ValueError):
        seasonal_data_months(monthly(), months)


def test_clim_change_on_integer_years():
    da = xr.DataArray(np.r_[np.zeros(10), np.ones(10)], dims="time",
                      coords={"time": np.arange(1990, 2010)})
    assert float(clim_change(da, ("1990", "1999"), ("2000", "2009"), season=[1])) == 1.0


def test_clim_change_without_season_on_datetimes():
    """Was a crash: int('1990-01') when season was None."""
    da = monthly("1990-01", "2009-12", np.r_[np.zeros(120), np.full(120, 2.0)])
    assert float(clim_change(da, ("1990", "1999"), ("2000", "2009"))) == 2.0


def test_clim_change_box_and_time_series():
    da = xr.DataArray(np.ones((4, 3, 3)), dims=("time", "lat", "lon"),
                      coords={"time": np.arange(2000, 2004), "lat": [-10, 0, 10],
                              "lon": [0, 90, 270]})
    box = {"lat_min": -5, "lat_max": 15, "lon_min": -100, "lon_max": 95}
    out = clim_change(da, (2000, 2001), (2002, 2003), box=box)
    assert out.sizes == {"lat": 2, "lon": 3}
    assert out.attrs["region_extent"] == [-100, 95, -5, 15]
    ts = clim_change(da, (2000, 2001), (2002, 2003), box=box, preserve_time_series=True)
    assert ts.sizes["time"] == 4


def test_clim_change_mask():
    pytest.importorskip("regionmask")
    lat = np.arange(-89.0, 90, 2)
    lon = np.arange(-179.0, 180, 2)
    da = xr.DataArray(np.ones((2, lat.size, lon.size)), dims=("time", "lat", "lon"),
                      coords={"time": [2000, 2001], "lat": lat, "lon": lon})
    out = clim_change(da, (2000, 2000), (2001, 2001), region_method="mask", region_id=18)
    assert out.notnull().sum() > 0 and out.isnull().sum() > 0
    assert len(out.attrs["region_extent"]) == 4


def test_adjust_longitudes():
    da = xr.DataArray(np.arange(4.0), dims="lon", coords={"lon": [0, 90, 180, 270]})
    out = adjust_longitudes(da)
    assert list(out.lon.values) == [-180, -90, 0, 90]
    assert float(out.sel(lon=-90)) == 3.0


def test_mean_significance_ttest():
    assert np.isnan(mean_significance(np.array([1.0])))
    assert mean_significance(np.array([5.0, 5.1, 4.9, 5.2])) < 0.01


def test_extract_metadata(tmp_path):
    """Was a NameError (pd not imported). Reads the variable from the
    second-to-last ``_`` field of the file name."""
    f = tmp_path / "X_ssp585_r1i1p1f1_pr_2000.nc"
    xr.Dataset({"pr": ("time", [1.0])}, coords={"time": [0]},
               attrs={"source_id": "X", "variable_id": "pr"}).to_netcdf(f)
    df = extract_metadata([str(f)])
    assert df.loc[0, "model"] == "X" and df.loc[0, "variable"] == "pr"
