"""
Shared fixtures for the storypy test suite.

Test layers (selected with ``-m``):

* ``smoke``       -- tests/smoke: imports and basic sanity, seconds.
* ``unit``        -- tests/unit: one function at a time, tiny in-memory data.
* ``integration`` -- tests/integration: several components, temporary files.
* ``e2e``         -- tests/e2e: raw CMIP-style files -> storylines.

The layer marker is added automatically from the folder name, so test
files do not need to repeat it.

The synthetic CMIP archive (:func:`cmip_archive`) is built so the right
answer is known: every model's precipitation response pattern is

    P_m(x) = alpha(x) + beta(x) * z(A)_m + gamma(x) * z(B)_m

where ``z(A)``, ``z(B)`` are the standardised GW-scaled driver responses.
The full pipeline must recover ``alpha``, ``beta`` and ``gamma``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # never open windows in tests

import numpy as np
import pandas as pd
import pytest
import xarray as xr

LAYERS = ("smoke", "unit", "integration", "e2e")


def pytest_collection_modifyitems(config, items):
    for item in items:
        parts = Path(str(item.fspath)).parts
        layer = next((name for name in LAYERS if name in parts), "unit")
        item.add_marker(getattr(pytest.mark, layer))


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


# ---------------------------------------------------------------------------
# Small in-memory data
# ---------------------------------------------------------------------------

LAT = np.array([-50.0, -30.0, -10.0, 10.0, 30.0, 50.0])
LON = np.array([-165.0, -120.0, -75.0, -30.0, 15.0, 60.0, 105.0, 150.0])


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def models():
    return [f"MODEL-{i:02d}" for i in range(15)]


@pytest.fixture
def planted_regression(rng, models):
    """
    Target ``(model, lat, lon)`` built from known coefficients plus small
    noise, and the drivers that generate it.
    """
    n = len(models)
    drivers = pd.DataFrame(
        {"ta": rng.standard_normal(n), "pw": rng.standard_normal(n)}, index=models)
    drivers = (drivers - drivers.mean()) / drivers.std(ddof=0)
    shape = (len(LAT), len(LON))
    coefs = {
        "MEM": rng.uniform(-1, 1, shape),
        "ta": rng.uniform(-0.5, 0.5, shape),
        "pw": rng.uniform(-0.5, 0.5, shape),
    }
    values = (coefs["MEM"][None]
              + coefs["ta"][None] * drivers["ta"].values[:, None, None]
              + coefs["pw"][None] * drivers["pw"].values[:, None, None]
              + 0.01 * rng.standard_normal((n, *shape)))
    target = xr.DataArray(values, dims=("model", "lat", "lon"),
                          coords={"model": models, "lat": LAT, "lon": LON}, name="pr")
    return target, drivers, coefs


# ---------------------------------------------------------------------------
# Synthetic CMIP archive on disk
# ---------------------------------------------------------------------------

@dataclass
class CmipArchive:
    data_dir: Path
    models: list
    members: dict                       # model -> list of variants
    gw: pd.Series                       # global warming per model [K]
    a: pd.Series                        # GW-scaled driver A per model
    b: pd.Series                        # GW-scaled driver B per model
    alpha: np.ndarray                   # planted coefficients (lat, lon)
    beta: np.ndarray
    gamma: np.ndarray
    lat: np.ndarray = field(default_factory=lambda: LAT)
    lon: np.ndarray = field(default_factory=lambda: LON)

    @staticmethod
    def zscore(s):
        return (s - s.mean()) / s.std(ddof=0)

    def expected_pattern(self, model):
        za, zb = self.zscore(self.a)[model], self.zscore(self.b)[model]
        return self.alpha + self.beta * za + self.gamma * zb


HIST = ("historical", "1950-01", "2014-12")
SCEN = ("ssp585", "2015-01", "2100-12")


def _write_run(path, var, values_hist, values_scen, lat, lon):
    """Write one historical and one scenario file for ``var``."""
    for (exp, start, end), field_ in ((HIST, values_hist), (SCEN, values_scen)):
        time = pd.date_range(start, end, freq="MS")
        month = time.month.values
        # Seasonal cycle, identical in both periods so it cancels in changes.
        amplitude = 0.01 * np.abs(values_hist).mean()
        seasonal = amplitude * np.cos(2 * np.pi * month / 12)[:, None, None]
        data = field_[None] + seasonal * np.ones((1, len(lat), len(lon)))
        ds = xr.Dataset({var: (("time", "lat", "lon"), data.astype("float64"))},
                        coords={"time": time, "lat": lat, "lon": lon})
        fname = path(exp, f"{start[:4]}01-{end[:4]}12")
        ds.to_netcdf(fname)


@pytest.fixture(scope="session")
def cmip_archive(tmp_path_factory) -> CmipArchive:
    """
    A small CMIP-like archive under ``<data_dir>/<var>/mon/g025``.

    * 12 models, three of which have two ensemble members; two names
      (``CESM2`` and ``CESM2-WACCM``) overlap on purpose.
    * ``tas`` jumps by ``gw_m`` from historical to scenario, everywhere.
    * ``psl`` / ``uas`` jump by ``gw_m * a_m`` / ``gw_m * b_m``.
    * ``pr`` jumps by ``gw_m * P_m(x) / 86400`` (stored in kg m-2 s-1).
    * One extra model has ``pr`` but no ``tas`` and must be dropped.
    """
    rng = np.random.default_rng(0)
    root = tmp_path_factory.mktemp("cmip")
    models = ["ACCESS-CM2", "CanESM5", "CESM2", "CESM2-WACCM", "EC-Earth3",
              "GFDL-ESM4", "INM-CM5-0", "IPSL-CM6A-LR", "MIROC6",
              "MPI-ESM1-2-LR", "MRI-ESM2-0", "UKESM1-0-LL"]
    multi = {"CanESM5", "IPSL-CM6A-LR", "MIROC6"}
    members = {m: ["r1i1p1f1", "r2i1p1f1"] if m in multi else ["r1i1p1f1"]
               for m in models}

    gw = pd.Series(rng.uniform(2.5, 5.5, len(models)), index=models)
    a = pd.Series(rng.normal(1.0, 0.3, len(models)), index=models)
    b = pd.Series(rng.normal(-0.5, 0.4, len(models)), index=models)
    shape = (len(LAT), len(LON))
    archive = CmipArchive(
        data_dir=root, models=models, members=members, gw=gw, a=a, b=b,
        alpha=rng.uniform(-0.3, 0.3, shape),
        beta=rng.uniform(-0.2, 0.2, shape),
        gamma=rng.uniform(-0.2, 0.2, shape),
    )

    for var in ("pr", "tas", "psl", "uas"):
        (root / var / "mon" / "g025").mkdir(parents=True)

    for m in models:
        for k, variant in enumerate(members[m]):
            # Zero-mean member jitter on the drivers: the ensemble mean keeps
            # the planted value, but the member spread is non-zero.
            jitter = 0.02 * (k - (len(members[m]) - 1) / 2)
            fields = {
                "tas": (288.0, gw[m] * (1 + jitter)),
                "psl": (101000.0, gw[m] * a[m] * (1 + jitter)),
                "uas": (5.0, gw[m] * b[m] * (1 + jitter)),
                "pr": (3e-5, gw[m] * archive.expected_pattern(m) / 86400.0),
            }
            for var, (base, delta) in fields.items():
                base_field = np.full(shape, base) + 1e-3 * rng.standard_normal(shape) * (
                    base if var != "pr" else 1e-6)

                def path(exp, period, var=var, m=m, variant=variant):
                    return root / var / "mon" / "g025" / (
                        f"{var}_Amon_{m}_{exp}_{variant}_gn_{period}.nc")

                _write_run(path, var, base_field, base_field + delta, LAT, LON)

    # A model with pr only (no tas): must not appear in the outputs.
    _write_run(lambda exp, period: root / "pr" / "mon" / "g025" / (
        f"pr_Amon_NO-TAS-MODEL_{exp}_r1i1p1f1_gn_{period}.nc"),
        "pr", np.full(shape, 3e-5), np.full(shape, 4e-5), LAT, LON)
    return archive


@pytest.fixture
def main_config(cmip_archive, tmp_path):
    # storypy writes into work_dir/plot_dir but expects them to exist
    # (ESMValTool always creates them).
    (tmp_path / "work").mkdir()
    (tmp_path / "plots").mkdir()
    return {
        "var_name": ["pr"],
        "data_dir": str(cmip_archive.data_dir),
        "work_dir": str(tmp_path / "work"),
        "plot_dir": str(tmp_path / "plots"),
        "exp_name": "ssp585",
        "freq": "mon",
        "grid": "g025",
        "region_method": "box",
        "region_id": None,
        "period1": ["1960", "1990"],
        "period2": ["2070", "2099"],
        "season": [12, 1, 2],
        # whole synthetic grid: the box is the padded union of the extents
        "region_extents": [(-60, 60, -175, 175)],
        "make_plots": False,
    }


@pytest.fixture
def driver_config(main_config):
    return {
        "var_name": ["tas", "psl", "uas"],
        "short_name": ["gw", "dA", "dB"],
        "box": {"lat_min": -40, "lat_max": 40, "lon_min": -100, "lon_max": 100},
        "work_dir": os.path.join(main_config["work_dir"], "driver_outputs"),
    }


# ---------------------------------------------------------------------------
# Fake ESMValTool metadata
# ---------------------------------------------------------------------------

@pytest.fixture
def esmval_metadata(tmp_path, rng):
    """
    ``input_data`` dict in ESMValTool's layout, pointing at small
    pre-averaged files (one value per model, already a future anomaly).
    Returns ``(input_data, expected)`` with the expected driver values.
    """
    input_data = {}
    expected = {}
    models = ["M1", "M2", "M3", "M4", "M5"]
    for i, model in enumerate(models):
        aliases = [f"{model}_r1i1p1f1", f"{model}_r2i1p1f1"] if i < 2 else [f"{model}_r1i1p1f1"]
        gw = 3.0 + i * 0.5
        values = {"gw": [], "sst_east": [], "jet": [], "pr": []}
        for j, alias in enumerate(aliases):
            sample = {"gw": gw + 0.1 * j, "sst_east": 1.0 + i * 0.2 + 0.1 * j,
                      "jet": -0.3 + i * 0.1, "pr": 0.5}
            for group, value in sample.items():
                fname = tmp_path / f"{group}_{alias}.nc"
                xr.Dataset({group: ("time", np.full(30, value))},
                           coords={"time": np.arange(2070, 2100)}).to_netcdf(fname)
                input_data[str(fname)] = {
                    "filename": str(fname), "dataset": model, "alias": alias,
                    "variable_group": group, "short_name": group,
                }
                values[group].append(value)
        expected[model] = {k: float(np.mean(v)) for k, v in values.items()}
    return input_data, expected
