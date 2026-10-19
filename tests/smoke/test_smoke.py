"""
Smoke tests: does the installed package import and expose its API?

These run first in CI and in ``make test``; if they fail, nothing else is
worth running. They must stay fast and must not need ESMValTool.
"""

import importlib
import subprocess
import sys

import pytest

SUBPACKAGES = [
    "storypy",
    "storypy.utils",
    "storypy.data",
    "storypy.compute",
    "storypy.preprocess",
    "storypy.evaluate",
    "storypy.evaluate.plot",
]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name):
    importlib.import_module(name)


def test_version_is_set_and_matches_metadata():
    from importlib.metadata import version

    import storypy

    assert storypy.__version__ == version("storypy")


@pytest.mark.parametrize("module, names", [
    ("storypy.compute", ["compute_drivers", "driver_indices", "compute_regression",
                         "SpatialRegression", "StipplingComputer"]),
    ("storypy.preprocess", ["ModelDataPreprocessor", "ESMValProcessor", "parse_config",
                            "seasonal_data_months", "clim_change"]),
    ("storypy.evaluate.plot", ["bivariate_dist", "storyline_evaluation",
                               "plot_storyline_map", "plot_anomaly_series"]),
    ("storypy.data", ["read_regression", "load_change_field", "list_targets"]),
])
def test_public_api_present(module, names):
    mod = importlib.import_module(module)
    missing = [n for n in names if not hasattr(mod, n)]
    assert not missing, f"{module} is missing {missing}"


def test_imports_without_esmvaltool():
    """Block ``esmvaltool`` and check that the package still imports."""
    code = (
        "import sys; sys.modules['esmvaltool'] = None\n"
        "import storypy.compute, storypy.preprocess\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_bundled_sample_data_is_installed():
    from storypy.data import list_drivers, list_targets

    assert "zs17_target_pr_NDJFM.nc" in list_targets()
    assert "zs17_drivers.nc" in list_drivers()


def test_regression_runs_on_tiny_input():
    import numpy as np
    import pandas as pd
    import xarray as xr

    from storypy.compute import SpatialRegression

    drivers = pd.DataFrame({"d": [0.0, 1.0, 2.0, 3.0, 4.0]})
    target = xr.DataArray(2 + 3 * drivers["d"].values[:, None]
                          + np.array([[0.1], [-0.1], [0.0], [0.1], [-0.1]]),
                          dims=("model", "x"))
    sr = SpatialRegression()
    sr.regression_data(target, drivers, ["MEM", "d"])
    coefs = sr.linear_regression(target.values[:, 0])
    assert coefs[1] == pytest.approx(3.0, abs=0.1)
