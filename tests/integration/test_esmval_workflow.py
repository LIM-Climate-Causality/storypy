"""
ESMValTool workflow without ESMValTool: ESMValProcessor + driver_indices fed
with ``input_data`` metadata pointing at files built from the synthetic archive.
"""

import numpy as np
import pytest
import xarray as xr

from storypy.compute import compute_regression, driver_indices
from storypy.preprocess import ESMValProcessor

MODELS = ["ACCESS-CM2", "CanESM5", "CESM2", "EC-Earth3", "GFDL-ESM4",
          "MIROC6", "MRI-ESM2-0", "UKESM1-0-LL"]


def _merged(archive, var, model, variant):
    d = archive.data_dir / var / "mon" / "g025"
    files = sorted(d.glob(f"{var}_Amon_{model}_*_{variant}_gn_*.nc"))
    parts = []
    for f in files:
        with xr.open_dataset(f) as ds:
            parts.append(ds.load())
    return xr.concat(parts, dim="time")


@pytest.fixture
def esmval_config(cmip_archive, tmp_path):
    """One preprocessed file per alias and variable group, as ESMValTool writes."""
    input_data = {}
    for model in MODELS:
        for variant in cmip_archive.members[model]:
            alias = f"{model}_{variant}"
            tas = _merged(cmip_archive, "tas", model, variant)
            # ESMValTool recipes deliver gw as a global-mean time series
            groups = {"pr": _merged(cmip_archive, "pr", model, variant),
                      "gw": tas.weighted(np.cos(np.deg2rad(tas.lat))).mean(("lat", "lon"))}
            # Driver groups: already-reduced scalars (future anomaly per model)
            groups["dA"] = xr.Dataset({"dA": ("time", np.full(
                30, cmip_archive.gw[model] * cmip_archive.a[model]))})
            for group, ds in groups.items():
                f = tmp_path / f"{group}_{alias}.nc"
                ds.to_netcdf(f)
                short = "tas" if group == "gw" else group
                input_data[str(f)] = {"filename": str(f), "dataset": model, "alias": alias,
                                      "variable_group": group, "short_name": short}
    return {"input_data": input_data, "plot_dir": str(tmp_path / "plots")}


@pytest.fixture
def user_config(main_config):
    return {**main_config, "variant_selection": "mean"}


def run_targets(config, user_cfg):
    """process_var() returns None in this version: read the file it writes."""
    ESMValProcessor(config, user_cfg).process_var()
    with xr.open_dataset(f"{user_cfg['work_dir']}/target_pr.nc") as ds:
        return ds.load()


# Each member is normalised by its own GW before averaging, so the +-1% member
# jitter of the synthetic GW gives a ~1e-4 relative difference for
# multi-member models; hence rtol=1e-3.
def test_esmval_processor_targets(esmval_config, user_config, cmip_archive):
    combined = run_targets(esmval_config, user_config)
    assert sorted(combined.model.values) == sorted(MODELS)
    for m in ("CanESM5", "CESM2"):
        np.testing.assert_allclose(combined.pr.sel(model=m).values,
                                   cmip_archive.expected_pattern(m), rtol=1e-3, atol=1e-6)
    assert int(combined.n_members.sel(model="CanESM5")) == 2


def test_variant_selection_first(esmval_config, user_config):
    combined = run_targets(esmval_config, {**user_config, "variant_selection": "first"})
    assert combined.member_ids.sel(model="MIROC6").item() == "MIROC6_r1i1p1f1"


def test_skipped_alias_does_not_shift_models(esmval_config, user_config, cmip_archive):
    """Was a silent bug: zip(all_model_names, changes) shifted every later
    model's data onto the wrong name once one alias was skipped."""
    meta = esmval_config["input_data"]
    drop = [k for k, m in meta.items() if m["alias"] == "ACCESS-CM2_r1i1p1f1"
            and m["variable_group"] == "pr"]
    for k in drop:
        meta.pop(k)
    combined = run_targets(esmval_config, user_config)
    assert "ACCESS-CM2" not in combined.model.values
    np.testing.assert_allclose(combined.pr.sel(model="CanESM5").values,
                               cmip_archive.expected_pattern("CanESM5"),
                               rtol=1e-3, atol=1e-6)


def test_driver_indices_then_regression(esmval_config, user_config, cmip_archive, tmp_path):
    ESMValProcessor(esmval_config, user_config).process_var()
    raw, scaled, _ = driver_indices({
        "input_data": esmval_config["input_data"], "work_dir": user_config["work_dir"],
        "season_months": [12, 1, 2], "ref_period": ("1960", "1990"),
        "future_period": ("2070", "2099")})
    assert list(raw.columns) == ["dA"]
    np.testing.assert_allclose(scaled["dA"], cmip_archive.a.loc[scaled.index], rtol=1e-6)
    files = compute_regression({"work_dir": user_config["work_dir"], "target_variable": ["pr"]})
    assert len(files) == 4


def test_esmval_driver_fields(esmval_config, user_config):
    cfg = {**user_config, "var_name": ["pr"]}
    out = ESMValProcessor(esmval_config, cfg, {"var_name": ["pr"]}).process_driver()
    assert out["pr"].sizes["model"] == sum(1 for m in esmval_config["input_data"].values()
                                           if m["variable_group"] == "pr")
