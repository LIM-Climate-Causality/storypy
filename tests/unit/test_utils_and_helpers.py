"""Unit tests for storypy.utils, storypy._compat and processor set-up."""

import os

import pandas as pd
import pytest

from storypy._compat import group_metadata
from storypy.preprocess import ModelDataPreprocessor
from storypy.utils import find_driver_csv
from tests.helpers import compare_driver_csvs


def test_find_driver_csv(tmp_path, capsys):
    with pytest.raises(FileNotFoundError):
        find_driver_csv(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        find_driver_csv(str(tmp_path), explicit=str(tmp_path / "x.csv"))
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "scaled_standardized_drivers.csv").write_text("model,x\n")
    assert find_driver_csv(str(tmp_path)).endswith(os.path.join("a", "scaled_standardized_drivers.csv"))
    assert "Multiple driver CSVs" in capsys.readouterr().out


def test_group_metadata():
    meta = [{"dataset": "A", "alias": "A_2"}, {"dataset": "B", "alias": "B_1"},
            {"dataset": "A", "alias": "A_1"}]
    grouped = group_metadata(meta, "dataset", sort="alias")
    assert [m["alias"] for m in grouped["A"]] == ["A_1", "A_2"]
    assert len(grouped["B"]) == 1


def test_preprocessor_validation(main_config):
    with pytest.raises(ValueError, match="must match"):
        ModelDataPreprocessor(main_config, {"var_name": ["a", "b"], "short_name": ["a"]})


def test_compare_driver_csvs(tmp_path):
    for d, v in (("x", 1.0), ("y", 1.001), ("z", 2.0)):
        (tmp_path / d).mkdir()
        pd.DataFrame({"a": [v]}, index=["m"]).to_csv(tmp_path / d / "drivers.csv")
    assert compare_driver_csvs(tmp_path / "x", tmp_path / "y")
    assert not compare_driver_csvs(tmp_path / "x", tmp_path / "z")
