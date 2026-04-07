"""Tests for src/data/dataset.py."""

import json

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import (
    compute_training_stats,
    load_credit_default,
    save_training_stats,
)

rng = np.random.default_rng(42)

_COLS = ["ID"] + [f"F{i}" for i in range(23)] + ["default payment next month"]
_ARR = rng.standard_normal(size=(100, 25))
_ARR[:, -1] = rng.integers(0, 2, size=100)
FAKE_CSV = pd.DataFrame(_ARR, columns=_COLS)

CONFIG: dict = {"data": {"n_features": 23}}


def _fake_read_csv(path: str) -> pd.DataFrame:
    return FAKE_CSV.copy()


def test_load_returns_dataframe_and_series(monkeypatch):
    monkeypatch.setattr(pd, "read_csv", _fake_read_csv)
    X, y = load_credit_default(CONFIG)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_feature_count_matches_config(monkeypatch):
    monkeypatch.setattr(pd, "read_csv", _fake_read_csv)
    X, y = load_credit_default(CONFIG)
    assert X.shape[1] == 23


def test_no_id_column_in_features(monkeypatch):
    monkeypatch.setattr(pd, "read_csv", _fake_read_csv)
    X, y = load_credit_default(CONFIG)
    assert "ID" not in X.columns
    assert "default payment next month" not in X.columns


def test_compute_training_stats_keys():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    stats = compute_training_stats(X)
    for col in ["a", "b"]:
        assert "mean" in stats[col]
        assert "std" in stats[col]
        assert "min" in stats[col]
        assert "max" in stats[col]


def test_compute_training_stats_std_positive():
    X = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    stats = compute_training_stats(X)
    for col in stats:
        assert stats[col]["std"] > 0


def test_save_training_stats_writes_json(tmp_path):
    stats = {"x": {"mean": 0.5, "std": 0.1, "min": 0.0, "max": 1.0}}
    out = str(tmp_path / "stats.json")
    save_training_stats(stats, out)
    with open(out) as fh:
        loaded = json.load(fh)
    assert loaded["x"]["std"] == pytest.approx(0.1)
