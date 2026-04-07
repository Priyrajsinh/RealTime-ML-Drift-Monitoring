"""Tests for src/data/validation.py."""

import pandas as pd
import pytest

from src.data.validation import validate_training_stats


def _make_stats(**overrides) -> pd.DataFrame:
    base = {"mean": [0.5, 1.2], "std": [0.1, 0.3], "min": [0.0, 0.0], "max": [1.0, 2.0]}
    base.update(overrides)
    return pd.DataFrame(base)


def test_valid_stats_pass():
    df = _make_stats()
    result = validate_training_stats(df)
    assert result is not None


def test_zero_std_fails():
    df = _make_stats(std=[0.0, 0.1], mean=[0.5, 1.2], min=[0.0, 0.0], max=[1.0, 2.0])
    with pytest.raises(Exception):
        validate_training_stats(df)


def test_min_greater_than_max_fails():
    df = pd.DataFrame(
        {
            "mean": [0.5],
            "std": [0.1],
            "min": [2.0],
            "max": [1.0],
        }
    )
    with pytest.raises(Exception):
        validate_training_stats(df)
