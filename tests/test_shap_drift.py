"""Tests for src/monitoring/shap_drift.py — SHAP mocked for speed."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.monitoring.shap_drift import compare_shap_under_drift, plot_shap_comparison

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FEATS = ["F0", "F1", "F2", "F3", "F4"]
_CONFIG: dict = {
    "shap": {
        "n_background_samples": 10,
        "n_drift_samples": 20,
    }
}
_BASELINE: dict = {
    "F0": 0.30,
    "F1": 0.20,
    "F2": 0.15,
    "F3": 0.10,
    "F4": 0.05,
}

rng = np.random.default_rng(1)


def _make_df(n: int = 50) -> pd.DataFrame:
    return pd.DataFrame(rng.standard_normal((n, 5)), columns=_FEATS)


def _make_shap_explainer_mock(shap_values_array: np.ndarray) -> MagicMock:
    """Return a mock TreeExplainer whose shap_values() returns a fixed array."""
    explainer = MagicMock()
    explainer.shap_values.return_value = shap_values_array
    return explainer


# ---------------------------------------------------------------------------
# compare_shap_under_drift
# ---------------------------------------------------------------------------


@patch("src.monitoring.shap_drift.shap")
def test_shap_comparison_returns_top_5(mock_shap):
    """Function must return baseline_top_5 and drifted_top_5 with exactly 5 items."""
    n = 50
    fake_shap_vals = rng.standard_normal((n, 5))
    mock_shap.TreeExplainer.return_value = _make_shap_explainer_mock(fake_shap_vals)

    model = MagicMock()
    X_train = _make_df(50)
    X_drifted = _make_df(50)

    result = compare_shap_under_drift(model, X_train, X_drifted, _BASELINE, _CONFIG)

    assert len(result["baseline_top_5"]) == 5
    assert len(result["drifted_top_5"]) == 5


@patch("src.monitoring.shap_drift.shap")
def test_shap_comparison_returns_required_keys(mock_shap):
    n = 50
    mock_shap.TreeExplainer.return_value = _make_shap_explainer_mock(
        rng.standard_normal((n, 5))
    )
    model = MagicMock()
    result = compare_shap_under_drift(
        model, _make_df(50), _make_df(50), _BASELINE, _CONFIG
    )
    required = {"baseline_top_5", "drifted_top_5", "biggest_shifts", "interpretation"}
    assert required.issubset(set(result.keys()))


@patch("src.monitoring.shap_drift.shap")
def test_shap_comparison_detects_shift(mock_shap):
    """F4 dominates drifted SHAP — must appear in biggest_shifts."""
    n = 50
    # F4 (index 4) has 10x the SHAP importance in drifted vs baseline
    fake_shap_vals = np.zeros((n, 5))
    fake_shap_vals[:, 4] = 5.0  # F4 dominates drifted importance
    mock_shap.TreeExplainer.return_value = _make_shap_explainer_mock(fake_shap_vals)

    model = MagicMock()
    result = compare_shap_under_drift(
        model, _make_df(50), _make_df(50), _BASELINE, _CONFIG
    )

    shifted_features = [f for f, _ in result["biggest_shifts"]]
    assert (
        "F4" in shifted_features
    ), f"Expected F4 in biggest_shifts, got {shifted_features}"


@patch("src.monitoring.shap_drift.shap")
def test_shap_comparison_interpretation_is_string(mock_shap):
    mock_shap.TreeExplainer.return_value = _make_shap_explainer_mock(
        rng.standard_normal((50, 5))
    )
    model = MagicMock()
    result = compare_shap_under_drift(
        model, _make_df(50), _make_df(50), _BASELINE, _CONFIG
    )
    assert isinstance(result["interpretation"], str)
    assert len(result["interpretation"]) > 0


@patch("src.monitoring.shap_drift.shap")
def test_shap_comparison_handles_binary_class_list(mock_shap):
    """shap_values returned as [class0, class1] list — should use class1."""
    n = 50
    sv_class0 = np.zeros((n, 5))
    sv_class1 = rng.standard_normal((n, 5))
    mock_shap.TreeExplainer.return_value = _make_shap_explainer_mock(
        [sv_class0, sv_class1]
    )
    model = MagicMock()
    result = compare_shap_under_drift(
        model, _make_df(50), _make_df(50), _BASELINE, _CONFIG
    )
    # If class1 used correctly, drifted importances should be non-zero
    assert any(v > 0 for _, v in result["drifted_top_5"])


# ---------------------------------------------------------------------------
# plot_shap_comparison
# ---------------------------------------------------------------------------


def test_plot_shap_comparison_saves_file(tmp_path):
    comparison = {
        "baseline_top_5": [
            ("F0", 0.30),
            ("F1", 0.20),
            ("F2", 0.15),
            ("F3", 0.10),
            ("F4", 0.05),
        ],
        "drifted_top_5": [
            ("F4", 0.40),
            ("F0", 0.25),
            ("F1", 0.18),
            ("F2", 0.12),
            ("F3", 0.08),
        ],
        "biggest_shifts": [
            ("F4", 0.35),
            ("F3", -0.02),
            ("F0", -0.05),
            ("F1", -0.02),
            ("F2", -0.03),
        ],
        "interpretation": "F4 rose from rank 5 to rank 1",
    }
    out = str(tmp_path / "shap_drift_comparison.png")
    plot_shap_comparison(comparison, out)
    assert os.path.exists(out)


def test_plot_shap_comparison_empty_shifts(tmp_path, caplog):
    """Empty biggest_shifts should log a warning and not crash."""
    comparison = {
        "baseline_top_5": [],
        "drifted_top_5": [],
        "biggest_shifts": [],
        "interpretation": "No shift.",
    }
    out = str(tmp_path / "shap_empty.png")
    plot_shap_comparison(comparison, out)
    # Should not raise and file should NOT be created (function returns early)
    assert not os.path.exists(out)
