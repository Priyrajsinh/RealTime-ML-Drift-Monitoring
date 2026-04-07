"""Tests for src/monitoring/drift_detector.py."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from src.monitoring.drift_detector import DriftDetector

rng = np.random.default_rng(42)

_FEATS = ["F0", "F1", "F2", "F3", "F4"]
_STATS = {col: {"mean": 0.0, "std": 1.0, "min": -3.0, "max": 3.0} for col in _FEATS}
_CONFIG: dict[str, str | int | dict] = {
    "drift": {
        "psi_threshold": 0.2,
        "ks_alpha": 0.05,
        "evidently_report_ttl": 60,
    }
}


def _make_detector() -> DriftDetector:
    return DriftDetector(_STATS, _CONFIG)


# --- PSI tests ---


def test_psi_no_drift():
    det = _make_detector()
    data = rng.standard_normal(5000)
    psi = det.compute_psi(data, data.copy())
    assert psi < 0.1


def test_psi_significant_drift():
    det = _make_detector()
    expected = rng.standard_normal(5000)
    actual = rng.standard_normal(5000) + 3.0
    psi = det.compute_psi(expected, actual)
    assert psi > 0.2


def test_psi_epsilon_prevents_log_zero():
    det = _make_detector()
    expected = np.concatenate([np.zeros(100), np.ones(100)])
    actual = np.zeros(200)
    psi = det.compute_psi(expected, actual)
    assert np.isfinite(psi)


# --- KS tests ---


def test_ks_same_distribution():
    det = _make_detector()
    data = rng.standard_normal(1000)
    _, p_value = det.compute_ks_test(data, data.copy())
    assert p_value > 0.05


def test_ks_different_distribution():
    det = _make_detector()
    expected = rng.standard_normal(1000)
    actual = rng.standard_normal(1000) + 5.0
    _, p_value = det.compute_ks_test(expected, actual)
    assert p_value < 0.05


# --- Data drift tests ---


def test_detect_data_drift_flags_shifted_features():
    det = _make_detector()
    n = 2000
    normal = pd.DataFrame(rng.standard_normal((n, 5)), columns=_FEATS)
    shifted = normal.copy()
    shifted["F1"] = shifted["F1"] + 10.0
    shifted["F3"] = shifted["F3"] + 10.0
    shifted["F4"] = shifted["F4"] + 10.0

    result = det.detect_data_drift(shifted)
    assert result["drift_detected"] is True
    for feat in ("F1", "F3", "F4"):
        assert feat in result["drifted_features"]
    assert result["n_drifted"] >= 3


# --- Concept drift tests ---


def test_detect_concept_drift_flags_accuracy_drop():
    det = _make_detector()
    X = pd.DataFrame(rng.standard_normal((200, 5)), columns=_FEATS)
    y_true = pd.Series(rng.integers(0, 2, size=200))
    model = MagicMock()
    model.predict.return_value = 1 - y_true.values

    result = det.detect_concept_drift(X, y_true, model, baseline_accuracy=0.85)
    assert result["concept_drift_detected"] is True
    assert result["accuracy_drop"] > 0.10


def test_detect_concept_drift_normal():
    det = _make_detector()
    X = pd.DataFrame(rng.standard_normal((200, 5)), columns=_FEATS)
    y_true = pd.Series(rng.integers(0, 2, size=200))
    model = MagicMock()
    model.predict.return_value = y_true.values

    result = det.detect_concept_drift(X, y_true, model, baseline_accuracy=1.0)
    assert result["concept_drift_detected"] is False
    assert result["accuracy_drop"] <= 0.10


# --- Evidently report test ---


def test_evidently_report_generates_html(tmp_path):
    det = _make_detector()
    n = 200
    ref = pd.DataFrame(rng.standard_normal((n, 3)), columns=["A", "B", "C"])
    cur = pd.DataFrame(rng.standard_normal((n, 3)), columns=["A", "B", "C"])
    out = str(tmp_path / "report.html")

    result_path = det.generate_evidently_report(ref, cur, out)
    assert result_path == out
    with open(out, encoding="utf-8") as fh:
        content = fh.read()
    assert "<html" in content.lower()


# --- Caching test ---


def test_report_caching(monkeypatch):
    det = _make_detector()
    n = 500
    ref = pd.DataFrame(rng.standard_normal((n, 5)), columns=_FEATS)
    cur = pd.DataFrame(rng.standard_normal((n, 5)), columns=_FEATS)

    result1 = det.get_cached_drift_report(ref, cur)
    t1 = det._last_report_time
    assert result1["drift_detected"] in (True, False)

    result2 = det.get_cached_drift_report(ref, cur)
    t2 = det._last_report_time
    assert t1 == t2
    assert result2 is result1
