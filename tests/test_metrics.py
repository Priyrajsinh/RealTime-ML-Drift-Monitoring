"""Tests for src/monitoring/metrics.py — Prometheus metric definitions."""

from prometheus_client import REGISTRY

from src.monitoring.metrics import (
    CURRENT_PSI,
    DRIFT_DETECTED_COUNTER,
    FEATURE_PSI,
    MODEL_ACCURACY_ROLLING,
    N_DRIFT_EVENTS,
    N_PREDICTIONS,
    PREDICTION_COUNTER,
    PREDICTION_LATENCY,
)


def _sample(collector, labels=None):
    """Return the first sample value for a metric, optionally filtered by labels."""
    for metric in REGISTRY.collect():
        if metric.name == collector._name:
            for sample in metric.samples:
                if labels is None or all(
                    sample.labels.get(k) == v for k, v in labels.items()
                ):
                    return sample.value
    return None


def test_prediction_counter_increments():
    before = PREDICTION_COUNTER._value.get()
    PREDICTION_COUNTER.inc()
    after = PREDICTION_COUNTER._value.get()
    assert after == before + 1.0


def test_prediction_latency_observes():
    PREDICTION_LATENCY.observe(0.05)
    # Histogram sum should be positive
    found = False
    for metric in REGISTRY.collect():
        if metric.name == "prediction_latency_seconds":
            for sample in metric.samples:
                if sample.name == "prediction_latency_seconds_sum":
                    assert sample.value > 0
                    found = True
    assert found


def test_psi_gauge_sets_value():
    CURRENT_PSI.set(0.42)
    val = _sample(CURRENT_PSI)
    assert val == 0.42


def test_feature_psi_labels():
    FEATURE_PSI.labels(feature_name="AGE").set(0.15)
    FEATURE_PSI.labels(feature_name="LIMIT_BAL").set(0.31)
    age_val = _sample(FEATURE_PSI, labels={"feature_name": "AGE"})
    lim_val = _sample(FEATURE_PSI, labels={"feature_name": "LIMIT_BAL"})
    assert age_val == 0.15
    assert lim_val == 0.31


def test_model_accuracy_rolling_sets_value():
    MODEL_ACCURACY_ROLLING.set(0.87)
    val = _sample(MODEL_ACCURACY_ROLLING)
    assert val == 0.87


def test_n_predictions_gauge_sets_value():
    N_PREDICTIONS.set(500)
    val = _sample(N_PREDICTIONS)
    assert val == 500.0


def test_n_drift_events_gauge_sets_value():
    N_DRIFT_EVENTS.set(3)
    val = _sample(N_DRIFT_EVENTS)
    assert val == 3.0


def test_drift_detected_counter_increments():
    before = DRIFT_DETECTED_COUNTER._value.get()
    DRIFT_DETECTED_COUNTER.inc()
    after = DRIFT_DETECTED_COUNTER._value.get()
    assert after == before + 1.0
