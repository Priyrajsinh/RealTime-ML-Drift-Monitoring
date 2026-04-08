"""Prometheus metric definitions for B5 monitoring.

All metrics defined here — imported by other modules, never re-created.
Rule 21: metric names use underscores, NOT hyphens.
Rule 23: Gauge metrics use .set(), NOT .inc().
"""

from prometheus_client import Counter, Gauge, Histogram

# Prediction metrics
PREDICTION_COUNTER = Counter(
    "prediction_total",
    "Total number of predictions served",
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

# Drift metrics
DRIFT_DETECTED_COUNTER = Counter(
    "drift_detected_total",
    "Total number of drift events detected",
)
CURRENT_PSI = Gauge(
    "current_psi_max",
    "Maximum PSI value across all features",
)
FEATURE_PSI = Gauge(
    "feature_psi",
    "PSI value per feature",
    ["feature_name"],
)

# Model health metrics
MODEL_ACCURACY_ROLLING = Gauge(
    "model_accuracy_rolling",
    "Rolling accuracy over last N predictions",
)
N_PREDICTIONS = Gauge(
    "n_predictions_total",
    "Total predictions served since startup",
)
N_DRIFT_EVENTS = Gauge(
    "n_drift_events_total",
    "Total drift events since startup",
)

# Backward-compatible aliases (used by test_scaffold.py)
prediction_counter = PREDICTION_COUNTER
drift_event_counter = DRIFT_DETECTED_COUNTER
psi_gauge = FEATURE_PSI
rolling_accuracy_gauge = MODEL_ACCURACY_ROLLING
