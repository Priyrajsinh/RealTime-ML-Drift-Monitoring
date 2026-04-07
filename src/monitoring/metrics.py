"""Prometheus metric definitions for B5 monitoring."""

from prometheus_client import Counter, Gauge, Histogram

prediction_counter = Counter(
    "b5_predictions_total",
    "Total number of predictions served",
)

drift_event_counter = Counter(
    "b5_drift_events_total",
    "Total number of drift events detected",
)

psi_gauge = Gauge(
    "b5_drift_psi_value",
    "Most recent PSI value for drift detection",
    ["feature"],
)

prediction_latency = Histogram(
    "b5_prediction_latency_seconds",
    "Prediction endpoint latency in seconds",
)

rolling_accuracy_gauge = Gauge(
    "b5_rolling_accuracy",
    "Rolling accuracy over last N predictions",
)
