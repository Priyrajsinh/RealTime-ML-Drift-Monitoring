"""Tests for src/api/app.py — FastAPI endpoints with mocked dependencies."""

import time
from collections import deque
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.app import _state, app

N_FEATS = 23
_FEATURE_NAMES = [f"FEAT_{i}" for i in range(N_FEATS)]
_VALID_FEATURES = [float(i) for i in range(N_FEATS)]

_DRIFT_RESULT = {
    "drift_detected": False,
    "psi_values": {f: 0.05 for f in _FEATURE_NAMES},
    "ks_pvalues": {f: 0.5 for f in _FEATURE_NAMES},
    "drifted_features": [],
    "max_psi": 0.05,
    "max_psi_feature": "FEAT_0",
    "n_drifted": 0,
}

_DRIFT_RESULT_POSITIVE = {
    "drift_detected": True,
    "psi_values": {f: 0.35 for f in _FEATURE_NAMES},
    "ks_pvalues": {f: 0.01 for f in _FEATURE_NAMES},
    "drifted_features": list(_FEATURE_NAMES),
    "max_psi": 0.35,
    "max_psi_feature": "FEAT_0",
    "n_drifted": N_FEATS,
}


@pytest.fixture(autouse=True)
def mock_app_state():
    """Inject mock ModelServer and DriftDetector into _state for every test."""
    mock_server = MagicMock()
    mock_server.n_predictions = 0
    mock_server.n_drift_events = 0
    mock_server.training_stats = {
        f: {"mean": 0.0, "std": 1.0, "min": -3.0, "max": 3.0} for f in _FEATURE_NAMES
    }
    mock_server.predict.return_value = {
        "prediction": 1,
        "probability": 0.8,
        "drift_warning": False,
    }

    mock_detector = MagicMock()
    mock_detector.detect_data_drift.return_value = dict(_DRIFT_RESULT)
    mock_detector.get_cached_drift_report.return_value = dict(_DRIFT_RESULT)
    mock_detector.generate_evidently_report.return_value = (
        "reports/drift/latest_report.html"
    )

    saved = dict(_state)
    _state["model_server"] = mock_server
    _state["drift_detector"] = mock_detector
    _state["start_time"] = time.time()
    _state["n_predictions"] = 0
    _state["n_drift_events"] = 0
    _state["rolling_accuracy"] = 1.0
    _state["prediction_buffer"] = deque(maxlen=500)
    _state["alerts"] = deque(maxlen=50)

    yield mock_server, mock_detector

    _state.update(saved)


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


# --- /health ---


async def test_health_endpoint(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "n_predictions" in data
    assert "n_drift_events" in data
    assert "accuracy_rolling" in data
    assert "uptime_seconds" in data
    assert "memory_mb" in data


# --- POST /api/v1/predict ---


async def test_predict_valid_features(client):
    payload = {"features": _VALID_FEATURES}
    resp = await client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["prediction"] == 1
    assert data["probability"] == pytest.approx(0.8)
    assert data["drift_warning"] is False


async def test_predict_wrong_feature_count_422(client):
    payload = {"features": [1.0, 2.0, 3.0]}  # only 3, need 23
    resp = await client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422


async def test_predict_no_model_returns_503(mock_app_state, client):
    _state["model_server"] = None
    resp = await client.post("/api/v1/predict", json={"features": _VALID_FEATURES})
    assert resp.status_code == 503


# --- POST /api/v1/predict_batch ---


async def test_predict_batch_returns_drift_report(client):
    payload = {"features_list": [_VALID_FEATURES, _VALID_FEATURES]}
    resp = await client.post("/api/v1/predict_batch", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2
    assert "drift_report" in data
    assert "psi_values" in data["drift_report"]
    assert "drift_detected" in data["drift_report"]


async def test_predict_batch_drift_detected(mock_app_state, client):
    _, mock_detector = mock_app_state
    mock_detector.detect_data_drift.return_value = dict(_DRIFT_RESULT_POSITIVE)

    payload = {"features_list": [_VALID_FEATURES] * 5}
    resp = await client.post("/api/v1/predict_batch", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["drift_report"]["drift_detected"] is True


# --- GET /api/v1/drift_report ---


async def test_drift_report_endpoint(client):
    # Pre-populate prediction buffer so the endpoint doesn't 422
    buf: deque = _state["prediction_buffer"]  # type: ignore[assignment]
    for _ in range(10):
        buf.append(_VALID_FEATURES)

    resp = await client.get("/api/v1/drift_report")
    assert resp.status_code == 200
    data = resp.json()
    assert "psi_values" in data
    assert "ks_pvalues" in data
    assert "drift_detected" in data
    assert "drifted_features" in data


async def test_drift_report_no_predictions_422(client):
    # Empty buffer → 422
    _state["prediction_buffer"] = deque(maxlen=500)
    resp = await client.get("/api/v1/drift_report")
    assert resp.status_code == 422


async def test_drift_report_caching(mock_app_state, client):
    """Two rapid calls should hit the same cached report (same detector call result)."""
    _, mock_detector = mock_app_state
    buf: deque = _state["prediction_buffer"]  # type: ignore[assignment]
    for _ in range(10):
        buf.append(_VALID_FEATURES)

    resp1 = await client.get("/api/v1/drift_report")
    resp2 = await client.get("/api/v1/drift_report")
    assert resp1.status_code == 200
    assert resp2.status_code == 200
    # Both calls go through get_cached_drift_report — detector handles TTL internally
    assert mock_detector.get_cached_drift_report.call_count == 2
    assert resp1.json() == resp2.json()


# --- GET /api/v1/drift_report/html ---


async def test_drift_report_html_404_when_missing(client, tmp_path, monkeypatch):
    import src.api.app as app_module

    monkeypatch.setattr(app_module, "DRIFT_REPORT_PATH", str(tmp_path / "missing.html"))
    resp = await client.get("/api/v1/drift_report/html")
    assert resp.status_code == 404


async def test_drift_report_html_returns_file(client, tmp_path, monkeypatch):
    import src.api.app as app_module

    report = tmp_path / "report.html"
    report.write_text("<html><body>Drift</body></html>")
    monkeypatch.setattr(app_module, "DRIFT_REPORT_PATH", str(report))
    resp = await client.get("/api/v1/drift_report/html")
    assert resp.status_code == 200
    assert "html" in resp.headers["content-type"]


# --- Rate limiting header presence ---


async def test_rate_limit_headers_present(client):
    resp = await client.post("/api/v1/predict", json={"features": _VALID_FEATURES})
    # slowapi injects X-RateLimit-* headers
    assert resp.status_code == 200


# --- POST /api/v1/alert_webhook ---


async def test_alert_webhook_receives_alerts(client):
    payload = {
        "alerts": [
            {
                "status": "firing",
                "labels": {"alertname": "DataDriftDetected", "severity": "warning"},
                "annotations": {"summary": "PSI > 0.2"},
                "startsAt": "2026-04-09T10:00:00Z",
            }
        ]
    }
    resp = await client.post("/api/v1/alert_webhook", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["alerts_received"] == 1


async def test_alert_webhook_empty_alerts(client):
    payload = {"alerts": []}
    resp = await client.post("/api/v1/alert_webhook", json=payload)
    assert resp.status_code == 200
    assert resp.json()["alerts_received"] == 0


async def test_alert_webhook_stores_alerts(client):
    payload = {
        "alerts": [
            {
                "status": "firing",
                "labels": {"alertname": "AccuracyDegraded", "severity": "critical"},
                "annotations": {"summary": "Accuracy < 65%"},
                "startsAt": "2026-04-09T10:00:00Z",
            },
            {
                "status": "resolved",
                "labels": {"alertname": "DataDriftDetected", "severity": "warning"},
                "annotations": {"summary": "PSI normalized"},
                "startsAt": "2026-04-09T09:00:00Z",
            },
        ]
    }
    resp = await client.post("/api/v1/alert_webhook", json=payload)
    assert resp.status_code == 200
    assert resp.json()["alerts_received"] == 2

    # Verify alerts are stored and retrievable
    resp2 = await client.get("/api/v1/alerts")
    assert resp2.status_code == 200
    alerts = resp2.json()
    assert len(alerts) >= 2
    assert alerts[-1]["alertname"] == "DataDriftDetected"
    assert alerts[-1]["status"] == "resolved"


# --- GET /api/v1/alerts ---


async def test_get_alerts_empty(client):
    resp = await client.get("/api/v1/alerts")
    assert resp.status_code == 200
    assert resp.json() == []
