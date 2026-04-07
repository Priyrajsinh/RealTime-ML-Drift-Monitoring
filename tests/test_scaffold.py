"""Scaffold tests for B5 Drift Monitor."""

import logging

import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from src.api.app import app
from src.data.dataset import load_credit_default
from src.data.schemas import PredictRequest
from src.data.validation import validate_training_stats
from src.exceptions import (
    ConfigError,
    DataLoadError,
    DataValidationError,
    DriftDetectionError,
    ModelNotFoundError,
    PredictionError,
    ProjectBaseError,
)
from src.logger import get_logger
from src.model.predict import predict
from src.model.train import train_model
from src.monitoring import metrics
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.drift_simulator import DriftSimulator
from src.monitoring.shap_drift import compare_shap_under_drift


def test_config_loads():
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["drift"]["psi_threshold"] == 0.2
    assert cfg["monitoring"]["rolling_window"] == 100
    assert cfg["data"]["n_features"] == 23


def test_get_logger_no_duplicate_handlers():
    logger = get_logger("test.scaffold")
    _ = get_logger("test.scaffold")
    assert len(logger.handlers) == 1
    assert isinstance(logger, logging.Logger)


def test_predict_request_valid():
    req = PredictRequest(features=[0.0] * 23)
    assert len(req.features) == 23


def test_predict_request_wrong_length():
    with pytest.raises(Exception):
        PredictRequest(features=[0.0] * 10)


def test_stats_schema_valid():
    df = pd.DataFrame(
        {
            "mean": [0.5, 1.2],
            "std": [0.1, 0.3],
            "min": [0.0, 0.0],
            "max": [1.0, 2.0],
        }
    )
    result = validate_training_stats(df)
    assert result is not None


def test_stats_schema_rejects_zero_std():
    df = pd.DataFrame(
        {
            "mean": [0.5],
            "std": [0.0],
            "min": [0.0],
            "max": [1.0],
        }
    )
    with pytest.raises(Exception):
        validate_training_stats(df)


# --- exceptions ---


def test_exception_hierarchy():
    assert issubclass(DataLoadError, ProjectBaseError)
    assert issubclass(DataValidationError, ProjectBaseError)
    assert issubclass(ModelNotFoundError, ProjectBaseError)
    assert issubclass(PredictionError, ProjectBaseError)
    assert issubclass(DriftDetectionError, ProjectBaseError)
    assert issubclass(ConfigError, ProjectBaseError)


def test_exceptions_raise():
    with pytest.raises(DataLoadError):
        raise DataLoadError("test")
    with pytest.raises(ModelNotFoundError):
        raise ModelNotFoundError("model.pkl missing")


# --- stubs return None ---


def test_load_credit_default_returns_tuple():
    X, y = load_credit_default({})
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape == (30000, 23)


def test_train_model_stub():
    result = train_model({})
    assert result is None


def test_predict_stub():
    result = predict([0.0] * 23)
    assert result is None


def test_compare_shap_stub():
    result = compare_shap_under_drift()
    assert result is None


# --- monitoring stubs ---


def test_drift_detector_instantiates():
    d = DriftDetector()
    assert d is not None


def test_drift_simulator_instantiates():
    s = DriftSimulator()
    assert s is not None


def test_metrics_module_loads():
    assert metrics.prediction_counter is not None
    assert metrics.drift_event_counter is not None
    assert metrics.psi_gauge is not None
    assert metrics.rolling_accuracy_gauge is not None


# --- api health endpoint ---


def test_health_endpoint():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is False
