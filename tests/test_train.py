"""Tests for src/model/train.py and src/model/predict.py."""

import json
import os
from unittest.mock import MagicMock

import joblib
import mlflow  # noqa: F401 — imported so patch resolver finds it in sys.modules
import numpy as np
import pandas as pd
import pytest
import shap  # noqa: F401 — imported so patch resolver finds it in sys.modules

import src.model.predict as predict_module
import src.model.train as train_module
from src.data.validation import validate_training_stats
from src.model.train import train_model

rng = np.random.default_rng(0)

_N = 120
_FEATS = [f"F{i}" for i in range(5)]
FAKE_X = pd.DataFrame(rng.standard_normal(size=(_N, 5)), columns=_FEATS)
FAKE_Y = pd.Series(rng.integers(0, 2, size=_N).astype(int))

TRAIN_CONFIG: dict = {
    "data": {"test_size": 0.2, "seed": 42},
    "model": {"n_estimators": 5, "max_depth": 3},
    "shap": {"n_background_samples": 10},
    "mlflow": {"experiment_name": "test-b5"},
}


def _mock_shap_explainer(model, data):
    expl = MagicMock()
    sv = rng.standard_normal(size=(len(data), data.shape[1]))
    expl.shap_values.return_value = [sv, sv]
    return expl


def _mock_mlflow():
    mock_mlf = MagicMock()
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=None)
    ctx.__exit__ = MagicMock(return_value=False)
    mock_mlf.start_run.return_value = ctx
    return mock_mlf


def test_model_trains_and_saves(tmp_path, monkeypatch):
    model_path = str(tmp_path / "rf.pkl")
    stats_path = str(tmp_path / "stats.json")
    shap_path = str(tmp_path / "shap.json")
    train_pq = str(tmp_path / "train.parquet")
    test_pq = str(tmp_path / "test.parquet")

    monkeypatch.setattr(train_module, "MODEL_PATH", model_path)
    monkeypatch.setattr(train_module, "STATS_PATH", stats_path)
    monkeypatch.setattr(train_module, "SHAP_PATH", shap_path)
    monkeypatch.setattr(train_module, "TRAIN_PARQUET", train_pq)
    monkeypatch.setattr(train_module, "TEST_PARQUET", test_pq)
    monkeypatch.setattr(
        train_module, "load_credit_default", lambda cfg: (FAKE_X, FAKE_Y)
    )
    monkeypatch.setattr(train_module.shap, "TreeExplainer", _mock_shap_explainer)
    monkeypatch.setattr(train_module, "mlflow", _mock_mlflow())

    train_model(TRAIN_CONFIG)

    assert os.path.exists(model_path)
    loaded = joblib.load(model_path)
    assert hasattr(loaded, "predict")


def test_training_stats_saved(tmp_path, monkeypatch):
    model_path = str(tmp_path / "rf.pkl")
    stats_path = str(tmp_path / "stats.json")
    shap_path = str(tmp_path / "shap.json")
    train_pq = str(tmp_path / "train.parquet")
    test_pq = str(tmp_path / "test.parquet")

    monkeypatch.setattr(train_module, "MODEL_PATH", model_path)
    monkeypatch.setattr(train_module, "STATS_PATH", stats_path)
    monkeypatch.setattr(train_module, "SHAP_PATH", shap_path)
    monkeypatch.setattr(train_module, "TRAIN_PARQUET", train_pq)
    monkeypatch.setattr(train_module, "TEST_PARQUET", test_pq)
    monkeypatch.setattr(
        train_module, "load_credit_default", lambda cfg: (FAKE_X, FAKE_Y)
    )
    monkeypatch.setattr(train_module.shap, "TreeExplainer", _mock_shap_explainer)
    monkeypatch.setattr(train_module, "mlflow", _mock_mlflow())

    train_model(TRAIN_CONFIG)

    assert os.path.exists(stats_path)
    with open(stats_path) as fh:
        stats = json.load(fh)
    for col in _FEATS:
        assert col in stats
        for key in ("mean", "std", "min", "max"):
            assert key in stats[col]


def test_pandera_validates_stats():
    df_valid = pd.DataFrame(
        {"mean": [0.0, 1.0], "std": [1.0, 0.5], "min": [-1.0, 0.0], "max": [1.0, 2.0]}
    )
    result = validate_training_stats(df_valid)
    assert result is not None

    df_invalid = pd.DataFrame({"mean": [0.5], "std": [0.0], "min": [0.0], "max": [1.0]})
    with pytest.raises(Exception):
        validate_training_stats(df_invalid)


def test_model_server_init(tmp_path, monkeypatch):
    from sklearn.ensemble import RandomForestClassifier

    X = FAKE_X.iloc[:80]
    y = FAKE_Y.iloc[:80]
    model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
    model.fit(X, y)
    model_path = str(tmp_path / "rf.pkl")
    joblib.dump(model, model_path)

    stats = {col: {"mean": 0.0, "std": 1.0, "min": -3.0, "max": 3.0} for col in _FEATS}
    stats_path = str(tmp_path / "stats.json")
    with open(stats_path, "w") as fh:
        json.dump(stats, fh)

    monkeypatch.setattr(predict_module, "MODEL_PATH", model_path)
    monkeypatch.setattr(predict_module, "STATS_PATH", stats_path)

    from src.model.predict import ModelServer

    server = ModelServer({})
    assert server.n_predictions == 0
    assert server.n_drift_events == 0


def test_model_server_predict(tmp_path, monkeypatch):
    from sklearn.ensemble import RandomForestClassifier

    X = FAKE_X.iloc[:80]
    y = FAKE_Y.iloc[:80]
    model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
    model.fit(X, y)
    model_path = str(tmp_path / "rf.pkl")
    joblib.dump(model, model_path)

    stats = {col: {"mean": 0.0, "std": 1.0, "min": -3.0, "max": 3.0} for col in _FEATS}
    stats_path = str(tmp_path / "stats.json")
    with open(stats_path, "w") as fh:
        json.dump(stats, fh)

    monkeypatch.setattr(predict_module, "MODEL_PATH", model_path)
    monkeypatch.setattr(predict_module, "STATS_PATH", stats_path)

    from src.model.predict import ModelServer

    server = ModelServer({})
    result = server.predict([0.0] * 5)
    assert "prediction" in result
    assert "probability" in result
    assert "drift_warning" in result
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0
    assert server.n_predictions == 1
