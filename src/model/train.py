"""Model training for B5 Drift Monitor."""

import json
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.data.dataset import (
    compute_training_stats,
    load_credit_default,
    save_training_stats,
)
from src.data.validation import validate_training_stats
from src.logger import get_logger

logger = get_logger(__name__)

MODEL_PATH = "models/random_forest.pkl"
STATS_PATH = "models/training_stats.json"
SHAP_PATH = "models/shap_baseline.json"
TRAIN_PARQUET = "data/processed/train.parquet"
TEST_PARQUET = "data/processed/test.parquet"


def train_model(config: dict) -> None:
    """Train RandomForest classifier and save model, stats, and SHAP baseline."""
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    shap_cfg = config.get("shap", {})
    mlflow_cfg = config.get("mlflow", {})

    test_size = data_cfg.get("test_size", 0.2)
    seed = data_cfg.get("seed", 42)
    n_estimators = model_cfg.get("n_estimators", 100)
    max_depth = model_cfg.get("max_depth", 10)

    X, y = load_credit_default(config)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    # Save processed splits
    os.makedirs(os.path.dirname(TRAIN_PARQUET), exist_ok=True)
    X_train.assign(target=y_train.values).to_parquet(TRAIN_PARQUET, index=False)
    X_test.assign(target=y_test.values).to_parquet(TEST_PARQUET, index=False)

    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info("Model saved to %s", MODEL_PATH)

    # Compute and validate training stats
    stats = compute_training_stats(X_train)
    save_training_stats(stats, STATS_PATH)

    stats_df = pd.DataFrame(stats).T.astype(float)
    validate_training_stats(stats_df)
    logger.info("Training stats validated with pandera")

    # SHAP baseline
    n_background = shap_cfg.get("n_background_samples", 100)
    background = X_train.iloc[:n_background]
    explainer = shap.TreeExplainer(model, data=background)
    shap_values = explainer.shap_values(background)
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values
    mean_abs_shap: dict[str, float] = {
        col: float(np.abs(sv[:, i]).mean()) for i, col in enumerate(X_train.columns)
    }
    with open(SHAP_PATH, "w") as fh:
        json.dump(mean_abs_shap, fh, indent=2)
    logger.info("SHAP baseline saved to %s", SHAP_PATH)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_prob))
    logger.info("acc=%.4f f1=%.4f auc=%.4f", acc, f1, auc)

    # MLflow
    experiment_name = mlflow_cfg.get("experiment_name", "b5-drift-monitor")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "n_samples": len(X),
                "n_features": X.shape[1],
            }
        )
        mlflow.log_metrics({"accuracy": acc, "f1_score": f1, "roc_auc": auc})
        mlflow.log_artifact(STATS_PATH)
        mlflow.log_artifact(SHAP_PATH)
