"""Model inference for B5 Drift Monitor."""

import json
from typing import List

import joblib
import pandas as pd

from src.data.validation import validate_training_stats
from src.logger import get_logger

MODEL_PATH = "models/random_forest.pkl"
STATS_PATH = "models/training_stats.json"


def predict(features: List[float]) -> dict:
    """Stub predict function (scaffold compatibility)."""
    pass


class ModelServer:
    """Loads the trained RandomForest model and serves predictions."""

    def __init__(self, config: dict) -> None:
        self._logger = get_logger(__name__)
        self.model = joblib.load(MODEL_PATH)
        with open(STATS_PATH) as fh:
            stats = json.load(fh)
        stats_df = pd.DataFrame(stats).T.astype(float)
        validate_training_stats(stats_df)
        self.training_stats = stats
        self.n_predictions = 0
        self.n_drift_events = 0
        self._logger.info("ModelServer initialized")

    def predict(self, features: List[float]) -> dict:
        """Run inference and return prediction, probability, and drift_warning."""
        feature_names = list(self.training_stats.keys())
        X = pd.DataFrame([features], columns=feature_names)
        pred = int(self.model.predict(X)[0])
        prob = float(self.model.predict_proba(X)[0][1])
        self.n_predictions += 1
        return {"prediction": pred, "probability": prob, "drift_warning": False}
