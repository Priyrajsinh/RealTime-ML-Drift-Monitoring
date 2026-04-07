"""Pydantic v2 schemas for B5 API request/response validation."""

from typing import List

from pydantic import BaseModel, field_validator

N_FEATURES = 23


class PredictRequest(BaseModel):
    features: List[float]

    @field_validator("features")
    @classmethod
    def check_feature_length(cls, v: List[float]) -> List[float]:
        if len(v) != N_FEATURES:
            raise ValueError(f"Expected {N_FEATURES} features, got {len(v)}")
        return v


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    drift_warning: bool


class DriftReport(BaseModel):
    psi_values: dict
    ks_pvalues: dict
    drift_detected: bool
    drifted_features: List[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    n_predictions: int
    n_drift_events: int
    accuracy_rolling: float
    uptime_seconds: float
    memory_mb: float
