"""FastAPI application for B5 Drift Monitor."""

import os
import time
from collections import deque
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import psutil
import yaml
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.data.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    DriftReport,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from src.logger import get_logger
from src.model.predict import ModelServer
from src.monitoring.drift_detector import DriftDetector
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

logger = get_logger(__name__)

with open("config/config.yaml") as _fh:
    _config = yaml.safe_load(_fh)

DRIFT_REPORT_PATH = "reports/drift/latest_report.html"
_ROLLING_WINDOW: int = int(_config["monitoring"]["rolling_window"])

_state: dict[str, object] = {
    "model_server": None,
    "drift_detector": None,
    "start_time": None,
    "prediction_buffer": deque(maxlen=500),
    "n_predictions": 0,
    "n_drift_events": 0,
    "rolling_accuracy": 1.0,
}

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load ModelServer and DriftDetector on startup."""
    _state["start_time"] = time.time()
    try:
        server = ModelServer(_config)
        detector = DriftDetector(server.training_stats, _config)
        _state["model_server"] = server
        _state["drift_detector"] = detector
        logger.info("startup_complete")
    except Exception as exc:
        logger.warning("model_not_loaded", extra={"reason": str(exc)})
    yield
    logger.info("shutdown")


app = FastAPI(
    title="B5 Drift Monitor API",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler,  # type: ignore[arg-type]
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_config["api"]["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def check_content_length(request: Request, call_next):
    """Reject payloads exceeding max_payload_mb from config."""
    max_bytes = int(_config["api"]["max_payload_mb"]) * 1024 * 1024
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > max_bytes:
        mb = _config["api"]["max_payload_mb"]
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={"detail": f"Payload exceeds {mb} MB limit."},
        )
    return await call_next(request)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Detailed health status including model state, uptime, and memory."""
    proc = psutil.Process()
    start_ts = _state["start_time"]
    start = float(start_ts) if isinstance(start_ts, float) else time.time()
    return HealthResponse(
        status="healthy",
        model_loaded=_state["model_server"] is not None,
        n_predictions=int(_state["n_predictions"]),  # type: ignore[arg-type]
        n_drift_events=int(_state["n_drift_events"]),  # type: ignore[arg-type]
        accuracy_rolling=float(_state["rolling_accuracy"]),  # type: ignore[arg-type]
        uptime_seconds=round(time.time() - start, 2),
        memory_mb=round(proc.memory_info().rss / 1024 / 1024, 2),
    )


@app.post("/api/v1/predict", response_model=PredictResponse, tags=["inference"])
@limiter.limit(_config["api"]["rate_limit_predict"])
async def predict_single(request: Request, body: PredictRequest) -> PredictResponse:
    """Run single prediction. Appends to rolling buffer for drift checks."""
    server = _state["model_server"]
    if server is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.time()
    result = server.predict(body.features)  # type: ignore[union-attr]
    latency = time.time() - t0

    PREDICTION_COUNTER.inc()
    PREDICTION_LATENCY.observe(latency)
    n_pred = int(_state["n_predictions"]) + 1  # type: ignore[arg-type]
    _state["n_predictions"] = n_pred
    N_PREDICTIONS.set(n_pred)

    buf: deque = _state["prediction_buffer"]  # type: ignore[assignment]
    buf.append(body.features)

    # Run drift check every ROLLING_WINDOW predictions
    if n_pred % _ROLLING_WINDOW == 0 and len(buf) >= 10:
        _check_drift_on_buffer(buf)

    return PredictResponse(
        prediction=int(result["prediction"]),
        probability=float(result["probability"]),
        drift_warning=bool(result["drift_warning"]),
    )


def _check_drift_on_buffer(buf: deque) -> None:
    """Run drift detection on the prediction buffer and update metrics."""
    detector = _state["drift_detector"]
    server = _state["model_server"]
    if detector is None or server is None:
        return

    feature_names = list(server.training_stats.keys())  # type: ignore[union-attr]
    df = pd.DataFrame(list(buf), columns=feature_names)
    drift_result: dict = detector.detect_data_drift(df)  # type: ignore[union-attr]

    if drift_result["drift_detected"]:
        _apply_drift_metrics(drift_result)


def _apply_drift_metrics(drift_result: dict) -> None:
    """Increment drift counters and update Prometheus gauges."""
    DRIFT_DETECTED_COUNTER.inc()
    n_drift = int(_state["n_drift_events"]) + 1  # type: ignore[arg-type]
    _state["n_drift_events"] = n_drift
    N_DRIFT_EVENTS.set(n_drift)
    CURRENT_PSI.set(float(drift_result["max_psi"]))
    for feature, psi in drift_result["psi_values"].items():
        FEATURE_PSI.labels(feature_name=feature).set(float(psi))
    rolling_acc = float(_state["rolling_accuracy"])  # type: ignore[arg-type]
    MODEL_ACCURACY_ROLLING.set(rolling_acc)


@app.post(
    "/api/v1/predict_batch",
    response_model=BatchPredictResponse,
    tags=["inference"],
)
async def predict_batch(body: BatchPredictRequest) -> BatchPredictResponse:
    """Run batch predictions and detect drift on the submitted batch."""
    server = _state["model_server"]
    detector = _state["drift_detector"]
    if server is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if detector is None:
        raise HTTPException(status_code=503, detail="Drift detector not loaded")

    predictions: list[PredictResponse] = []
    for features in body.features_list:
        result = server.predict(features)  # type: ignore[union-attr]
        PREDICTION_COUNTER.inc()
        n_pred = int(_state["n_predictions"]) + 1  # type: ignore[arg-type]
        _state["n_predictions"] = n_pred
        predictions.append(
            PredictResponse(
                prediction=int(result["prediction"]),
                probability=float(result["probability"]),
                drift_warning=bool(result["drift_warning"]),
            )
        )
    N_PREDICTIONS.set(int(_state["n_predictions"]))  # type: ignore[arg-type]

    feature_names = list(server.training_stats.keys())  # type: ignore[union-attr]
    df = pd.DataFrame(body.features_list, columns=feature_names)
    drift_result: dict = detector.detect_data_drift(df)  # type: ignore[union-attr]

    if drift_result["drift_detected"]:
        _apply_drift_metrics(drift_result)

    report = DriftReport(
        psi_values=drift_result["psi_values"],
        ks_pvalues=drift_result["ks_pvalues"],
        drift_detected=drift_result["drift_detected"],
        drifted_features=drift_result["drifted_features"],
    )
    return BatchPredictResponse(predictions=predictions, drift_report=report)


@app.get("/api/v1/drift_report", response_model=DriftReport, tags=["monitoring"])
async def drift_report() -> DriftReport:
    """Return drift report (cached for TTL seconds per config)."""
    detector = _state["drift_detector"]
    if detector is None:
        raise HTTPException(status_code=503, detail="Drift detector not loaded")

    buf: deque = _state["prediction_buffer"]  # type: ignore[assignment]
    server = _state["model_server"]
    if server is None or len(buf) == 0:
        raise HTTPException(
            status_code=422,
            detail="No predictions recorded yet — send predictions first.",
        )

    feature_names = list(server.training_stats.keys())  # type: ignore[union-attr]
    ref_df = pd.DataFrame(np.zeros((10, len(feature_names))), columns=feature_names)
    cur_df = pd.DataFrame(list(buf), columns=feature_names)

    result: dict = detector.get_cached_drift_report(  # type: ignore[union-attr]
        ref_df, cur_df
    )

    try:
        os.makedirs("reports/drift", exist_ok=True)
        detector.generate_evidently_report(  # type: ignore[union-attr]
            ref_df, cur_df, DRIFT_REPORT_PATH
        )
    except Exception as exc:
        logger.warning("evidently_report_failed", extra={"reason": str(exc)})

    return DriftReport(
        psi_values=result["psi_values"],
        ks_pvalues=result["ks_pvalues"],
        drift_detected=result["drift_detected"],
        drifted_features=result["drifted_features"],
    )


@app.get("/api/v1/drift_report/html", tags=["monitoring"])
async def drift_report_html() -> FileResponse:
    """Return the latest Evidently HTML drift report."""
    if not os.path.exists(DRIFT_REPORT_PATH):
        raise HTTPException(
            status_code=404,
            detail=(
                "No HTML report found. Call GET /api/v1/drift_report first "
                "to generate it."
            ),
        )
    return FileResponse(DRIFT_REPORT_PATH, media_type="text/html")
