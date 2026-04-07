"""FastAPI application stub for B5 Drift Monitor."""

from fastapi import FastAPI

from src.data.schemas import HealthResponse

app = FastAPI(title="B5 Drift Monitor", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model_loaded=False,
        n_predictions=0,
        n_drift_events=0,
        accuracy_rolling=0.0,
        uptime_seconds=0.0,
        memory_mb=0.0,
    )
