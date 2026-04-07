"""Data loading utilities for UCI Credit Card Default dataset."""

import json
import os

import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)

DATASET_PATH = "data/raw/credit_default.csv"
STATS_PATH = "models/training_stats.json"


def load_credit_default(config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Load UCI Credit Card Default dataset from local CSV.

    Returns (X, y) where X is features DataFrame and y is the target Series.
    """
    df = pd.read_csv(DATASET_PATH)
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    target_col = "default payment next month"
    X = df.drop(columns=[target_col])
    y = df[target_col]
    logger.info(
        "Loaded %d samples with %d features from UCI Credit Default",
        len(X),
        X.shape[1],
    )
    return X, y


def compute_training_stats(X_train: pd.DataFrame) -> dict:
    """Compute per-feature statistics: mean, std, min, max."""
    stats: dict[str, dict[str, float]] = {}
    for col in X_train.columns:
        stats[col] = {
            "mean": float(X_train[col].mean()),
            "std": float(X_train[col].std()),
            "min": float(X_train[col].min()),
            "max": float(X_train[col].max()),
        }
    return stats


def save_training_stats(stats: dict, path: str) -> None:
    """Save training stats dict to JSON using built-in open()."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(stats, fh, indent=2)
