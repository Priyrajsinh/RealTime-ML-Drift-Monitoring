"""Pandera schema for validating training_stats.json content."""

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

StatsSchema = DataFrameSchema(
    {
        "mean": Column(float),
        "std": Column(
            float,
            checks=Check(lambda s: (s > 0).all(), error="std must be > 0"),
        ),
        "min": Column(float),
        "max": Column(float),
    },
    checks=[
        Check(
            lambda df: (df["min"] <= df["max"]).all(),
            error="min must be <= max",
        ),
    ],
)


def validate_training_stats(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Validate that training stats DataFrame has required columns and std > 0."""
    return StatsSchema.validate(stats_df)
