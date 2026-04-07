"""Drift detection using PSI, KS test, and Evidently."""

import time

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.logger import get_logger

logger = get_logger(__name__)


class DriftDetector:
    """Detects covariate and concept drift in incoming data."""

    def __init__(self, training_stats: dict, config: dict) -> None:
        self.training_stats = training_stats
        drift_cfg = config.get("drift", {})
        self.psi_threshold: float = drift_cfg.get("psi_threshold", 0.2)
        self.ks_alpha: float = drift_cfg.get("ks_alpha", 0.05)
        self.report_ttl: int = drift_cfg.get("evidently_report_ttl", 60)
        self._last_report_time: float = 0.0
        self._last_report_result: dict = {}
        logger.info(
            "DriftDetector initialised — PSI threshold=%.2f, KS alpha=%.2f",
            self.psi_threshold,
            self.ks_alpha,
        )

    def compute_psi(
        self, expected: np.ndarray, actual: np.ndarray, bins: int = 10
    ) -> float:
        """Population Stability Index between two distributions.

        PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))
        Epsilon added to prevent log(0).
        """
        eps = 1e-6
        min_val = min(expected.min(), actual.min())
        max_val = max(expected.max(), actual.max())
        bin_edges = np.linspace(min_val, max_val, bins + 1)

        expected_counts = np.histogram(expected, bins=bin_edges)[0]
        actual_counts = np.histogram(actual, bins=bin_edges)[0]

        expected_pct = expected_counts / len(expected) + eps
        actual_pct = actual_counts / len(actual) + eps

        psi = float(
            np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        )
        return psi

    def compute_ks_test(
        self, expected: np.ndarray, actual: np.ndarray
    ) -> tuple[float, float]:
        """Kolmogorov-Smirnov two-sample test.

        Returns (ks_statistic, p_value).
        """
        stat, p_value = ks_2samp(expected, actual)
        return float(stat), float(p_value)

    def detect_data_drift(self, incoming_batch: pd.DataFrame) -> dict:
        """Detect DATA drift — feature distribution shift via PSI and KS.

        Returns dict with drift_detected, psi_values, ks_pvalues,
        drifted_features, n_drifted, max_psi, max_psi_feature.
        """
        psi_values: dict[str, float] = {}
        ks_pvalues: dict[str, float] = {}
        drifted_features: list[str] = []

        for feature in incoming_batch.columns:
            if feature not in self.training_stats:
                continue

            stats = self.training_stats[feature]
            expected = np.random.default_rng(42).normal(
                loc=stats["mean"], scale=stats["std"], size=len(incoming_batch)
            )
            actual = incoming_batch[feature].values

            psi = self.compute_psi(expected, actual)
            psi_values[feature] = psi

            _, p_value = self.compute_ks_test(expected, actual)
            ks_pvalues[feature] = p_value

            if psi > self.psi_threshold or p_value < self.ks_alpha:
                drifted_features.append(feature)

        max_psi = max(psi_values.values()) if psi_values else 0.0
        max_psi_feature = (
            max(psi_values, key=psi_values.get)  # type: ignore[arg-type]
            if psi_values
            else ""
        )

        result = {
            "drift_detected": len(drifted_features) > 0,
            "psi_values": psi_values,
            "ks_pvalues": ks_pvalues,
            "drifted_features": drifted_features,
            "n_drifted": len(drifted_features),
            "max_psi": max_psi,
            "max_psi_feature": max_psi_feature,
        }
        logger.info("Data drift check: %d features drifted", len(drifted_features))
        return result

    def detect_concept_drift(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        model: object,
        baseline_accuracy: float,
    ) -> dict:
        """Detect CONCEPT drift — feature-label relationship changes.

        Compares current batch accuracy against baseline.
        Accuracy drop > 10% signals concept drift.
        """
        y_pred = model.predict(X)  # type: ignore[union-attr]
        current_accuracy = float(np.mean(y_pred == y_true))
        accuracy_drop = baseline_accuracy - current_accuracy

        concept_drift_detected = accuracy_drop > 0.10
        if concept_drift_detected:
            logger.warning(
                "Concept drift detected: accuracy dropped %.2f%% (%.4f -> %.4f)",
                accuracy_drop * 100,
                baseline_accuracy,
                current_accuracy,
            )

        return {
            "concept_drift_detected": concept_drift_detected,
            "current_accuracy": current_accuracy,
            "baseline_accuracy": baseline_accuracy,
            "accuracy_drop": accuracy_drop,
        }

    def generate_evidently_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        output_path: str,
    ) -> str:
        """Generate Evidently HTML drift report using DataDriftPreset."""
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report([DataDriftPreset()])
        snapshot = report.run(reference_data=reference_data, current_data=current_data)
        snapshot.save_html(output_path)
        logger.info("Evidently report saved to %s", output_path)
        return output_path

    def get_cached_drift_report(
        self, reference_data: pd.DataFrame, current_data: pd.DataFrame
    ) -> dict:
        """Return cached drift results if within TTL, else regenerate."""
        now = time.time()
        if now - self._last_report_time < self.report_ttl and self._last_report_result:
            logger.info("Returning cached drift report")
            return self._last_report_result

        result = self.detect_data_drift(current_data)
        self._last_report_time = time.time()
        self._last_report_result = result
        logger.info("Generated fresh drift report, cached for %ds", self.report_ttl)
        return result
