"""Drift simulation engine — generates normal/drifted batches and portfolio plots."""

import os

import joblib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.logger import get_logger
from src.monitoring.drift_detector import DriftDetector

plt.switch_backend("Agg")

logger = get_logger(__name__)

_MODEL_PATH = "models/random_forest.pkl"


class DriftSimulator:
    """Generates normal and drifted data batches for testing the monitoring pipeline."""

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        training_stats: dict,
        config: dict,
        model: object = None,
    ) -> None:
        self._logger = get_logger(__name__)
        self.X_train = X_train
        self.y_train = y_train
        self.training_stats = training_stats
        self.config = config
        self._rng = np.random.default_rng(42)
        self._detector = DriftDetector(training_stats, config)

        if model is not None:
            self.model = model
        else:
            self.model = joblib.load(_MODEL_PATH)

        sim_cfg = config.get("drift", {}).get("simulation", {})
        self.shift_intensity: float = sim_cfg.get("shift_intensity", 1.0)
        self.concept_drift_noise: float = sim_cfg.get("concept_drift_noise", 0.3)

        self._logger.info(
            "DriftSimulator initialised — %d training samples, %d features",
            len(X_train),
            X_train.shape[1],
        )

    def simulate_data_drift(
        self,
        n_normal: int = 5000,
        n_drifted: int = 5000,
        shift_intensity: float = 1.0,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Simulate DATA drift — feature distributions shift gradually.

        Normal batch: resampled from training distribution.
        Drifted batch: 5 sub-batches of n_drifted//5 samples each, with
        shifts of +0.2, +0.4, +0.6, +0.8, +1.0 * std per feature.
        """
        # Normal batch
        idx_normal = self._rng.integers(0, len(self.X_train), size=n_normal)
        normal_batch = self.X_train.iloc[idx_normal].reset_index(drop=True)

        # Drifted batch: 5 sub-batches with increasing shift
        sub_size = n_drifted // 5
        shift_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
        drifted_parts = []
        for shift in shift_levels:
            idx = self._rng.integers(0, len(self.X_train), size=sub_size)
            sub = self.X_train.iloc[idx].copy().reset_index(drop=True)
            for feature, stats in self.training_stats.items():
                if feature in sub.columns:
                    sub[feature] = sub[feature] + shift * shift_intensity * stats["std"]
            drifted_parts.append(sub)
        drifted_batch = pd.concat(drifted_parts, ignore_index=True)

        self._logger.info(
            "simulate_data_drift: normal=%d, drifted=%d",
            len(normal_batch),
            len(drifted_batch),
        )
        return normal_batch, drifted_batch

    def simulate_concept_drift(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        noise_prob: float = 0.3,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Simulate CONCEPT drift — features unchanged, labels flipped.

        KEY DISTINCTION: features are NOT shifted. Only the feature-label
        relationship is corrupted by randomly flipping noise_prob of labels.
        """
        X_out = X.copy()
        y_arr = y.values.copy().astype(int)
        flip_mask = self._rng.random(len(y_arr)) < noise_prob
        y_arr[flip_mask] = 1 - y_arr[flip_mask]
        y_out = pd.Series(y_arr, index=y.index)
        self._logger.info(
            "simulate_concept_drift: flipped %d/%d labels (%.1f%%)",
            int(flip_mask.sum()),
            len(y_arr),
            100.0 * flip_mask.mean(),
        )
        return X_out, y_out

    def run_full_simulation(self) -> list[dict]:
        """Run 150-batch simulation: 50 normal, 50 data drift, 50 concept drift.

        Per-batch (100 samples): run predictions, compute max PSI vs training
        distribution, compute accuracy vs true labels.

        Returns list of dicts with keys: batch_id, psi_max, accuracy, drift_type.
        """
        batch_size = 100
        results: list[dict] = []

        # Baseline accuracy on training set (sampled for speed)
        n_base = min(2000, len(self.X_train))
        idx_base = self._rng.integers(0, len(self.X_train), size=n_base)
        X_base = self.X_train.iloc[idx_base]
        y_base = self.y_train.iloc[idx_base]
        y_pred_base = self.model.predict(X_base)  # type: ignore[union-attr]
        baseline_accuracy = float(np.mean(y_pred_base == y_base.values))
        self._logger.info("Baseline accuracy: %.4f", baseline_accuracy)

        for batch_id in range(1, 151):
            idx = self._rng.integers(0, len(self.X_train), size=batch_size)
            X_batch = self.X_train.iloc[idx].reset_index(drop=True)
            y_batch = self.y_train.iloc[idx].reset_index(drop=True)

            if batch_id <= 50:
                drift_type = "none"
                X_eval = X_batch
                y_eval = y_batch

            elif batch_id <= 100:
                # Gradual data drift: 10 batches per shift level
                sub_idx = batch_id - 51  # 0..49
                shift_level = (sub_idx // 10 + 1) * 0.2  # 0.2..1.0
                X_eval = X_batch.copy()
                for feature, stats in self.training_stats.items():
                    if feature in X_eval.columns:
                        X_eval[feature] = (
                            X_eval[feature]
                            + shift_level * self.shift_intensity * stats["std"]
                        )
                y_eval = y_batch
                drift_type = "data"

            else:
                # Concept drift: features unchanged, labels flipped
                X_eval, y_eval = self.simulate_concept_drift(
                    X_batch, y_batch, noise_prob=self.concept_drift_noise
                )
                drift_type = "concept"

            # Accuracy
            y_pred = self.model.predict(X_eval)  # type: ignore[union-attr]
            accuracy = float(np.mean(y_pred == y_eval.values))

            # Max PSI across features vs training distribution
            psi_values: dict[str, float] = {}
            for feature in X_eval.columns:
                if feature not in self.training_stats:
                    continue
                stats = self.training_stats[feature]
                expected = self._rng.normal(
                    loc=stats["mean"], scale=stats["std"], size=500
                )
                actual = X_eval[feature].values
                psi = self._detector.compute_psi(expected, actual)
                psi_values[feature] = psi

            max_psi = max(psi_values.values()) if psi_values else 0.0

            results.append(
                {
                    "batch_id": batch_id,
                    "psi_max": max_psi,
                    "accuracy": accuracy,
                    "drift_type": drift_type,
                }
            )

        self._logger.info("run_full_simulation complete: %d batches", len(results))
        return results


def plot_accuracy_collapse(simulation_results: list[dict], output_path: str) -> None:
    """THE MONEY SHOT — accuracy collapse under drift.

    Color bands: green (normal), orange (data drift), red (concept drift).
    Vertical dashed lines at batch 50 and 100 marking drift onset.
    """
    batch_ids = [r["batch_id"] for r in simulation_results]
    accuracies = [r["accuracy"] for r in simulation_results]
    baseline = np.mean(accuracies[:50]) if len(accuracies) >= 50 else accuracies[0]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Color band backgrounds
    ax.axvspan(1, 50, alpha=0.08, color="green", label="_nolegend_")
    ax.axvspan(50, 100, alpha=0.08, color="orange", label="_nolegend_")
    ax.axvspan(100, 150, alpha=0.08, color="red", label="_nolegend_")

    # Baseline accuracy
    ax.axhline(
        baseline,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label=f"Baseline accuracy ({baseline:.2f})",
    )

    # Drift onset lines
    ax.axvline(
        50,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label="Data drift onset (batch 50)",
    )
    ax.axvline(
        100,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Concept drift onset (batch 100)",
    )

    # Accuracy curve
    ax.plot(batch_ids, accuracies, color="#1f77b4", linewidth=2, zorder=3)

    # Legend patches for bands
    green_patch = mpatches.Patch(color="green", alpha=0.3, label="Normal operation")
    orange_patch = mpatches.Patch(color="orange", alpha=0.3, label="Data drift")
    red_patch = mpatches.Patch(color="red", alpha=0.3, label="Concept drift")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles + [green_patch, orange_patch, red_patch],
        loc="lower left",
        fontsize=9,
    )

    ax.set_xlabel("Batch Number", fontsize=12)
    ax.set_ylabel("Model Accuracy", fontsize=12)
    ax.set_title(
        "Model Accuracy Collapse Under Drift\n"
        "Without monitoring, you won't know this is happening",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlim(1, 150)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved accuracy_collapse plot to %s", output_path)


def plot_psi_timeline(simulation_results: list[dict], output_path: str) -> None:
    """PSI over time — shows PSI rising during data drift but NOT during concept drift.

    Horizontal red dashed line at PSI=0.2 (alert threshold).
    """
    batch_ids = [r["batch_id"] for r in simulation_results]
    psi_values = [r["psi_max"] for r in simulation_results]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.axvspan(1, 50, alpha=0.08, color="green")
    ax.axvspan(50, 100, alpha=0.08, color="orange")
    ax.axvspan(100, 150, alpha=0.08, color="red")

    ax.axhline(
        0.2,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="PSI alert threshold (0.2)",
    )
    ax.axvline(
        50, color="orange", linestyle="--", linewidth=1.2, label="Data drift onset"
    )
    ax.axvline(
        100, color="red", linestyle="--", linewidth=1.2, label="Concept drift onset"
    )

    ax.plot(batch_ids, psi_values, color="#d62728", linewidth=2, label="Max PSI")

    ax.annotate(
        "PSI rises\n(data drift)",
        xy=(75, max(psi_values[50:100]) if len(psi_values) > 100 else 0.3),
        xytext=(65, max(psi_values[50:100]) * 1.2 if len(psi_values) > 100 else 0.4),
        fontsize=9,
        color="orange",
        arrowprops=dict(arrowstyle="->", color="orange"),
    )
    ax.annotate(
        "PSI stable\n(concept drift invisible!)",
        xy=(125, np.mean(psi_values[100:]) if len(psi_values) >= 150 else 0.1),
        xytext=(110, 0.35),
        fontsize=9,
        color="darkred",
        arrowprops=dict(arrowstyle="->", color="darkred"),
    )

    ax.set_xlabel("Batch Number", fontsize=12)
    ax.set_ylabel("Max PSI across features", fontsize=12)
    ax.set_title(
        "PSI Timeline — Data Drift Detected, Concept Drift Invisible to PSI",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlim(1, 150)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved psi_timeline plot to %s", output_path)


def plot_drift_flag_timeline(simulation_results: list[dict], output_path: str) -> None:
    """Binary flag chart: data drift / concept drift / PSI alert fired per batch."""
    psi_threshold = 0.2
    batch_ids = [r["batch_id"] for r in simulation_results]
    data_drift_flag = [
        1 if r["drift_type"] == "data" else 0 for r in simulation_results
    ]
    concept_drift_flag = [
        1 if r["drift_type"] == "concept" else 0 for r in simulation_results
    ]
    alert_flag = [1 if r["psi_max"] > psi_threshold else 0 for r in simulation_results]

    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    rows = [
        (data_drift_flag, "Data Drift Active", "orange"),
        (concept_drift_flag, "Concept Drift Active", "red"),
        (alert_flag, "PSI Alert Fired", "purple"),
    ]

    for ax, (flags, label, color) in zip(axes, rows):
        ax.fill_between(batch_ids, flags, step="post", alpha=0.6, color=color)
        ax.set_ylabel(label, fontsize=9)
        ax.set_ylim(-0.1, 1.4)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["No", "Yes"], fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        ax.axvline(50, color="orange", linestyle="--", linewidth=0.8)
        ax.axvline(100, color="red", linestyle="--", linewidth=0.8)

    axes[-1].set_xlabel("Batch Number", fontsize=12)
    fig.suptitle("Drift Detection Flag Timeline", fontsize=13, fontweight="bold")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved drift_flag_timeline plot to %s", output_path)
