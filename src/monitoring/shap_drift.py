"""SHAP-based feature importance comparison under drift (B2 tie-in)."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.logger import get_logger

plt.switch_backend("Agg")

logger = get_logger(__name__)


def compare_shap_under_drift(
    model: object,
    X_train: pd.DataFrame,
    X_drifted: pd.DataFrame,
    shap_baseline: dict,
    config: dict,
) -> dict:
    """Compare SHAP feature importances: training baseline vs drifted data.

    Loads baseline importances from shap_baseline dict (computed on Day 1).
    Computes fresh SHAP values on X_drifted via TreeExplainer.
    Returns top-5 features for each, plus the biggest importance shifts.

    B2 taught SHAP; B5 reuses it here to answer *what changed* after drift.
    """

    shap_cfg = config.get("shap", {})
    n_bg = min(int(shap_cfg.get("n_background_samples", 100)), len(X_train))
    n_drift = min(int(shap_cfg.get("n_drift_samples", 200)), len(X_drifted))

    # Background dataset for TreeExplainer
    bg = X_train.sample(n_bg, random_state=42)

    explainer = shap.TreeExplainer(model, bg)

    # SHAP values on drifted sample
    X_sample = X_drifted.sample(n_drift, random_state=42)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification shap_values is a list [class_0, class_1]
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # Mean absolute SHAP per feature on drifted data
    drifted_importance: dict[str, float] = {
        feat: float(np.abs(sv[:, i]).mean()) for i, feat in enumerate(X_drifted.columns)
    }

    # Baseline importances
    baseline_importance: dict[str, float] = {
        k: float(v) for k, v in shap_baseline.items()
    }

    # Top-5 lists
    baseline_top_5 = sorted(baseline_importance.items(), key=lambda x: -x[1])[:5]
    drifted_top_5 = sorted(drifted_importance.items(), key=lambda x: -x[1])[:5]

    # Importance shift per feature (drifted - baseline)
    common_features = set(baseline_importance) & set(drifted_importance)
    shifts: dict[str, float] = {
        f: drifted_importance[f] - baseline_importance[f] for f in common_features
    }
    biggest_shifts = sorted(shifts.items(), key=lambda x: -abs(x[1]))[:5]

    # Human-readable interpretation
    if biggest_shifts:
        top_feat, top_shift = biggest_shifts[0]
        b_rank_list = [f for f, _ in baseline_top_5]
        d_rank_list = [f for f, _ in drifted_top_5]
        b_rank: str | int = (
            b_rank_list.index(top_feat) + 1
            if top_feat in b_rank_list
            else "outside top 5"
        )
        d_rank: str | int = (
            d_rank_list.index(top_feat) + 1
            if top_feat in d_rank_list
            else "outside top 5"
        )
        interpretation = (
            f"Feature {top_feat} rose from rank {b_rank} to rank {d_rank} "
            f"after drift — primarily affected {top_feat} "
            f"(shift: {top_shift:+.4f})"
        )
    else:
        interpretation = "No significant shift detected in feature importances."

    logger.info("SHAP drift comparison complete — top shift: %s", biggest_shifts[:1])
    return {
        "baseline_top_5": baseline_top_5,
        "drifted_top_5": drifted_top_5,
        "biggest_shifts": biggest_shifts,
        "interpretation": interpretation,
    }


def plot_shap_comparison(comparison: dict, output_path: str) -> None:
    """Side-by-side bar chart: baseline (blue) vs drifted (red) SHAP importances.

    Features sorted by absolute importance shift — largest shift at top.
    """
    biggest_shifts = comparison["biggest_shifts"]
    if not biggest_shifts:
        logger.warning("No shift data to plot — skipping shap comparison chart")
        return

    features = [f for f, _ in biggest_shifts]
    baseline_dict = dict(comparison["baseline_top_5"])
    drifted_dict = dict(comparison["drifted_top_5"])

    baseline_vals = [baseline_dict.get(f, 0.0) for f in features]
    drifted_vals = [drifted_dict.get(f, 0.0) for f in features]

    x = np.arange(len(features))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(
        x - width / 2,
        baseline_vals,
        width,
        label="Training (baseline)",
        color="#1f77b4",
        alpha=0.85,
    )
    ax.barh(
        x + width / 2,
        drifted_vals,
        width,
        label="Drifted data",
        color="#d62728",
        alpha=0.85,
    )

    ax.set_yticks(x)
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(
        "SHAP Feature Importance: Baseline vs Drifted Data",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved shap_drift_comparison plot to %s", output_path)
