"""Run the full drift simulation and save all portfolio plots.

Usage (from project root):
    python scripts/run_simulation.py
"""

import json
import os
import sys

import pandas as pd

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.monitoring.drift_simulator import (
    DriftSimulator,
    plot_accuracy_collapse,
    plot_drift_flag_timeline,
    plot_psi_timeline,
)
from src.monitoring.shap_drift import compare_shap_under_drift, plot_shap_comparison

TRAIN_PATH = "data/processed/train.parquet"
STATS_PATH = "models/training_stats.json"
SHAP_PATH = "models/shap_baseline.json"
FIGURES_DIR = "reports/figures"


def main() -> None:
    print("=== B5 Drift Monitor — Full Simulation ===\n")

    # 1. Load training data
    print("[1/5] Loading training data...")
    df = pd.read_parquet(TRAIN_PATH)
    target_col = "target"
    feature_cols = [c for c in df.columns if c != target_col]
    X_train = df[feature_cols]
    y_train = df[target_col]
    print(f"      {len(X_train)} rows, {len(feature_cols)} features")

    # 2. Load supporting files
    print("[2/5] Loading training stats and SHAP baseline...")
    with open(STATS_PATH) as fh:
        training_stats = json.load(fh)
    with open(SHAP_PATH) as fh:
        shap_baseline = json.load(fh)

    config = {
        "drift": {
            "psi_threshold": 0.2,
            "ks_alpha": 0.05,
            "evidently_report_ttl": 60,
            "simulation": {
                "shift_intensity": 1.0,
                "concept_drift_noise": 0.3,
            },
        },
        "shap": {
            "n_background_samples": 100,
            "n_drift_samples": 200,
        },
    }

    # 3. Run full 150-batch simulation
    print("[3/5] Running full simulation (150 batches)...")
    sim = DriftSimulator(X_train, y_train, training_stats, config)
    results = sim.run_full_simulation()

    normal_acc = sum(r["accuracy"] for r in results if r["drift_type"] == "none") / 50
    data_acc = sum(r["accuracy"] for r in results if r["drift_type"] == "data") / 50
    concept_acc = sum(r["accuracy"] for r in results if r["drift_type"] == "concept") / 50
    print(f"      Normal accuracy:       {normal_acc:.3f}")
    print(f"      Data-drift accuracy:   {data_acc:.3f}")
    print(f"      Concept-drift accuracy:{concept_acc:.3f}")

    # 4. Save portfolio plots
    print("[4/5] Saving portfolio plots...")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    acc_path = os.path.join(FIGURES_DIR, "accuracy_collapse.png")
    psi_path = os.path.join(FIGURES_DIR, "psi_timeline.png")
    flag_path = os.path.join(FIGURES_DIR, "drift_timeline.png")

    plot_accuracy_collapse(results, acc_path)
    print(f"      Saved: {acc_path}")

    plot_psi_timeline(results, psi_path)
    print(f"      Saved: {psi_path}")

    plot_drift_flag_timeline(results, flag_path)
    print(f"      Saved: {flag_path}")

    # 5. SHAP comparison on drifted batch
    print("[5/5] Running SHAP comparison (drifted vs baseline)...")
    _, drifted_batch = sim.simulate_data_drift(n_normal=100, n_drifted=500, shift_intensity=1.0)
    comparison = compare_shap_under_drift(
        sim.model, X_train, drifted_batch, shap_baseline, config
    )

    shap_path = os.path.join(FIGURES_DIR, "shap_drift_comparison.png")
    plot_shap_comparison(comparison, shap_path)
    print(f"      Saved: {shap_path}")

    print("\n--- SHAP interpretation ---")
    print(f"  {comparison['interpretation']}")
    print(f"\n  Baseline top-5: {[f for f, _ in comparison['baseline_top_5']]}")
    print(f"  Drifted  top-5: {[f for f, _ in comparison['drifted_top_5']]}")

    print("\n=== Done. All plots saved to reports/figures/ ===")


if __name__ == "__main__":
    main()
