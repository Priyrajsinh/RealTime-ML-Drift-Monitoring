"""B5 Drift Monitor — Hugging Face Space (100% self-contained, zero src/ imports).

Tab 1: Live Drift Monitor  (recruiter-facing UX)
Tab 2: Analysis — Developer
Tab 3: How It Works
"""

import json
import os
import tempfile

import gradio as gr
import joblib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import ks_2samp

plt.switch_backend("Agg")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PSI_THRESHOLD = 0.2
KS_ALPHA = 0.05
FEATURES = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]

# ---------------------------------------------------------------------------
# Model + stats loading
# ---------------------------------------------------------------------------
_BASE = os.path.dirname(os.path.abspath(__file__))


def _load_assets():
    model = joblib.load(os.path.join(_BASE, "random_forest.pkl"))
    with open(os.path.join(_BASE, "training_stats.json"), "r", encoding="utf-8") as fh:
        training_stats = json.load(fh)
    with open(os.path.join(_BASE, "shap_baseline.json"), "r", encoding="utf-8") as fh:
        shap_baseline = json.load(fh)
    return model, training_stats, shap_baseline


try:
    MODEL, TRAINING_STATS, SHAP_BASELINE = _load_assets()
    _ASSETS_LOADED = True
except Exception as _load_err:
    MODEL, TRAINING_STATS, SHAP_BASELINE = None, {}, {}
    _ASSETS_LOADED = False
    print(f"[WARN] Asset load failed: {_load_err}")

# ---------------------------------------------------------------------------
# PSI computation (inlined — no src/ dependency)
# ---------------------------------------------------------------------------


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    eps = 1e-6
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    if min_val == max_val:
        return 0.0
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    expected_counts = np.histogram(expected, bins=bin_edges)[0]
    actual_counts = np.histogram(actual, bins=bin_edges)[0]
    expected_pct = expected_counts / len(expected) + eps
    actual_pct = actual_counts / len(actual) + eps
    return float(
        np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    )


def psi_label(psi: float) -> tuple[str, str]:
    """Return (text_label, color_hex) for a PSI value."""
    if psi < 0.1:
        return "Normal", "#27ae60"
    elif psi < 0.2:
        return "Moderate", "#f39c12"
    else:
        return "DRIFT DETECTED", "#e74c3c"


# ---------------------------------------------------------------------------
# Synthetic data generation (inlined — no src/ dependency)
# ---------------------------------------------------------------------------


def _make_training_data(
    stats: dict, n: int = 2000, rng: np.random.Generator = None
) -> pd.DataFrame:
    if rng is None:
        rng = np.random.default_rng(42)
    rows = {}
    for feat, s in stats.items():
        rows[feat] = rng.normal(loc=s["mean"], scale=max(s["std"], 1e-6), size=n)
    return pd.DataFrame(rows)


def _apply_drift(
    df: pd.DataFrame, stats: dict, shift_intensity: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Shift all feature means by shift_intensity * std."""
    out = df.copy()
    for feat, s in stats.items():
        if feat in out.columns:
            out[feat] = out[feat] + shift_intensity * s["std"]
    return out


# ---------------------------------------------------------------------------
# Core simulation (inlined)
# ---------------------------------------------------------------------------

_SIM_CACHE: dict[float, dict] = {}


def run_simulation(shift_intensity: float) -> dict:
    """Run 150-batch drift simulation. Cached per intensity (rounded to 1dp)."""
    key = round(float(shift_intensity), 1)
    if key in _SIM_CACHE:
        return _SIM_CACHE[key]

    if not _ASSETS_LOADED:
        return {"error": "Model assets not loaded."}

    rng = np.random.default_rng(42)
    batch_size = 100
    results = []
    psi_per_feature_drifted: dict[str, float] = {}

    # Collect batches for Evidently reference vs current
    X_normal_batches: list[pd.DataFrame] = []
    X_drifted_batches: list[pd.DataFrame] = []

    for batch_id in range(1, 151):
        X_batch = _make_training_data(TRAINING_STATS, n=batch_size, rng=rng)
        X_batch = X_batch[FEATURES]

        if batch_id <= 50:
            drift_type = "none"
            X_eval = X_batch
            X_normal_batches.append(X_batch)

        elif batch_id <= 100:
            drift_type = "data"
            sub_idx = batch_id - 51
            level = (sub_idx // 10 + 1) * 0.2
            X_eval = _apply_drift(X_batch, TRAINING_STATS, level * shift_intensity, rng)
            X_drifted_batches.append(X_eval)

        else:
            drift_type = "concept"
            X_eval = X_batch  # features unchanged

        y_pred = MODEL.predict(X_eval)
        if drift_type == "concept":
            # Simulate concept drift: flip ~30% of labels
            y_true = (MODEL.predict_proba(X_batch)[:, 1] > 0.5).astype(int)
            flip = rng.random(len(y_true)) < 0.3
            y_true[flip] = 1 - y_true[flip]
            accuracy = float(np.mean(y_pred == y_true))
        else:
            y_true = (MODEL.predict_proba(X_batch)[:, 1] > 0.5).astype(int)
            accuracy = float(np.mean(y_pred == y_true))

        # PSI per feature
        psi_values: dict[str, float] = {}
        ks_results: dict[str, tuple[float, float]] = {}
        for feat in FEATURES:
            if feat not in TRAINING_STATS:
                continue
            s = TRAINING_STATS[feat]
            expected = rng.normal(loc=s["mean"], scale=max(s["std"], 1e-6), size=500)
            actual = X_eval[feat].values
            psi = compute_psi(expected, actual)
            psi_values[feat] = psi
            stat, pval = ks_2samp(expected, actual)
            ks_results[feat] = (float(stat), float(pval))

        max_psi = max(psi_values.values()) if psi_values else 0.0

        # Accumulate per-feature PSI during data drift phase for leaderboard
        if drift_type == "data":
            for f, v in psi_values.items():
                psi_per_feature_drifted[f] = max(psi_per_feature_drifted.get(f, 0.0), v)

        results.append(
            {
                "batch_id": batch_id,
                "psi_max": max_psi,
                "accuracy": accuracy,
                "drift_type": drift_type,
                "psi_values": psi_values,
                "ks_results": ks_results,
            }
        )

    # Summary statistics
    acc_normal = np.mean([r["accuracy"] for r in results if r["drift_type"] == "none"])
    acc_concept = np.mean(
        [r["accuracy"] for r in results if r["drift_type"] == "concept"]
    )
    max_psi_all = max(r["psi_max"] for r in results)
    drift_start_batch = next(
        (r["batch_id"] for r in results if r["psi_max"] > PSI_THRESHOLD), None
    )
    top_drifted = sorted(psi_per_feature_drifted.items(), key=lambda x: -x[1])[:5]

    out = {
        "results": results,
        "baseline_accuracy": float(acc_normal),
        "concept_accuracy": float(acc_concept),
        "max_psi": max_psi_all,
        "drift_start_batch": drift_start_batch,
        "top_drifted_features": top_drifted,
        "psi_per_feature": psi_per_feature_drifted,
        "last_ks": results[-51]["ks_results"] if len(results) >= 51 else {},
        "X_reference": pd.concat(X_normal_batches, ignore_index=True),
        "X_current": pd.concat(X_drifted_batches, ignore_index=True),
    }
    _SIM_CACHE[key] = out
    return out


# ---------------------------------------------------------------------------
# Plot generators
# ---------------------------------------------------------------------------


def plot_accuracy_collapse(results: list[dict]) -> plt.Figure:
    batch_ids = [r["batch_id"] for r in results]
    accuracies = [r["accuracy"] for r in results]
    baseline = float(np.mean(accuracies[:50])) if len(accuracies) >= 50 else 0.8

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axvspan(1, 50, alpha=0.08, color="green")
    ax.axvspan(50, 100, alpha=0.08, color="orange")
    ax.axvspan(100, 150, alpha=0.08, color="red")
    ax.axhline(
        baseline,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label=f"Baseline accuracy ({baseline:.2f})",
    )
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
    ax.plot(batch_ids, accuracies, color="#1f77b4", linewidth=2.2, zorder=3)

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
    fig.tight_layout()
    return fig


def plot_psi_timeline(results: list[dict]) -> plt.Figure:
    batch_ids = [r["batch_id"] for r in results]
    psi_values = [r["psi_max"] for r in results]

    fig, ax = plt.subplots(figsize=(11, 4))
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
    ax.set_xlabel("Batch Number", fontsize=12)
    ax.set_ylabel("Max PSI", fontsize=12)
    ax.set_title(
        "PSI Timeline — Data Drift Detected, Concept Drift Invisible to PSI",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlim(1, 150)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_feature_leaderboard(psi_dict: dict) -> plt.Figure:
    if not psi_dict:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No drift data", ha="center", va="center")
        return fig

    sorted_feats = sorted(psi_dict.items(), key=lambda x: -x[1])[:15]
    feats = [f for f, _ in sorted_feats]
    vals = [v for _, v in sorted_feats]
    colors = [
        "#e74c3c" if v > 0.2 else "#f39c12" if v > 0.1 else "#27ae60" for v in vals
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(feats))
    ax.barh(y, vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(
        0.2, color="red", linestyle="--", linewidth=1.2, label="Alert threshold (0.2)"
    )
    ax.axvline(
        0.1,
        color="orange",
        linestyle="--",
        linewidth=1.0,
        label="Moderate threshold (0.1)",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(feats, fontsize=9)
    ax.set_xlabel("Max PSI during drift phase", fontsize=11)
    ax.set_title(
        "Feature Drift Leaderboard (sorted by PSI)", fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_shap_comparison(shap_baseline: dict, drifted_importances: dict) -> plt.Figure:
    common = set(shap_baseline) & set(drifted_importances)
    shifts = {f: abs(drifted_importances[f] - shap_baseline[f]) for f in common}
    top_feats = [f for f, _ in sorted(shifts.items(), key=lambda x: -x[1])[:8]]

    baseline_vals = [shap_baseline.get(f, 0.0) for f in top_feats]
    drifted_vals = [drifted_importances.get(f, 0.0) for f in top_feats]
    x = np.arange(len(top_feats))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
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
    ax.set_yticklabels(top_feats, fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(
        "SHAP Feature Importance: Baseline vs Drifted", fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# SHAP on drifted data (inlined)
# ---------------------------------------------------------------------------


def compute_shap_drifted(model, stats: dict, shift_intensity: float) -> dict:
    rng = np.random.default_rng(42)
    X_bg = _make_training_data(stats, n=100, rng=rng)
    X_bg = X_bg[FEATURES]
    X_drift = _apply_drift(X_bg, stats, shift_intensity, rng)

    explainer = shap.TreeExplainer(model, X_bg)
    sv = explainer.shap_values(X_drift)
    if isinstance(sv, list):
        sv = sv[1]

    return {feat: float(np.abs(sv[:, i]).mean()) for i, feat in enumerate(FEATURES)}


# ---------------------------------------------------------------------------
# Evidently report generation (inlined — uses evidently package)
# ---------------------------------------------------------------------------


def generate_evidently_html(
    reference: pd.DataFrame, current: pd.DataFrame, intensity: float
) -> str:
    """Generate Evidently DataDrift HTML report. Returns temp file path."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=reference, current_data=current)
    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".html",
        prefix=f"evidently_drift_intensity{intensity}_",
    )
    snapshot.save_html(tmp.name)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Main simulation handler
# ---------------------------------------------------------------------------


def simulate(shift_intensity: float):
    """Gradio handler — returns all outputs for Tab 1 + Tab 2."""
    data = run_simulation(shift_intensity)

    if "error" in data:
        err = data["error"]
        err_div = (
            f"<div style='background:#e74c3c;color:white;padding:12px;"
            f"border-radius:8px;text-align:center;font-weight:bold'>{err}</div>"
        )
        return (
            f"<div style='color:red;font-size:18px'>{err}</div>",
            "<div style='color:red;font-size:24px'>N/A</div>",
            "<div style='color:red;font-size:24px'>N/A</div>",
            err_div,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    results = data["results"]
    baseline_acc = data["baseline_accuracy"]
    concept_acc = data["concept_accuracy"]
    max_psi = data["max_psi"]
    drift_start = data["drift_start_batch"]
    top_drifted = data["top_drifted_features"]
    psi_per_feat = data["psi_per_feature"]

    # --- PSI gauge HTML ---
    psi_text, psi_color = psi_label(max_psi)
    psi_gauge_html = f"""
<div style="text-align:center;padding:16px;background:#1a1a2e;border-radius:12px">
  <div style="font-size:14px;color:#aaa;margin-bottom:4px">Peak Data Drift (PSI)</div>
  <div style="font-size:52px;font-weight:bold;color:{psi_color}">{max_psi:.3f}</div>
  <div style="font-size:20px;color:{psi_color};margin-top:4px">{psi_text}</div>
  <div style="font-size:11px;color:#888;margin-top:8px">
    &lt;0.1 Normal &nbsp;|&nbsp; 0.1–0.2 Moderate &nbsp;|&nbsp; &gt;0.2 Alert
  </div>
</div>"""

    # --- Accuracy meter HTML ---
    acc_delta = baseline_acc - concept_acc
    acc_color = (
        "#27ae60" if acc_delta < 0.05 else "#f39c12" if acc_delta < 0.15 else "#e74c3c"
    )
    acc_label = (
        "STABLE"
        if acc_delta < 0.05
        else "DEGRADED" if acc_delta < 0.15 else "COLLAPSED"
    )
    acc_html = f"""
<div style="text-align:center;padding:16px;background:#1a1a2e;border-radius:12px">
  <div style="font-size:14px;color:#aaa;margin-bottom:4px">Model Accuracy</div>
  <div style="font-size:36px;font-weight:bold;color:#27ae60">{baseline_acc:.0%}</div>
  <div style="font-size:24px;color:#aaa;margin:4px 0">↓</div>
  <div style="font-size:36px;font-weight:bold;color:{acc_color}">{concept_acc:.0%}</div>
  <div style="font-size:18px;color:{acc_color};margin-top:4px">{acc_label}</div>
</div>"""

    # --- Alert banner ---
    _alert_style_red = (
        "background:#e74c3c;color:white;padding:16px;border-radius:8px;"
        "text-align:center;font-size:22px;font-weight:bold;letter-spacing:1px"
    )
    _alert_style_green = (
        "background:#27ae60;color:white;padding:16px;border-radius:8px;"
        "text-align:center;font-size:22px;font-weight:bold"
    )
    if max_psi > PSI_THRESHOLD:
        _msg = f"DRIFT DETECTED — PSI={max_psi:.3f} at batch {drift_start}"
        alert_html = f'<div style="{_alert_style_red}">{_msg}</div>'
    else:
        alert_html = f'<div style="{_alert_style_green}">No significant drift</div>'

    # --- Summary text ---
    top_feats_str = (
        ", ".join(f"{f} (PSI={v:.2f})" for f, v in top_drifted[:3])
        if top_drifted
        else "N/A"
    )
    silent_batches = 150 - (drift_start or 150)
    drift_at = drift_start if drift_start else "N/A"
    summary = (
        f"**Simulation complete** — 150 batches "
        f"(50 normal \u2192 50 data drift \u2192 50 concept drift)\n\n"
        f"- **Drift detected at:** batch {drift_at} "
        f"(PSI threshold {PSI_THRESHOLD})\n"
        f"- **Accuracy drop:** {baseline_acc:.0%} \u2192 {concept_acc:.0%} "
        f"({acc_delta:.0%} degradation)\n"
        f"- **Peak PSI:** {max_psi:.3f} ({psi_text})\n"
        f"- **Top drifted features:** {top_feats_str}\n\n"
        f"Without monitoring, this model would silently serve bad predictions "
        f"for **{silent_batches} batches** before anyone noticed."
    )

    # --- Plots ---
    fig_accuracy = plot_accuracy_collapse(results)
    fig_psi = plot_psi_timeline(results)

    # --- Tab 2: feature leaderboard dataframe ---
    if psi_per_feat:
        leaderboard_rows = []
        last_ks = data.get("last_ks", {})
        for feat, psi_val in sorted(psi_per_feat.items(), key=lambda x: -x[1]):
            _, ks_pval = last_ks.get(feat, (0.0, 1.0))
            psi_t, _ = psi_label(psi_val)
            leaderboard_rows.append(
                {
                    "Feature": feat,
                    "Max PSI": round(psi_val, 4),
                    "Status": psi_t,
                    "KS p-value": round(ks_pval, 4),
                    "KS Drift": "Yes" if ks_pval < KS_ALPHA else "No",
                }
            )
        leaderboard_df = pd.DataFrame(leaderboard_rows)
    else:
        leaderboard_df = pd.DataFrame(
            columns=["Feature", "Max PSI", "Status", "KS p-value", "KS Drift"]
        )

    # --- Tab 2: SHAP comparison ---
    if _ASSETS_LOADED:
        drifted_shap = compute_shap_drifted(MODEL, TRAINING_STATS, shift_intensity)
        fig_shap = plot_shap_comparison(SHAP_BASELINE, drifted_shap)
    else:
        fig_shap = None

    fig_leaderboard = plot_feature_leaderboard(psi_per_feat)

    # --- CSV download (leaderboard) ---
    csv_tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".csv", prefix="drift_results_"
    )
    leaderboard_df.to_csv(csv_tmp.name, index=False)
    csv_tmp.close()

    # --- PNG download (accuracy collapse) ---
    png_tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".png", prefix="accuracy_collapse_"
    )
    fig_accuracy.savefig(png_tmp.name, dpi=150, bbox_inches="tight")
    png_tmp.close()

    # --- Evidently HTML report (generated from this user's simulation data) ---
    html_path = generate_evidently_html(
        reference=data["X_reference"],
        current=data["X_current"],
        intensity=shift_intensity,
    )

    return (
        psi_gauge_html,
        acc_html,
        alert_html,
        summary,
        fig_accuracy,
        fig_psi,
        leaderboard_df,
        fig_leaderboard,
        fig_shap,
        f"Simulated 150 batches \u00d7 {len(FEATURES)} features "
        f"| intensity: {shift_intensity}x",
        csv_tmp.name,
        png_tmp.name,
        html_path,
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

HOW_IT_WORKS_MD = """
## How It Works

### Architecture
```
┌─────────────────────────────────────────────────────────┐
│                   B5 Drift Monitor                       │
│                                                          │
│  Live Data ──► Feature Extractor ──► Random Forest       │
│       │                                    │             │
│       ▼                                    ▼             │
│  PSI Detector ──► Prometheus ──► Grafana Dashboard       │
│       │                │                  │             │
│       ▼                ▼                  ▼             │
│  KS Test     Alertmanager ──────────► Alert!             │
│       │                                    │             │
│       ▼                                    ▼             │
│  Evidently   ◄─────────────── SHAP Drift Comparison      │
│  HTML Report                                             │
└─────────────────────────────────────────────────────────┘
```

### PSI Formula
**Population Stability Index** measures distribution shift:

```
PSI = Σ (actual% − expected%) × ln(actual% / expected%)
```

| PSI Value   | Interpretation                       |
|-------------|--------------------------------------|
| < 0.1       | No significant shift — model is safe |
| 0.1 – 0.2   | Moderate shift — monitor closely     |
| > 0.2       | Significant shift — **retrain!**     |

### Data Drift vs Concept Drift

| Type     | What Changes          | Method       | Example              |
|----------|-----------------------|--------------|----------------------|
| Data     | Feature distributions | PSI, KS test | Credit limit shifts  |
| Concept  | Label relationship    | Accuracy drop| Same data, new rates |

**Why you need BOTH:**
- PSI catches data drift (distributions shift) — but is BLIND to concept drift
- Accuracy monitoring catches concept drift — but requires labelled data
- Run both in production or you will miss failures

### Why SHAP Helps
SHAP (SHapley Additive exPlanations) answers **what changed** after drift:
- Baseline SHAP: which features drove decisions during training
- Drifted SHAP: which features now dominate
- Large shifts → those features carry the distribution change

### Full Stack (Local Docker)
```
4 services:
├── app       — FastAPI + Prometheus /metrics endpoint
├── prometheus — scrapes every 15s, stores time-series
├── grafana   — pre-provisioned PSI + accuracy dashboard
└── alertmanager — fires when PSI > 0.2
```

### Tech Stack
scikit-learn · FastAPI · Prometheus · Grafana · Alertmanager
Evidently · Streamlit · SHAP · pandera · Docker Compose · MLflow

### Portfolio
| Project | What It Teaches |
|---------|----------------|
| B1 | HuggingFace Fine-Tuning + Production FastAPI |
| B2 | XGBoost + SHAP Explainability Dashboard |
| B3 | PyTorch LSTM Time Series Forecasting |
| B4 | Semantic Search with FAISS + Hybrid Search |
| **B5** | **Real-Time ML Monitoring + Drift Detection** |

[GitHub Repository](https://github.com/Priyrajsinh)
"""


def build_ui():
    with gr.Blocks(
        theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
        title="B5 — ML Model Drift Monitor",
        css="""
        .gradio-container { max-width: 1100px; margin: 0 auto; }
        h1 { text-align: center; }
        .footer-note { text-align:center; color:#888; font-size:12px; }
        """,
    ) as demo:

        gr.Markdown("""
# ML Model Drift Monitor
### Watch a machine learning model break in real time as data shifts

This demo simulates **150 prediction batches**: the model starts healthy,
then data distributions shift, and finally the feature-label relationship
collapses — all without the model knowing anything changed.
        """)

        with gr.Tabs():

            # ── Tab 1 — Live Drift Monitor ─────────────────────────────────
            with gr.Tab("Live Drift Monitor"):
                with gr.Row():
                    with gr.Column(scale=3):
                        intensity_slider = gr.Slider(
                            minimum=0.0,
                            maximum=3.0,
                            value=1.0,
                            step=0.1,
                            label="How much should the data change?",
                            info="Slide right = more drift = model breaks faster",
                        )
                    with gr.Column(scale=1):
                        simulate_btn = gr.Button(
                            "Simulate Drift", variant="primary", size="lg"
                        )

                gr.Markdown("---")

                with gr.Row():
                    psi_gauge = gr.HTML(label="Data Drift (PSI)")
                    acc_meter = gr.HTML(label="Model Accuracy")

                alert_banner = gr.HTML()

                with gr.Row():
                    fig_accuracy_out = gr.Plot(
                        label="Accuracy Collapse (the money shot)"
                    )
                    fig_psi_out = gr.Plot(label="PSI Timeline")

                summary_out = gr.Markdown()
                footer_out = gr.Markdown(elem_classes=["footer-note"])

            # ── Tab 2 — Developer Analysis ──────────────────────────────────
            with gr.Tab("Analysis (Developer)"):
                gr.Markdown("### Feature-Level Drift Leaderboard")
                leaderboard_df_out = gr.Dataframe(
                    headers=["Feature", "Max PSI", "Status", "KS p-value", "KS Drift"],
                    label="Features ranked by PSI (highest = most drifted)",
                )
                with gr.Row():
                    fig_leaderboard_out = gr.Plot(label="PSI by Feature (bar chart)")
                    fig_shap_out = gr.Plot(label="SHAP Comparison: Baseline vs Drifted")
                gr.Markdown("### Downloads")
                gr.Markdown(
                    "All files are generated from **your** simulation run "
                    "and reflect your chosen drift intensity."
                )
                with gr.Row():
                    csv_download = gr.File(
                        label="Drift Results CSV",
                        file_types=[".csv"],
                    )
                    png_download = gr.File(
                        label="Accuracy Collapse PNG",
                        file_types=[".png"],
                    )
                html_download = gr.File(
                    label=(
                        "Evidently Drift Report (HTML) — "
                        "interactive, works offline in any browser"
                    ),
                    file_types=[".html"],
                )

            # ── Tab 3 — How It Works ────────────────────────────────────────
            with gr.Tab("How It Works"):
                gr.Markdown(HOW_IT_WORKS_MD)

        # Wire button
        outputs = [
            psi_gauge,
            acc_meter,
            alert_banner,
            summary_out,
            fig_accuracy_out,
            fig_psi_out,
            leaderboard_df_out,
            fig_leaderboard_out,
            fig_shap_out,
            footer_out,
            csv_download,
            png_download,
            html_download,
        ]
        simulate_btn.click(fn=simulate, inputs=[intensity_slider], outputs=outputs)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
