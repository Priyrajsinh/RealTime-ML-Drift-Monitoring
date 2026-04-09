"""Streamlit dashboard for B5 Drift Monitor — 3-tab layout."""

import json
import os
import sys
import time

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import yaml  # noqa: E402

plt.switch_backend("Agg")
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "config.yaml")
_STATS_PATH = os.path.join(_PROJECT_ROOT, "models", "training_stats.json")
_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "random_forest.pkl")
_SHAP_PATH = os.path.join(_PROJECT_ROOT, "models", "shap_baseline.json")
_REPORT_PATH = os.path.join(_PROJECT_ROOT, "reports", "drift", "latest_report.html")


def _load_config() -> dict:
    with open(_CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def _load_training_stats() -> dict:
    with open(_STATS_PATH) as fh:
        return json.load(fh)


def _load_model():
    import joblib

    return joblib.load(_MODEL_PATH)


def _load_shap_baseline() -> dict:
    with open(_SHAP_PATH) as fh:
        return json.load(fh)


def _psi_status(psi: float) -> tuple[str, str]:
    """Return (label, color) for a PSI value."""
    if psi < 0.1:
        return "No Drift", "green"
    elif psi < 0.2:
        return "Moderate Shift", "orange"
    return "DRIFT DETECTED", "red"


def _run_simulation(config: dict, training_stats: dict):
    """Run the 150-batch simulation, yielding results for live updates."""
    import joblib

    from src.monitoring.drift_detector import DriftDetector

    model = joblib.load(_MODEL_PATH)
    detector = DriftDetector(training_stats, config)

    train_path = os.path.join(_PROJECT_ROOT, "data", "processed", "train.parquet")
    df_train = pd.read_parquet(train_path)
    feature_cols = list(training_stats.keys())
    X_train = df_train[feature_cols]
    y_train = df_train["default payment next month"]

    rng = np.random.default_rng(42)
    batch_size = 100
    shift_intensity = config["drift"]["simulation"]["shift_intensity"]
    concept_noise = config["drift"]["simulation"]["concept_drift_noise"]

    n_base = min(2000, len(X_train))
    idx_base = rng.integers(0, len(X_train), size=n_base)
    y_pred_base = model.predict(X_train.iloc[idx_base])
    baseline_acc = float(np.mean(y_pred_base == y_train.iloc[idx_base].values))

    for batch_id in range(1, 151):
        idx = rng.integers(0, len(X_train), size=batch_size)
        X_batch = X_train.iloc[idx].reset_index(drop=True)
        y_batch = y_train.iloc[idx].reset_index(drop=True)

        if batch_id <= 50:
            drift_type = "none"
            X_eval = X_batch
            y_eval = y_batch
        elif batch_id <= 100:
            sub_idx = batch_id - 51
            shift_level = (sub_idx // 10 + 1) * 0.2
            X_eval = X_batch.copy()
            for feature, stats in training_stats.items():
                if feature in X_eval.columns:
                    X_eval[feature] = (
                        X_eval[feature] + shift_level * shift_intensity * stats["std"]
                    )
            y_eval = y_batch
            drift_type = "data"
        else:
            X_eval = X_batch.copy()
            y_arr = y_batch.values.copy().astype(int)
            flip_mask = rng.random(len(y_arr)) < concept_noise
            y_arr[flip_mask] = 1 - y_arr[flip_mask]
            y_eval = pd.Series(y_arr)
            drift_type = "concept"

        y_pred = model.predict(X_eval)
        accuracy = float(np.mean(y_pred == y_eval.values))

        psi_values = {}
        for feature in X_eval.columns:
            if feature not in training_stats:
                continue
            stats = training_stats[feature]
            expected = rng.normal(loc=stats["mean"], scale=stats["std"], size=500)
            actual = X_eval[feature].values
            psi = detector.compute_psi(expected, actual)
            psi_values[feature] = psi

        max_psi = max(psi_values.values()) if psi_values else 0.0

        yield {
            "batch_id": batch_id,
            "psi_max": max_psi,
            "accuracy": accuracy,
            "drift_type": drift_type,
            "psi_values": psi_values,
            "baseline_accuracy": baseline_acc,
        }


def tab_live_monitor():
    """Tab 1 — Live Monitor (recruiter-proof, one-button UX)."""
    st.title("ML Model Drift Monitor")
    st.markdown("*Watch a machine learning model break in real time*")

    config = _load_config()
    training_stats = _load_training_stats()
    n_features = len(training_stats)

    if st.button("Simulate Drift", type="primary"):
        col_psi, col_acc = st.columns(2)
        psi_metric = col_psi.empty()
        acc_metric = col_acc.empty()

        banner = st.empty()
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        psi_chart_data = []
        acc_chart_data = []
        psi_chart = st.empty()
        acc_chart = st.empty()

        alert_log = st.empty()
        alerts = []

        for result in _run_simulation(config, training_stats):
            bid = result["batch_id"]
            psi = result["psi_max"]
            acc = result["accuracy"]
            baseline = result["baseline_accuracy"]
            dtype = result["drift_type"]

            psi_label, psi_color = _psi_status(psi)
            psi_metric.metric(
                "Max PSI",
                f"{psi:.3f}",
                delta=psi_label,
                delta_color="off" if psi_color == "green" else "inverse",
            )
            acc_metric.metric(
                "Model Accuracy",
                f"{acc:.1%}",
                delta=f"{(acc - baseline):+.1%} vs baseline",
                delta_color="normal",
            )

            if psi >= 0.2:
                banner.error(
                    "DRIFT DETECTED — Model accuracy degrading. "
                    "Retraining recommended."
                )
            elif psi >= 0.1:
                banner.warning("Moderate distribution shift detected — monitoring.")
            else:
                banner.success("Model operating normally")

            progress_bar.progress(bid / 150)

            phase = {"none": "Normal", "data": "Data Drift", "concept": "Concept Drift"}
            status_text.caption(f"Batch {bid}/150 — Phase: {phase.get(dtype, dtype)}")

            psi_chart_data.append({"batch": bid, "Max PSI": psi, "Threshold": 0.2})
            acc_chart_data.append({"batch": bid, "Accuracy": acc})

            if bid % 5 == 0 or bid == 150:
                psi_chart.line_chart(
                    pd.DataFrame(psi_chart_data).set_index("batch"),
                    color=["#d62728", "#999999"],
                )
                acc_chart.line_chart(
                    pd.DataFrame(acc_chart_data).set_index("batch"),
                    color=["#1f77b4"],
                )

            if psi >= 0.2 and (not alerts or alerts[-1]["batch"] < bid - 2):
                alerts.append({"batch": bid, "type": "PSI Alert", "psi": f"{psi:.3f}"})
            if acc < 0.65 and (
                not alerts or alerts[-1].get("type") != "Accuracy Alert"
            ):
                alerts.append(
                    {"batch": bid, "type": "Accuracy Alert", "accuracy": f"{acc:.1%}"}
                )

            if alerts:
                alert_log.dataframe(
                    pd.DataFrame(alerts),
                    use_container_width=True,
                    hide_index=True,
                )

            time.sleep(0.02)

        st.balloons()
        st.info(
            f"Simulation complete — monitored **{n_features}** features "
            f"across **{150 * 100:,}** predictions"
        )
    else:
        st.info(
            "Click **Simulate Drift** to watch the model degrade under "
            "data drift and concept drift."
        )
        st.caption(
            f"Monitoring {n_features} features | "
            f"PSI threshold: {config['drift']['psi_threshold']}"
        )


def tab_analysis():
    """Tab 2 — Analysis (developer/technical users)."""
    st.header("Drift Analysis")

    config = _load_config()
    training_stats = _load_training_stats()

    col1, col2 = st.columns(2)
    shift_intensity = col1.slider(
        "Drift intensity",
        0.0,
        3.0,
        1.0,
        0.1,
        help="How many standard deviations to shift features",
    )
    concept_drift = col2.checkbox("Enable concept drift", value=True)

    if st.button("Run Analysis", type="primary"):
        config["drift"]["simulation"]["shift_intensity"] = shift_intensity
        if not concept_drift:
            config["drift"]["simulation"]["concept_drift_noise"] = 0.0

        results = list(_run_simulation(config, training_stats))

        # Feature-level drift leaderboard (last drifted batch)
        st.subheader("Feature Drift Leaderboard")
        last_drifted = None
        for r in reversed(results):
            if r["drift_type"] == "data":
                last_drifted = r
                break

        if last_drifted:
            psi_vals = last_drifted["psi_values"]

            rows = []
            for feat in sorted(psi_vals, key=psi_vals.get, reverse=True):
                psi = psi_vals[feat]
                label, _ = _psi_status(psi)
                rows.append({"Feature": feat, "PSI": round(psi, 4), "Status": label})

            df_lb = pd.DataFrame(rows)
            st.dataframe(
                df_lb.style.apply(
                    lambda row: [
                        (
                            "background-color: #ffcccc"
                            if row["Status"] == "DRIFT DETECTED"
                            else (
                                "background-color: #fff3cd"
                                if row["Status"] == "Moderate Shift"
                                else "background-color: #d4edda"
                            )
                        )
                    ]
                    * len(row),
                    axis=1,
                ),
                use_container_width=True,
                hide_index=True,
            )

        # SHAP comparison
        st.subheader("SHAP Feature Importance: Baseline vs Drifted")
        try:
            shap_baseline = _load_shap_baseline()
            model = _load_model()
            train_path = os.path.join(
                _PROJECT_ROOT, "data", "processed", "train.parquet"
            )
            df_train = pd.read_parquet(train_path)
            feature_cols = list(training_stats.keys())
            X_train = df_train[feature_cols]

            if last_drifted:
                rng = np.random.default_rng(42)
                idx = rng.integers(0, len(X_train), size=500)
                X_drifted = X_train.iloc[idx].copy().reset_index(drop=True)
                for feat, stats in training_stats.items():
                    if feat in X_drifted.columns:
                        X_drifted[feat] = (
                            X_drifted[feat] + shift_intensity * stats["std"]
                        )

                from src.monitoring.shap_drift import compare_shap_under_drift

                comparison = compare_shap_under_drift(
                    model, X_train, X_drifted, shap_baseline, config
                )
                st.write(f"**{comparison['interpretation']}**")

                col_b, col_d = st.columns(2)
                col_b.markdown("**Baseline Top 5**")
                col_b.dataframe(
                    pd.DataFrame(
                        comparison["baseline_top_5"],
                        columns=["Feature", "Importance"],
                    ),
                    hide_index=True,
                )
                col_d.markdown("**Drifted Top 5**")
                col_d.dataframe(
                    pd.DataFrame(
                        comparison["drifted_top_5"],
                        columns=["Feature", "Importance"],
                    ),
                    hide_index=True,
                )
        except Exception as exc:
            st.warning(f"SHAP comparison unavailable: {exc}")

        # Evidently report embed
        st.subheader("Evidently Drift Report")
        if os.path.exists(_REPORT_PATH):
            with open(_REPORT_PATH) as fh:
                html_content = fh.read()
            st.components.v1.html(html_content, height=600, scrolling=True)
        else:
            st.info(
                "No Evidently report found. Run the API drift_report endpoint "
                "or the simulation script to generate one."
            )

        # CSV export
        st.subheader("Export Batch Metrics")
        df_export = pd.DataFrame(
            [
                {
                    "batch_id": r["batch_id"],
                    "psi_max": r["psi_max"],
                    "accuracy": r["accuracy"],
                    "drift_type": r["drift_type"],
                }
                for r in results
            ]
        )
        st.download_button(
            "Download CSV",
            df_export.to_csv(index=False),
            "drift_metrics.csv",
            "text/csv",
        )


def tab_how_it_works():
    """Tab 3 — How It Works (educational)."""
    st.header("How It Works")

    st.subheader("Architecture")
    st.code(
        """
Training Data
     |
     v
[RandomForest Model] --> models/random_forest.pkl
     |
     v
[Training Stats] --> models/training_stats.json
     |                (per-feature mean/std/min/max)
     |
=== DEPLOYMENT ===
     |
     v
[FastAPI /predict] <-- incoming data batch
     |
     +---> [PSI Calculator] -- "How much did each feature shift?"
     |          |
     |          +---> PSI < 0.1: No drift
     |          +---> PSI 0.1-0.2: Monitor
     |          +---> PSI > 0.2: ALERT!
     |
     +---> [KS Test] -- "Is this statistically different?"
     |
     +---> [Accuracy Monitor] -- "Is the model still correct?"
     |          |
     |          +---> Accuracy drop > 10%: CONCEPT DRIFT
     |
     +---> [Prometheus] --> [Grafana Dashboard]
     |                  --> [Alertmanager] --> Alert!
     |
     v
[Evidently Report] --> HTML drift report
""",
        language=None,
    )

    st.subheader("PSI (Population Stability Index)")
    st.latex(
        r"PSI = \sum (P_{actual} - P_{expected})"
        r" \cdot \ln\frac{P_{actual}}{P_{expected}}"
    )
    st.markdown("""
PSI measures how much a feature's distribution has changed since training.
Think of it like a **credit score for your data** — if PSI goes above 0.2,
your model's "credit" is bad and it needs retraining.

| PSI Range | Interpretation |
|-----------|---------------|
| < 0.1 | No significant change |
| 0.1 - 0.2 | Moderate shift — monitor closely |
| > 0.2 | Significant drift — retrain the model |
""")

    st.subheader("Data Drift vs Concept Drift")
    st.markdown("""
**DATA DRIFT**: The inputs change
(e.g., average income shifts from \\$50K to \\$80K).
PSI catches this by comparing feature distributions.

**CONCEPT DRIFT**: The rules change
(e.g., same income now predicts different defaults).
The features look the same, but the model's predictions become wrong.
Only accuracy monitoring catches this.

**You need BOTH** — PSI alone misses concept drift,
accuracy alone can't tell you *which* features shifted.
""")

    st.subheader("Monitoring Stack")
    st.markdown("""
| Component | Role |
|-----------|------|
| **FastAPI** | Serves predictions + computes drift metrics |
| **Prometheus** | Scrapes metrics every 15s, evaluates alert rules |
| **Grafana** | Visualizes PSI gauges, latency, accuracy over time |
| **Alertmanager** | Fires alerts when PSI > 0.2 or accuracy < 65% |
""")

    st.markdown("---")
    st.markdown(
        "[View source on GitHub]" "(https://github.com/Priyrajsinh/B5-Drift-Monitor)"
    )


def main() -> None:
    """Launch the Streamlit dashboard."""
    st.set_page_config(
        page_title="B5 Drift Monitor",
        page_icon="📊",
        layout="wide",
    )

    tab1, tab2, tab3 = st.tabs(["Live Monitor", "Analysis", "How It Works"])

    with tab1:
        tab_live_monitor()
    with tab2:
        tab_analysis()
    with tab3:
        tab_how_it_works()


if __name__ == "__main__":
    main()
