# B5 — Real-Time ML API + Drift Monitoring

## Live Demo
[**Try it on Hugging Face Spaces**](https://huggingface.co/spaces/Priyrajsinh/B5-Drift-Monitor) *(add after deploying)*

## What This Does
This project monitors a machine learning model in real-time, detecting when
the data it receives starts looking different from what it was trained on.
Click the demo above to watch a model's accuracy collapse as data drifts —
and see how monitoring catches the problem before it causes damage.

The model is a Random Forest trained on the UCI Credit Card Default dataset (30,000 samples,
23 features). Two types of drift are simulated: **data drift** (feature distributions
shift) and **concept drift** (same features, different default patterns).

## Architecture
```
[FastAPI API] ──► [Prometheus] ──► [Grafana Dashboard]
      │                                     │
      ▼                                     ▼
[Drift Detector] ──► [Alertmanager] ──► Alert!
      │
      ├── PSI (data drift — distribution shift)
      ├── KS test (statistical distribution test)
      ├── Accuracy monitor (concept drift)
      ├── Evidently HTML reports
      └── SHAP comparison (what changed?)
```

## Key Results

### Accuracy Collapse Under Drift
![Accuracy Collapse](reports/figures/accuracy_collapse.png)

The model maintains ~82% accuracy during normal operation (batches 1–50),
degrades as data distributions shift (batches 51–100), then collapses
when the feature-label relationship changes (batches 101–150).

### PSI Formula
**Population Stability Index** measures distribution shift:
```
PSI = Σ (actual_% − expected_%) × ln(actual_% / expected_%)
```

| PSI Value   | Interpretation                       |
|-------------|--------------------------------------|
| < 0.1       | No significant shift — model is safe |
| 0.1 – 0.2   | Moderate shift — monitor closely     |
| > 0.2       | Significant shift — **retrain!**     |

### Data Drift vs Concept Drift
| Type          | What Changes              | Detection Method    | Example                                 |
|---------------|---------------------------|---------------------|-----------------------------------------|
| Data Drift    | Feature distributions     | PSI, KS test        | Average credit limit rises from 167K→250K |
| Concept Drift | Feature-label relationship | Accuracy monitoring | Same income → different default rates   |

**Key insight:** PSI catches data drift but is *blind* to concept drift.
You need accuracy monitoring running in parallel.

### SHAP Under Drift (B2 tie-in)
![SHAP Comparison](reports/figures/shap_drift_comparison.png)

SHAP reveals *which features* drove the change — not just that drift happened.
`PAY_0` (recent payment status) consistently dominates when distributions shift.

### PSI Timeline
![PSI Timeline](reports/figures/psi_timeline.png)

Note: PSI rises during data drift (batches 51–100) but stays flat during
concept drift (batches 101–150). This demonstrates why accuracy monitoring
is required alongside PSI.

## Docker Compose Stack
```
4 services:
├── app           — FastAPI + Prometheus /metrics endpoint
├── prometheus    — scrapes every 15s, stores PSI time-series
├── grafana       — pre-provisioned dashboard (PSI gauge + accuracy)
└── alertmanager  — fires alert when PSI > 0.2
```

## Quick Start
```bash
make install        # pip install requirements
make train          # train model + save stats + SHAP baseline
make simulate       # generate drift plots (reports/figures/)
make serve          # FastAPI on :8000
make dashboard      # Streamlit on :8501
make docker-up      # full stack (FastAPI + Prometheus + Grafana + Alertmanager)
```

API endpoints:
- `GET  /health`              — health check
- `POST /api/v1/predict`      — single prediction
- `POST /api/v1/predict_batch`— batch prediction
- `GET  /api/v1/drift_report` — cached drift report (60s TTL)
- `GET  /metrics`             — Prometheus scrape endpoint

## Tech Stack
| Tool | Role |
|------|------|
| scikit-learn | Random Forest model |
| FastAPI + uvicorn | REST API |
| Prometheus | Time-series metrics collection |
| Grafana | Dashboard visualisation |
| Alertmanager | PSI threshold alerting |
| Evidently | HTML drift reports |
| Streamlit | Interactive dashboard |
| SHAP | Feature importance under drift |
| pandera | Runtime data validation |
| Docker Compose | 4-service orchestration |
| MLflow | Experiment tracking |

## Project Structure
```
B5-Drift-Monitor/
├── config/config.yaml          # single source of truth for all hyperparams
├── src/
│   ├── api/app.py              # FastAPI routes + rate limiting
│   ├── dashboard/streamlit_app.py
│   ├── data/                   # dataset loading + pandera schemas
│   ├── model/                  # train.py + predict.py
│   └── monitoring/
│       ├── drift_detector.py   # PSI + KS + Evidently
│       ├── drift_simulator.py  # 150-batch simulation engine
│       ├── metrics.py          # Prometheus metrics definitions
│       └── shap_drift.py       # SHAP baseline vs drifted comparison
├── hf_space/                   # self-contained Gradio demo
│   ├── app.py
│   └── requirements.txt
├── tests/                      # 92 tests, 75% coverage
├── docker-compose.yml
├── prometheus/
├── grafana/
├── alertmanager/
└── Makefile
```

## Part of a 5-Project ML Portfolio
| Project | What It Teaches |
|---------|----------------|
| B1 | HuggingFace Fine-Tuning + Production FastAPI |
| B2 | XGBoost + SHAP Explainability Dashboard |
| B3 | PyTorch LSTM Time Series Forecasting |
| B4 | Semantic Search with FAISS + Hybrid Search |
| **B5** | **Real-Time ML Monitoring + Drift Detection** |
