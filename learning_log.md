---

## Day 0 — 2026-04-07 — Scaffold B5 Real-Time ML API + Drift Monitoring
> Project: B5-Drift-Monitor

### What was done
- Created full project scaffold: venv, pyproject.toml, .flake8, config/, src/, utils/, tests/, .github/CI
- Wrote Pydantic v2 schemas (PredictRequest, PredictResponse, DriftReport, HealthResponse)
- Wrote pandera validation schema for training_stats.json with std > 0 check
- Defined Prometheus metrics stubs (Counter, Gauge, Histogram) with underscore names
- All 5 CI gates green: black / isort / flake8 / bandit / pytest @ 93% coverage

### Why it was done
- Establish a reproducible, linted, tested foundation before any real implementation
- Carry-forward rules from B4 require all gates green from Day 0 to avoid cascade failures later

### How it was done
- Copied pyproject.toml pattern from B4, changed name and confirmed `target-version = ["py310"]`
- Used `from pythonjsonlogger import json as jsonlogger` (not deprecated path)
- Used `from pandera.pandas import Check, Column, DataFrameSchema` (pandera v2 API)
- Used `@field_validator` + `@classmethod` throughout (Pydantic v2 only)
- Added tests for every src/ module including FastAPI TestClient for /health

### Why this tool / library — not alternatives
| Tool Used | Why This | Rejected Alternative | Why Not |
|-----------|----------|---------------------|---------|
| evidently | Open-source, production-ready ML monitoring with drift presets | whylogs | Less battle-tested for structured tabular data |
| prometheus-client | Industry-standard time-series metrics; pairs with Grafana | statsd | No native histogram/gauge semantics |
| pandera | Schema validation with statistical checks (std > 0) | pydantic alone | Pydantic validates shape, not statistical properties |
| RandomForestClassifier | SHAP-compatible TreeExplainer, strong baseline | XGBoost | RF needs no GPU, simpler debug path |
| PSI | Banking industry standard for distribution shift | MMD | PSI is interpretable (threshold 0.2 well-known) |

### Definitions (plain English)
- **PSI (Population Stability Index)**: a number that measures how much a data distribution has shifted; >0.2 means significant drift.
- **KS test**: statistical test that checks if two samples come from the same distribution without assuming a shape.
- **Prometheus Gauge**: a metric that can go up or down (like temperature); Counter only goes up.
- **Pandera**: a library that validates DataFrame columns have correct types AND correct statistical properties.
- **Concept drift**: when the relationship between features and labels changes (model becomes wrong even if input data looks the same).

### Real-world use case
- Evidently + Prometheus: used by Booking.com and Mercado Libre for real-time ML monitoring in production
- PSI threshold 0.2: standard metric in credit-risk models at banks (Basel III compliance monitoring)

### How to remember it
- PSI = "how far did my population move?" — like checking if the people walking into your store today look the same as last month.
- Prometheus Gauge vs Counter: Gauge = thermometer (goes up/down), Counter = odometer (only goes up).

### Status
- [x] Done
- Next step: Day 1 — data loading (load_credit_default), feature engineering, train/test split, save training_stats.json

---

## Day 1 — 2026-04-07 — Data loading, RF training, SHAP baseline, pandera validation
> Project: B5-Drift-Monitor

### What was done
- Implemented `load_credit_default()` — reads local CSV, drops ID column, returns (X, y)
- Implemented `compute_training_stats()` and `save_training_stats()` using built-in `open()`
- Added DataFrame-level pandera check: min <= max (extends existing std > 0 check)
- Implemented `train_model()` — RF, SHAP baseline, MLflow logging, parquet splits saved
- Implemented `ModelServer` class with `predict()` returning prediction + probability + drift_warning
- All 5 CI gates green; 30/30 tests pass; 97% coverage

### Why it was done
- `training_stats.json` stores per-feature mean/std/min/max — needed by DriftDetector on Day 2
- SHAP baseline captured at train time so Day 4 can detect SHAP drift under covariate shift
- MLflow logs params/metrics/artifacts for experiment reproducibility

### How it was done
- `pd.read_csv("data/raw/credit_default.csv")` — file already downloaded, no ucimlrepo fetch
- `shap.TreeExplainer(model, data=background).shap_values(background)` — list[class0, class1] for RF binary
- Monkeypatched `train_module.MODEL_PATH` etc. to `tmp_path` in tests — avoids real FS side effects
- Mocked `shap.TreeExplainer` and entire `mlflow` object to keep tests fast and isolated

### Why this tool / library — not alternatives
| Tool Used | Why This | Rejected Alternative | Why Not |
|-----------|----------|---------------------|---------|
| `shap.TreeExplainer` | Tree-native, exact SHAP values; no sampling needed for RF | KernelExplainer | 100x slower, uses sampling approximation |
| `joblib.dump` | scikit-learn's own serialiser; handles large numpy arrays efficiently | pickle | joblib is pickle-based but handles ndarrays with memory-mapping |
| `mlflow.log_artifact` | Tracks JSON files as run artifacts alongside metrics | Manual file copy | mlflow links artifact to run ID for reproducibility |
| `pd.DataFrame(stats).T` | Converts feature→stat dict to a row-per-feature DataFrame in one line | Iterating rows | Concise; pandera expects row = one observation |

### Definitions (plain English)
- **SHAP baseline**: the average feature importance (mean |SHAP|) computed on training data; used later to detect when feature importance shifts after drift.
- **TreeExplainer**: a SHAP explainer that uses the tree structure itself to compute exact Shapley values (no approximation).
- **training_stats.json**: a JSON file storing mean/std/min/max per feature, computed on training data only — the reference distribution for drift detection.
- **Parquet**: a columnar file format; faster to read than CSV for large DataFrames because it stores column statistics in the file header.

### Real-world use case
- Saving a SHAP baseline at train time: used by Uber (Michelangelo) and LinkedIn to detect when model explanations drift — a signal that concept drift has occurred even if PSI looks normal.

### How to remember it
- SHAP baseline = "take a photograph of feature importances at training time". On Day 4, you compare the new photograph to the old one. If they look different, something changed inside the model.

### Status
- [x] Done
- Next step: Day 2 — DriftDetector (PSI + KS test), DriftSimulator, Prometheus metrics wiring

---
