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

## Day 2 — 2026-04-07 — Drift detector with PSI + KS + Evidently + concept drift
> Project: B5-Drift-Monitor

### What was done
- Implemented `DriftDetector` class with `compute_psi()`, `compute_ks_test()`, `detect_data_drift()`, `detect_concept_drift()`, `generate_evidently_report()`, and `get_cached_drift_report()`
- PSI uses epsilon (1e-6) to prevent log(0) division errors on empty bins
- Concept drift detection compares batch accuracy against baseline (>10% drop = drift)
- Evidently report generates HTML using `DataDriftPreset` via v0.7 API (`Report([preset])` + `snapshot.save_html()`)
- Drift report caching with TTL (60s default) to avoid expensive regeneration
- 10 new tests covering PSI, KS, data drift, concept drift, Evidently HTML, and caching
- Updated scaffold test to pass required `training_stats` and `config` args
- Added numpy.core deprecation warning filter for evidently compatibility
- All 5 CI gates green; 40/40 tests pass; 97% coverage

### Why it was done
- Core monitoring capability: detect when incoming data distributions shift (data drift) or when the model's learned relationship breaks (concept drift)
- Three complementary methods: PSI (industry-standard threshold), KS test (statistical significance), Evidently (visual HTML report)
- Caching prevents redundant computation on repeated API calls within short windows

### How it was done
- PSI: bin both distributions into equal-width buckets, compute `sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))`
- KS test: `scipy.stats.ks_2samp` returns (statistic, p_value); p < 0.05 means distributions differ significantly
- Data drift synthesises reference distribution from `training_stats` (mean/std) via `np.random.default_rng`
- Concept drift: predict on batch, compare accuracy to baseline; accuracy drop >10% flags drift
- Evidently 0.7 API: `from evidently import Report` + `from evidently.presets import DataDriftPreset`; `report.run()` returns a `Snapshot` with `save_html()`
- Caching stores `_last_report_time` + `_last_report_result`; returns cached if within TTL

### Why this tool / library — not alternatives
| Tool Used | Why This | Rejected Alternative | Why Not |
|-----------|----------|---------------------|---------|
| PSI | Banking industry standard; interpretable 0.1/0.2 thresholds | JS divergence | PSI thresholds are more widely understood in production |
| KS test (scipy) | Non-parametric; no distribution assumption required | Chi-squared | Chi-squared needs categorical bins; KS works on continuous |
| Evidently DataDriftPreset | One-call HTML report with per-feature drift stats | Custom matplotlib | Evidently handles 20+ statistical tests automatically |
| Time-based TTL cache | Simple, predictable invalidation for expensive reports | lru_cache | lru_cache needs hashable args; DataFrames are unhashable |
| Accuracy-based concept drift | Simple, interpretable; no extra library needed | ADWIN/DDM | ADWIN planned for P4; accuracy drop is sufficient for B5 |

### Definitions (plain English)
- **PSI (Population Stability Index)**: measures how much a distribution shifted; <0.1 = fine, 0.1-0.2 = watch, >0.2 = alert.
- **KS test**: checks if two samples come from the same distribution; low p-value = they differ.
- **Data drift**: input feature distributions change (e.g., customers are suddenly older on average).
- **Concept drift**: the relationship between features and labels changes (e.g., same age now means different default risk).
- **TTL cache**: store a result for N seconds; serve the cached version until time expires.

### Real-world use case
- PSI monitoring: required by US/EU banking regulators (OCC, Basel III) to detect model decay in credit scoring models.
- Data vs concept drift distinction: used by Spotify to separate "users changed" (data drift) from "taste changed" (concept drift) in recommendation models.

### How to remember it
- Data drift = "different people walked into the store" (input changed). Concept drift = "same people, but they want different things now" (relationship changed).
- PSI threshold: 0.2 is like a speed limit — below it you're fine, above it you get flagged.

### Status
- [x] Done
- Next step: Day 3 — DriftSimulator, Prometheus metrics wiring, FastAPI drift endpoints

---

## Day 3 — 2026-04-08 — Drift Simulation Engine + Portfolio Plots
> Project: B5-Drift-Monitor

### What was done
- Implemented `DriftSimulator` with `simulate_data_drift` (gradual +0.2→+1.0 std shift) and `simulate_concept_drift` (label flip, features unchanged).
- Implemented `run_full_simulation`: 150 batches (50 normal / 50 data drift / 50 concept drift), computing per-batch PSI and accuracy.
- Created 3 portfolio plots: `accuracy_collapse.png` (money shot), `psi_timeline.png`, `drift_flag_timeline.png`.
- Implemented `compare_shap_under_drift` and `plot_shap_comparison` (B2 SHAP tie-in).
- 22 new tests; 62 total passing; coverage 97.81%.

### Why it was done
- Recruiters need a visual that shows WHY drift monitoring matters without any technical explanation.
- Concept drift vs data drift distinction is the key educational insight of this project.
- PSI alone cannot catch concept drift — the PSI timeline plot proves this visually.

### How it was done
- Gradual shift: 5 sub-batches per drift epoch, shift intensity 0.2→1.0 × std.
- Concept drift: `numpy` random mask flips ~30% of labels; X is untouched.
- `plt.switch_backend("Agg")` placed after ALL imports to avoid E402 flake8 error.
- SHAP mocked in tests with `@patch("src.monitoring.shap_drift.shap")` — module-level import required for mock to intercept.

### Why this tool / library — not alternatives
| Tool Used | Why This | Rejected Alternative | Why Not |
|-----------|----------|---------------------|---------|
| `numpy.random.default_rng` | Reproducible, modern API | `random.random()` | No seeding, scalar only |
| `matplotlib Agg backend` | Headless PNG output, no display needed | Default TkAgg | Crashes in CI (no display) |
| `shap.TreeExplainer` | Exact SHAP for tree models, fast | KernelExplainer | Slow, model-agnostic approximation |
| `pd.concat` for sub-batches | Vectorised, preserves dtypes | Loop append | Deprecated `.append()`, O(n²) |

### Definitions (plain English)
- **Data drift**: The distribution of input features changes — the model sees different kinds of data than it trained on.
- **Concept drift**: Same features, but the correct answer has changed — the world's rules have shifted.
- **PSI (Population Stability Index)**: A number measuring how much a distribution shifted; >0.2 means "alert, something changed".
- **SHAP importance shift**: How much a feature's influence on predictions changed between training and drifted data.

### Real-world use case
- Stripe: uses accuracy monitoring + feature distribution checks to detect when fraud patterns shift after an economic shock.
- Netflix: monitors concept drift in recommendation models when user behaviour changes (e.g., holiday season).

### How to remember it
- **Data drift = the weather changed**; concept drift = **the rules of cricket changed mid-game**.
- PSI catches the weather change but not the rule change — you need accuracy monitoring for that.

### Status
- [x] Done
- Next step: Day 4 — FastAPI drift endpoints + Prometheus metrics wiring

---

## Day 4 — 2026-04-08 — FastAPI + Prometheus metrics + drift report caching
> Project: B5-Drift-Monitor

### What was done
- Rewrote `src/monitoring/metrics.py` with uppercase canonical names (`PREDICTION_COUNTER`, `FEATURE_PSI` with label, etc.) plus backward-compat aliases.
- Implemented full `src/api/app.py`: lifespan startup, slowapi rate limiting, CORS, content-length guard, Prometheus instrumentator, 5 endpoints.
- Added `BatchPredictRequest` / `BatchPredictResponse` to `src/data/schemas.py`.
- Wrote `tests/test_api.py` (12 tests, async httpx client, mocked state) and `tests/test_metrics.py` (8 tests).
- All 5 CI gates green; 82/82 tests passed; 94% coverage.

### Why it was done
- Expose the drift monitor as a production-grade HTTP API with observable Prometheus metrics and cached drift reports.

### How it was done
- Copied B4 lifespan/slowapi/CORS/content-length pattern into B5.
- `_state` dict holds `model_server`, `drift_detector`, rolling buffer (deque maxlen=500), and counters.
- Single predict → appends to buffer; runs drift check every `ROLLING_WINDOW` calls.
- Batch predict → drift detection runs on the submitted batch directly.
- `/api/v1/drift_report` delegates TTL caching to `DriftDetector.get_cached_drift_report` (TTL=60s from config).
- Tests patch `_state` via `autouse` fixture; `ASGITransport` skips lifespan so mocks are injected cleanly.

### Why this tool / library — not alternatives
| Tool Used | Why This | Rejected Alternative | Why Not |
|-----------|----------|---------------------|---------|
| slowapi | Starlette-native rate limiter, zero config | fastapi-limiter | Requires Redis dependency |
| prometheus-fastapi-instrumentator | Auto-instruments all routes in one line | manual middleware | More boilerplate, misses edge cases |
| httpx AsyncClient + ASGITransport | Async test client, no server needed | TestClient (sync) | Can't test async endpoints directly |
| deque(maxlen=500) | O(1) append, automatic eviction | list + slice | Slower, needs manual trimming |

### Definitions (plain English)
- **Gauge**: A Prometheus metric that can go up and down (e.g., current PSI value); use `.set()`.
- **Counter**: A Prometheus metric that only increases (e.g., total predictions); use `.inc()`.
- **PSI (Population Stability Index)**: Measures how much a feature's distribution has shifted; > 0.2 = significant drift.
- **TTL cache**: "Time-to-live" — return a stored result until it expires, then regenerate.
- **ASGI lifespan**: Startup/shutdown hook that runs once when the FastAPI server starts and stops.

### Real-world use case
- Stripe uses a similar pattern: a FastAPI service exposes `/metrics` that Prometheus scrapes every 15s; Grafana dashboards alert when fraud-model PSI spikes above threshold.

### How to remember it
- Gauge = Gas gauge on a car (can go up and down). Counter = odometer (only goes up). Never `.inc()` a gas gauge.

### Status
- [x] Done
- Next step: Day 5 — Streamlit dashboard wiring + Gradio HF Space

---
