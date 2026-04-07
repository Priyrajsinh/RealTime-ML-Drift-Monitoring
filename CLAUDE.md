# CLAUDE.md — B5 Drift Monitor Project Rules

## Carry-Forward Error Prevention (from carry.md)
1. Set `target-version = ["py310"]` in `[tool.black]` in `pyproject.toml`.
2. Only import modules you actually use. Run `flake8` before every commit.
3. Run `isort src/ tests/ --profile black` before every commit.
4. Every `src/` file needs at least one test. Coverage gate: 70%.
5. Use `from pythonjsonlogger import json as jsonlogger` (not deprecated path).
6. Use `from pandera.pandas import Column, DataFrameSchema` (not top-level pandera).
7. Pydantic v2 only: `@field_validator` + `@classmethod`, never `@validator`.
8. Run ALL 5 gates before every commit: black -> isort -> flake8 -> bandit -> pytest.
9. Use Python 3.10. Verify with `py --list` before scaffold.
10. Create `.flake8` with `max-line-length = 88` matching `[tool.black]`.
11. Use built-in `open()` for file I/O, never `Path.open()`.
12. No module-level side effects (no `set_seed()` at import time).
13. HF Space must be 100% self-contained. Never import from `src/`.
14. HF Space `requirements.txt` must match every import in `app.py`.
15. Do not use `allow_flagging` in Gradio (removed in v5).
16. Use `plt.switch_backend("Agg")` after all imports for matplotlib.
17. Annotate mixed-type dicts explicitly for mypy: `dict[str, str | int]`.
18. Verify `git config user.email` = `priyrajsinh03@gmail.com` before first commit.
19. Tab 1 = non-technical UX. Dev features in Tab 2. "How It Works" in Tab 3.

## B5-Specific Rules (Monitoring Pitfalls)
20. Prometheus metric names MUST use underscores, NOT hyphens (e.g. `drift_psi_value` not `drift-psi-value`).
21. Evidently API: use `DataDriftPreset()` not `DataDriftTab()` (changed between versions).
22. Gauge metrics use `.set(value)`, NOT `.inc()` — `.inc()` is for Counters only.
23. `training_stats.json` must be validated with pandera on load (std > 0 check).
24. Drift report caching: TTL 60 seconds on `/api/v1/drift_report` (same principle as B4 lru_cache).
25. Convert lists to tuples before passing to `@lru_cache` functions (carry.md ERROR — unhashable type).
26. Import modules at top of test files, not inside test functions (carry.md ERROR — patch resolver).
27. Scripts with `if __name__ == "__main__"` must have explicit tests (carry.md ERROR — 0% coverage).
28. Streamlit/Gradio Tab 3 = "How It Works" (ASCII architecture, PSI formula, drift types).
29. UX TEST: a recruiter must be able to use the app in 3 seconds (B3 Day 8 lesson).
30. CLAUDE.md at project root with all rules inlined (auto-read by Claude Code).

## Project-Specific Rules
- `config/config.yaml` is the single source of truth. Never hardcode hyperparameters.
- Use `get_logger(__name__)` everywhere. Zero `print()` statements.
- `filterwarnings = ["error::DeprecationWarning"]` is set in pytest config — do not suppress warnings.
- All dependency versions are pinned in `requirements.txt`. Do not add unpinned deps.
- `training_stats.json` stores per-feature mean/std/min/max — validated with pandera before use.
- Prometheus metrics defined in `src/monitoring/metrics.py` — imported, never re-created.
- Evidently reports cached with TTL to avoid regeneration on repeated requests.
