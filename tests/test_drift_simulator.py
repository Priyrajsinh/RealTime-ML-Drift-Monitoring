"""Tests for src/monitoring/drift_simulator.py."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from src.monitoring.drift_simulator import (
    DriftSimulator,
    plot_accuracy_collapse,
    plot_drift_flag_timeline,
    plot_psi_timeline,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FEATS = ["F0", "F1", "F2", "F3", "F4"]
_STATS: dict = {
    col: {"mean": 0.0, "std": 1.0, "min": -4.0, "max": 4.0} for col in _FEATS
}
_CONFIG: dict = {
    "drift": {
        "psi_threshold": 0.2,
        "ks_alpha": 0.05,
        "evidently_report_ttl": 60,
        "simulation": {
            "shift_intensity": 1.0,
            "concept_drift_noise": 0.3,
        },
    }
}

rng = np.random.default_rng(0)


def _make_df(n: int = 500) -> pd.DataFrame:
    return pd.DataFrame(rng.standard_normal((n, 5)), columns=_FEATS)


def _make_y(n: int = 500) -> pd.Series:
    return pd.Series(rng.integers(0, 2, size=n))


def _mock_model(n_samples: int | None = None) -> MagicMock:
    """Return a mock model whose predict always returns zeros."""
    m = MagicMock()
    m.predict.side_effect = lambda X: np.zeros(len(X), dtype=int)
    return m


def _make_simulator(n: int = 500) -> DriftSimulator:
    X = _make_df(n)
    y = _make_y(n)
    model = _mock_model()
    return DriftSimulator(X, y, _STATS, _CONFIG, model=model)


# ---------------------------------------------------------------------------
# simulate_data_drift
# ---------------------------------------------------------------------------


def test_data_drift_returns_two_dataframes():
    sim = _make_simulator()
    normal, drifted = sim.simulate_data_drift(n_normal=200, n_drifted=200)
    assert isinstance(normal, pd.DataFrame)
    assert isinstance(drifted, pd.DataFrame)
    assert len(normal) == 200
    assert len(drifted) == 200


def test_data_drift_shifts_features():
    """Drifted batch mean should be higher than normal batch mean for all features."""
    sim = _make_simulator()
    normal, drifted = sim.simulate_data_drift(
        n_normal=1000, n_drifted=1000, shift_intensity=2.0
    )
    for feat in _FEATS:
        assert (
            drifted[feat].mean() > normal[feat].mean()
        ), f"Expected drifted mean > normal mean for {feat}"


def test_data_drift_gradual():
    """The last sub-batch should have a larger mean than the first sub-batch."""
    sim = _make_simulator(n=2000)
    _, drifted = sim.simulate_data_drift(
        n_normal=100, n_drifted=500, shift_intensity=1.0
    )
    sub_size = len(drifted) // 5
    first_sub = drifted.iloc[:sub_size]
    last_sub = drifted.iloc[-sub_size:]
    # Last sub-batch uses +1.0*std; first sub-batch uses +0.2*std
    for feat in _FEATS:
        assert (
            last_sub[feat].mean() > first_sub[feat].mean()
        ), f"Expected last sub-batch mean > first sub-batch mean for {feat}"


def test_data_drift_preserves_columns():
    sim = _make_simulator()
    normal, drifted = sim.simulate_data_drift(n_normal=100, n_drifted=100)
    assert list(normal.columns) == _FEATS
    assert list(drifted.columns) == _FEATS


# ---------------------------------------------------------------------------
# simulate_concept_drift
# ---------------------------------------------------------------------------


def test_concept_drift_preserves_features():
    """Features must be IDENTICAL before and after concept drift."""
    sim = _make_simulator()
    X = _make_df(200)
    y = _make_y(200)
    X_out, _ = sim.simulate_concept_drift(X, y, noise_prob=0.3)
    pd.testing.assert_frame_equal(X, X_out)


def test_concept_drift_flips_labels():
    """Approximately noise_prob fraction of labels should differ."""
    sim = _make_simulator()
    n = 5000
    X = _make_df(n)
    y = _make_y(n)
    _, y_out = sim.simulate_concept_drift(X, y, noise_prob=0.3)
    flip_rate = float((y.values != y_out.values).mean())
    # Allow wide tolerance — expected ~0.3 * (1 - p_same_after_flip) ≈ 0.21
    assert 0.10 <= flip_rate <= 0.40, f"Unexpected flip rate: {flip_rate:.3f}"


def test_concept_drift_returns_series():
    sim = _make_simulator()
    X = _make_df(100)
    y = _make_y(100)
    _, y_out = sim.simulate_concept_drift(X, y)
    assert isinstance(y_out, pd.Series)
    assert len(y_out) == 100


# ---------------------------------------------------------------------------
# run_full_simulation
# ---------------------------------------------------------------------------


def test_full_simulation_returns_150_batches():
    sim = _make_simulator()
    results = sim.run_full_simulation()
    assert len(results) == 150


def test_full_simulation_returns_metrics():
    sim = _make_simulator()
    results = sim.run_full_simulation()
    required_keys = {"batch_id", "psi_max", "accuracy", "drift_type"}
    for r in results:
        assert required_keys == set(r.keys()), f"Missing keys in batch {r['batch_id']}"


def test_full_simulation_drift_types():
    sim = _make_simulator()
    results = sim.run_full_simulation()

    none_batches = [r for r in results if r["drift_type"] == "none"]
    data_batches = [r for r in results if r["drift_type"] == "data"]
    concept_batches = [r for r in results if r["drift_type"] == "concept"]

    assert len(none_batches) == 50
    assert len(data_batches) == 50
    assert len(concept_batches) == 50


def test_full_simulation_batch_ids_sequential():
    sim = _make_simulator()
    results = sim.run_full_simulation()
    ids = [r["batch_id"] for r in results]
    assert ids == list(range(1, 151))


def test_full_simulation_psi_rises_during_data_drift():
    """Max PSI during data drift batches should exceed PSI during normal batches."""
    sim = _make_simulator()
    results = sim.run_full_simulation()
    normal_psi = np.mean([r["psi_max"] for r in results if r["drift_type"] == "none"])
    data_psi = np.mean([r["psi_max"] for r in results if r["drift_type"] == "data"])
    assert data_psi > normal_psi, "Data drift should raise PSI above normal level"


# ---------------------------------------------------------------------------
# Plotting functions (smoke tests — just check files are created)
# ---------------------------------------------------------------------------

_DUMMY_RESULTS = (
    [
        {"batch_id": i, "psi_max": 0.05, "accuracy": 0.82, "drift_type": "none"}
        for i in range(1, 51)
    ]
    + [
        {
            "batch_id": i,
            "psi_max": 0.05 + (i - 50) * 0.005,
            "accuracy": 0.80 - (i - 50) * 0.003,
            "drift_type": "data",
        }
        for i in range(51, 101)
    ]
    + [
        {"batch_id": i, "psi_max": 0.08, "accuracy": 0.60, "drift_type": "concept"}
        for i in range(101, 151)
    ]
)


def test_plot_accuracy_collapse_saves_file(tmp_path):
    out = str(tmp_path / "accuracy_collapse.png")
    plot_accuracy_collapse(_DUMMY_RESULTS, out)
    import os

    assert os.path.exists(out)


def test_plot_psi_timeline_saves_file(tmp_path):
    out = str(tmp_path / "psi_timeline.png")
    plot_psi_timeline(_DUMMY_RESULTS, out)
    import os

    assert os.path.exists(out)


def test_plot_drift_flag_timeline_saves_file(tmp_path):
    out = str(tmp_path / "drift_timeline.png")
    plot_drift_flag_timeline(_DUMMY_RESULTS, out)
    import os

    assert os.path.exists(out)
