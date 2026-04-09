"""Tests for src/dashboard/streamlit_app.py — importability and helpers."""

from src.dashboard.streamlit_app import (
    _psi_status,
    main,
    tab_analysis,
    tab_how_it_works,
    tab_live_monitor,
)


def test_psi_status_no_drift():
    label, color = _psi_status(0.05)
    assert label == "No Drift"
    assert color == "green"


def test_psi_status_moderate():
    label, color = _psi_status(0.15)
    assert label == "Moderate Shift"
    assert color == "orange"


def test_psi_status_drift_detected():
    label, color = _psi_status(0.25)
    assert label == "DRIFT DETECTED"
    assert color == "red"


def test_psi_status_boundary_low():
    label, _ = _psi_status(0.1)
    assert label == "Moderate Shift"


def test_psi_status_boundary_high():
    label, _ = _psi_status(0.2)
    assert label == "DRIFT DETECTED"


def test_functions_are_callable():
    assert callable(main)
    assert callable(tab_live_monitor)
    assert callable(tab_analysis)
    assert callable(tab_how_it_works)
