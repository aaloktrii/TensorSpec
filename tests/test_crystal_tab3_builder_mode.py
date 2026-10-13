"""Crystal Tab 3: Plain default vs Bernevig opt-in; no silent strain.

Full CrystalViewerSuite opens PyVista (segfaults headless) — test routing logic
and crystallography helpers without constructing the suite.
"""
from __future__ import annotations

import inspect

from pymatgen.core import Lattice, Structure

from tensorspec.core.crystallography import CrystalEngine
from tensorspec.gui.suites import crystal_suite as cs


def _hex_layer(a: float, symbol: str = "C") -> Structure:
    lat = Lattice.from_parameters(a, a, 20.0, 90, 90, 120)
    return Structure(lat, [symbol], [[0, 0, 0.5]])


def test_aligned_needs_strain_detects_mismatch():
    layers = [
        {"struct": _hex_layer(2.46), "sc_x": 1, "sc_y": 1, "twist": 0.0, "z_shift": 0.0},
        {"struct": _hex_layer(3.15, "B"), "sc_x": 1, "sc_y": 1, "twist": 0.0, "z_shift": 3.0},
    ]
    assert CrystalEngine.aligned_needs_strain_dialog(layers, 0) is True


def test_matched_lattices_no_strain_dialog():
    layers = [
        {"struct": _hex_layer(2.46), "sc_x": 1, "sc_y": 1, "twist": 0.0, "z_shift": 0.0},
        {"struct": _hex_layer(2.46, "C"), "sc_x": 1, "sc_y": 1, "twist": 0.0, "z_shift": 3.0},
    ]
    assert CrystalEngine.aligned_needs_strain_dialog(layers, 0) is False


def test_push_source_no_silent_ab_to_twist():
    """Regression: AB stacking must not rewrite aligned → twist."""
    src = inspect.getsource(cs.CrystalViewerSuite.push_current_to_workspace)
    assert 'combo_bernevig_stacking.currentText() == "AB"' not in src
    assert 'currentData() or "plain"' in src or 'builder == "bernevig"' in src


def test_bernevig_opted_in_source_is_builder_only():
    src = inspect.getsource(cs.CrystalViewerSuite._bernevig_opted_in)
    assert "combo_bernevig_stacking" not in src
    assert "bernevig" in src
