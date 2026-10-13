"""W90 Dirac/semimetal midgap preferred over offset QE EF for plots."""
from pathlib import Path

import pytest

from tensorspec.core.dft.qe_fermi import (
    detect_plot_fermi_eV,
    detect_qe_fermi_eV,
    detect_w90_touch_fermi_eV,
)


def test_graphene_drive_prefers_midgap_over_qe_ef():
    wd = Path(__file__).resolve().parents[2] / "scratch" / "graphene_dft_drive"
    if not (wd / "wannier90_band.dat").is_file():
        pytest.skip("graphene_dft_drive bands missing")
    qe, qe_src = detect_qe_fermi_eV(str(wd))
    touch, touch_src = detect_w90_touch_fermi_eV(str(wd))
    assert touch is not None, "expected near-touch at K"
    assert abs(touch - qe) > 0.05
    # Must not pick deep Γ/A accidental degeneracy (~−5.4 eV on this drive).
    assert abs(touch - (-2.7913)) < 0.05, f"unexpected touch midgap {touch}"
    assert "K" in touch_src or "H" in touch_src
    plot_ef, plot_src = detect_plot_fermi_eV(str(wd))
    assert abs(plot_ef - touch) < 1e-6
    assert "midgap" in plot_src
    assert qe_src != "none"
