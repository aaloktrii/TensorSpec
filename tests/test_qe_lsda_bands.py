"""LSDA QE bands split into spin up / spin down for the DFT plot and ARPES push."""
from pathlib import Path

from tensorspec.core.dft_engine import DFTEngineRouter
from tensorspec.gui.components.arpes_panel import ARPESPanel


def test_lsda_xml_splits_up_then_down(tmp_path: Path):
    xml = tmp_path / "bands.xml"
    xml.write_text(
        """<?xml version="1.0"?>
<data>
  <band_structure>
    <lsda>true</lsda>
    <nbnd_up>2</nbnd_up>
    <nbnd_dw>2</nbnd_dw>
    <two_fermi_energies>0.1 0.3</two_fermi_energies>
    <ks_energies>
      <k_point>0 0 0</k_point>
      <eigenvalues>1 2 3 4</eigenvalues>
    </ks_energies>
    <ks_energies>
      <k_point>0.5 0 0</k_point>
      <eigenvalues>1.5 2.5 3.5 4.5</eigenvalues>
    </ks_energies>
  </band_structure>
</data>
"""
    )
    k_dist, eigenvals, fermi, spins = DFTEngineRouter().parse_qe_xml(str(xml))
    ha = 27.211386
    assert len(k_dist) == 2
    assert eigenvals.shape == (2, 4)
    assert abs(fermi - 0.2 * ha) < 1e-6
    assert spins["axis"] == "z"
    assert spins["up"].shape == (2, 2)
    assert abs(spins["up"][0, 0] - ha) < 1e-6
    assert abs(spins["down"][0, 0] - 3 * ha) < 1e-6


def test_spin_channels_ride_into_arpes_npz():
    packed = ARPESPanel._spin_channel_npz(
        {"spin_up": [[0.0, 1.0]], "spin_down": [[0.1, 1.1]], "spin_axis": "z"}
    )
    assert packed["spin_up"].shape == (1, 2)
    assert packed["spin_down"].shape == (1, 2)
    assert str(packed["spin_axis"]) == "z"
    assert ARPESPanel._spin_channel_npz({}) == {}
