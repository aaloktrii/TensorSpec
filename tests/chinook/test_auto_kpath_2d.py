"""2D Auto-Detect k-path keeps HighSymmKpath (incl. A/L/H for quasi-bulk)."""
from pathlib import Path

import pytest
from pymatgen.core import Structure

from tensorspec.core.dft.chinook_tb import ChinookTightBindingEngine


def test_graphene_auto_kpath_has_gmk():
    cif = Path(__file__).resolve().parents[2] / "tensorspec" / "cif_file" / "graphene.cif"
    if not cif.is_file():
        pytest.skip("graphene.cif missing")
    eng = ChinookTightBindingEngine()
    eng.crystal_structure = Structure.from_file(str(cif))
    pts, labels = eng.get_auto_kpath()

    def is_g(lab: str) -> bool:
        s = str(lab)
        return "Gamma" in s or s in ("Γ", r"$\Gamma$", r"\Gamma")

    assert any(is_g(l) for l in labels)
    assert any(str(l) == "K" for l in labels)
    assert any(str(l) == "M" for l in labels)
    assert len(pts) == len(labels)
    # Quasi-bulk graphene (large c) may also list A/L/H — that is OK.
