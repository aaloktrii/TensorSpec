"""Guards for the Raman suite layering (sandy_rule rule 5): core has no Qt and no plotting,
plotting has no Qt."""
from __future__ import annotations

from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "tensorspec"
CORE = [*(PKG / "core" / "raman").rglob("*.py"), PKG / "core" / "raman_engine.py",
        PKG / "core" / "io" / "raman_loaders.py"]
QT = ("PySide6", "PyQt", "QThread", "from Qt")


def _hits(paths, needles):
    out = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        out += [f"{path.name}: {n}" for n in needles if n in text]
    return out


def test_raman_core_has_no_qt_or_plotting():
    assert _hits(CORE, QT + ("matplotlib", "tensorspec.plotting")) == []


def test_raman_figures_have_no_qt():
    assert _hits([PKG / "plotting" / "raman_figures.py"], QT) == []


def test_raman_service_has_no_qt():
    assert _hits([PKG / "gui" / "services" / "raman_service.py"], QT) == []
