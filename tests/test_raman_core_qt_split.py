"""Guards for the Raman suite layering (sandy_rule rule 5): core has no Qt and no plotting,
plotting has no Qt."""
from __future__ import annotations

import re
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "tensorspec"
CORE = [*(PKG / "core" / "raman").rglob("*.py"), PKG / "core" / "raman_engine.py",
        PKG / "core" / "io" / "raman_loaders.py"]
QT = ("PySide6", "PyQt5", "PyQt6")


def _hits(paths, modules):
    """Import statements (import X / from X import ...) of any of ``modules`` in ``paths``."""
    pat = re.compile(r"^\s*(?:from|import)\s+(%s)\b" % "|".join(re.escape(m) for m in modules), re.M)
    out = []
    for path in paths:
        out += [f"{path.name}: {m}" for m in pat.findall(path.read_text(encoding="utf-8"))]
    return out


def test_raman_core_has_no_qt_or_plotting():
    assert _hits(CORE, QT + ("matplotlib", "mpl_toolkits", "tensorspec.plotting", "tensorspec.gui")) == []


def test_raman_matplotlib_engine_has_no_qt():
    assert _hits([PKG / "plotting" / "backends" / "raman_matplotlib_engine.py"], QT) == []


def test_raman_service_has_no_qt():
    assert _hits([PKG / "gui" / "services" / "raman_service.py"], QT) == []


def test_guard_sees_imports():
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write("import numpy\nfrom PySide6 import QtCore\n    import matplotlib.pyplot as plt\n")
    assert sorted(_hits([Path(fh.name)], QT + ("matplotlib",))) == [
        f"{Path(fh.name).name}: PySide6", f"{Path(fh.name).name}: matplotlib"]
