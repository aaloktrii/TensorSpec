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
    views = list((PKG / "plotting" / "backends" / "raman_views").glob("*.py"))
    assert views
    assert _hits([PKG / "plotting" / "backends" / "raman_matplotlib_engine.py", *views], QT) == []


def test_raman_tabs_do_not_draw():
    """Rendering lives in plotting/: the Raman tabs import no matplotlib themselves."""
    tabs = [*(PKG / "gui" / "components" / "raman_tabs").glob("*.py"),
            PKG / "gui" / "components" / "raman_panel.py"]
    assert _hits(tabs, ("matplotlib", "mpl_toolkits")) == []


def test_raman_service_has_no_qt():
    assert _hits([PKG / "gui" / "services" / "raman_service.py"], QT) == []


def test_guard_sees_imports():
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write("import numpy\nfrom PySide6 import QtCore\n    import matplotlib.pyplot as plt\n")
    assert sorted(_hits([Path(fh.name)], QT + ("matplotlib",))) == [
        f"{Path(fh.name).name}: PySide6", f"{Path(fh.name).name}: matplotlib"]


#: What the Raman frontend (Qt tabs) may take from core: constants, data types, presets, text
#: parsing and display labels. Everything else goes through RamanService.
FRONTEND_MAY_IMPORT = {
    "tensorspec.core.raman.constants": None,                      # any name
    "tensorspec.core.raman.ranges": None,
    "tensorspec.core.raman.presets": {"PRESETS"},
    "tensorspec.core.raman.fit_model": {"FitModel", "as_band", "band"},
    "tensorspec.core.io.raman_loaders": {"Spectrum", "MapFile", "short_label"},
    "tensorspec.core.raman.classification": {"label_of", "windows_for", "LAYER_THICK"},
    "tensorspec.core.raman.voigt_2d": {"HAVE_LMFIT"},
}


def test_raman_frontend_reaches_the_science_only_through_the_service():
    import ast
    files = [*(PKG / "gui" / "components" / "raman_tabs").glob("*.py"),
             PKG / "gui" / "components" / "raman_panel.py", PKG / "gui" / "suites" / "raman_suite.py"]
    bad = []
    for path in files:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("tensorspec.core"):
                allowed = FRONTEND_MAY_IMPORT.get(node.module, set())
                names = {a.name for a in node.names}
                if allowed is not None and not names <= allowed:
                    bad.append(f"{path.name}: {node.module} {sorted(names - allowed)}")
            elif isinstance(node, ast.Import):
                bad += [f"{path.name}: import {a.name}" for a in node.names
                        if a.name.startswith("tensorspec.core")]
    assert bad == []
