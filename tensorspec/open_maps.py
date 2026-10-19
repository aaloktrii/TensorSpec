"""Open map files in the TensorSpec PEEM window.

    python -m tensorspec.open_maps FILE.nc [MORE.nc ...]      (or: scripts/tensorspec-open FILE.nc ...)

Each .nc file (batch maps.nc, analysis outputs, any NetCDF with 2-D variables; TIF stacks and PEEM
folders also work) becomes a dataset in the PEEM panel: the Map view shows its first asymmetry-type
map, the Show list every map in the file, and the panel's tools work on them (colour range,
profiles, histogram, FFT, ROIs, ROI statistics, domain map, export, 7. Compare with a reference).
Several files: pick one in 1. Load → Dataset, compare them in 7.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if args else 2
    missing = [a for a in args if not Path(a).exists()]
    if missing:
        print("not found: " + ", ".join(missing), file=sys.stderr)
        return 1
    from PySide6.QtWidgets import QApplication

    from tensorspec.gui.suites.peem_suite import PEEMSuite

    app = QApplication.instance() or QApplication(sys.argv[:1])
    win = PEEMSuite()
    win.resize(1300, 850)
    win.setWindowTitle("TensorSpec - PEEM Suite: " + ", ".join(Path(a).name for a in args)[:120])
    win.show()
    win.raise_()
    win.panel.open_files([str(Path(a).resolve()) for a in args])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
