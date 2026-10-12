# TensorSpec (Desktop GUI)

**Branch:** `TensorSpec_GUI` — PySide6 workspace browser and analysis suites.

A general-purpose framework for N-dimensional spectroscopic analysis (ARPES, XAS/XMCD, PEEM, DFT/TB, and ML on hyperspectral maps). Data lives in an `xarray.DataTree` hierarchy (`/raw`, `/processed`, `/analysis`, `/history`) coordinated by a global in-memory workspace.

> The HTML / Einstein web app is a **separate** tree and branch. This README is for the **desktop GUI** only.

## Prerequisites

* **Python 3.11 or 3.12** (3.9 / 3.10 not supported)
* A virtual environment is recommended (`./TensorSpec_env/` in this clone)

## Install

```bash
cd /path/to/TensorSpec_GUI
python3.11 -m venv TensorSpec_env
source TensorSpec_env/bin/activate   # Windows: TensorSpec_env\Scripts\activate
pip install -r requirements.txt
```

Optional remote GPU / GrizzlyME CUDA ARPES: configure a local cluster JSON (example kept in the maintainer-only local `docs/` tree, not shipped on GitHub).

## Launch

```bash
./TensorSpec_env/bin/python tensorspec/gui/main_browser.py
```

The Workspace Browser ribbon opens independent suites (Crystal, DFT, ARPES, PEEM, XAS, Transport, Machine Learning) plus Remote Compute.

## Suite status (GUI)

| Suite | Status | Notes |
|-------|--------|--------|
| Crystal Viewer | Live | CIF, supercells, twist/Bernevig, BZ, CDW |
| DFT | Live | Chinook TB, QE/Wannier helpers, SPR-KKR SCF |
| ARPES | Live (sim) | Kinematics + three-step / Chinook B1 / SPR-KKR B3; MAESTRO loaders partial |
| PEEM | Live | TIF/ZIP load, pair CP/CM·LH/LV, drift, BG, XMCD sum rule |
| XAS / XMCD | Live | 1D spectra; shared PEEM BG + sum-rule core |
| Transport | Shell | Demo R(H)/Hall/R(T) UI; `transport_engine` not built |
| Machine Learning | Live | SSL / clustering / AL; domain maps → shared DataViewer |
| Data Viewer | Live | N-D slicing + ML `domains_*` overlay layers |

## Architecture (short)

* **Core** (`tensorspec/core/`): physics, loaders, workspace, DataTree — no Qt imports
* **GUI** (`tensorspec/gui/`): `main_browser.py`, suite wrappers, panels, `services/`, `ml/`
* **Plotting** (`tensorspec/plotting/`): Matplotlib / PyVista backends

Agent / modularity rules: `sandy_rule.md`. Checklist roadmap: `roadmap.md`.
