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

### External codes & data (you install)

| Need | What to do |
|------|------------|
| Pseudopotentials (QE) | Download UPFs into `pseudo/` — see [`pseudo/README.md`](pseudo/README.md). Or set `TENSORSPEC_PSEUDO_DIR`. |
| Quantum ESPRESSO / Wannier90 | Install binaries on your PATH (or remote cluster). |
| SPR-KKR | Install locally or on your cluster; not vendored in this repo. |
| GrizzlyME (optional CUDA ARPES) | Separate install; not required for core GUI. |
| Cluster SSH | Create **your** local `tensorspec_clusters.json` (gitignored). Never commit passwords/hosts. |

A fresh clone has **no** UPF library and **no** lab servers — that is intentional.

## Launch

```bash
./TensorSpec_env/bin/python tensorspec/gui/main_browser.py
```

The Workspace Browser ribbon opens independent suites (Crystal, DFT, ARPES, PEEM, XAS, Transport, Machine Learning) plus Remote Compute.

## Suite status (GUI)

| Suite | Status | Notes |
|-------|--------|--------|
| Crystal Viewer | Live | Tabs View (Structure/Camera/Export) · CDW · Stack · BZ; CIF, supercells, twist/Bernevig |
| DFT | Live | Chinook TB, QE/Wannier helpers, SPR-KKR SCF |
| ARPES | Live (sim) | Kinematics + three-step / Chinook B1 / SPR-KKR B3; MAESTRO loaders partial |
| PEEM | Live | Tab *XPEEM map*: one loader for scan tables, .dat and .P3B data (names not needed), stack clean-up, drift (incl. SPEEM Igor methods), stack + asymmetry, image analysis, ROIs, SDA domain maps, in the TensorSpec data viewer. Tab *Stack · BG · sum rule*: TIF/ZIP load, pair CP/CM·LH/LV, NCC drift, BG, XMCD sum rule |
| XAS / XMCD | Live | 1D spectra; shared PEEM BG + sum-rule core |
| Transport | Shell | Demo R(H)/Hall/R(T) UI; `transport_engine` not built |
| Machine Learning | Live | SSL / clustering / AL; domain maps → shared DataViewer |
| Data Viewer | Live | N-D slicing + ML `domains_*` overlay layers |

## Architecture (short)

* **Core** (`tensorspec/core/`): physics, loaders, workspace, DataTree — no Qt imports
* **GUI** (`tensorspec/gui/`): `main_browser.py`, suite wrappers, panels, `services/`, `ml/`
* **Plotting** (`tensorspec/plotting/`): Matplotlib / PyVista backends

Agent / modularity rules: `sandy_rule.md`. Checklist roadmap: `roadmap.md`.
