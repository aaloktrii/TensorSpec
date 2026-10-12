# AI Interaction Guidelines for TensorSpec

When assisting with this repository, strictly adhere to the following rules:

1.  **Snippet-Only Output:** NEVER output the entire codebase or full script unless explicitly asked to generate a brand new file. Only output the exact classes, functions, or UI blocks that need to be updated or added.
2.  **Contextual Placement:** Always clearly state exactly *where* the provided code block should be inserted or what existing code it replaces (e.g., "Replace the `draw_structure` function" or "Insert this below line 42"). if you lose context, always ask what file to be uploaded for better reference.
3.  **No Silent Deletions:** Do not remove existing features, buttons, or imports unless specifically instructed to refactor them out. 
4.  **Acknowledge Roadmap:** Always refer back to `ROADMAP.md` to ensure UI additions fit into the planned Tabbed architecture. If the new path emerges during the development, always refer back to the roadmap.md and tell me to update which part into what. do not make a new roadmap from scratch.
5.  **Strict Modularity & Separation of Concerns:** Never write monolithic single-file suites. New features and refactored components must strictly separate logic into three distinct layers:
    * **Core Math & Physics Engine (`tensorspec/core/`):** Pure Python/NumPy/PyMatgen logic (e.g., symmetry parsing, Moiré math, ARPES momentum transformations). Zero GUI or plotting imports allowed.
    * **Rendering & Visualization Backends (`tensorspec/plotting/`):** Dedicated wrapper classes for PyVista, Matplotlib, or PyQtGraph engines.
    * **UI Controllers (`tensorspec/gui/`):** PySide6 layout definitions, widgets, and signal/slot connection routing.
    * if a long monolithic files need to be separated, always tell me which block to be moved where instead of giving me the whole code so I can follow the logic. when separating files, I want you to tell me what to copy from the old file and what to paste in the new file. I only want to move what I know exist in the old files so we dont lose any feature.
6.  **Hierarchical Data Architecture:** All multi-dimensional spectroscopic data containers must adopt the **Hierarchical Tree Model** (via `xarray.DataTree` aligned with NeXus/HDF5 standards). Never store disconnected arrays. Every data object must structure its nodes as:
    * `/raw`: Immutable experimental intensity matrices, hardware coordinates, and metadata (`attrs`).
    * `/processed`: Transformed datasets (e.g., $E, k$ space, drift-corrected PEEM stacks).
    * `/analysis`: Sub-nodes for mathematical fits (e.g., `/analysis/peakfit`, `/analysis/background`).
    * `/history`: Provenance log tracking all sequential operations and parameters applied to the tree.
7.  **Target Directory Blueprint:** Whenever generating new files or breaking down monolithic scripts, strictly organize code inside this folder structure (Qt GUI / `TensorSpec_GUI` reality as of 2026-10):
tensorspec/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── workspace.py          # CENTRAL MEMORY: Global dictionary/manager for all active loaded data
│   ├── data_tree.py          # Hierarchical xarray.DataTree structure & /history audit tracking
│   ├── crystallography.py    # Math engine: PyMatgen symmetry, Miller cleavage, CDW, Moiré strain
│   ├── kinematics.py         # Angle/energy to momentum space (k_parallel, k_z) conversions & photon momentum
│   ├── io/                   # Dedicated file loaders
│   │   ├── __init__.py
│   │   ├── peem_loaders.py   # TIF / ZIP / folder PEEM stacks
│   │   ├── xas_loaders.py    # 1D XAS/XMCD CSV/TXT (single or paired channels)
│   │   └── loaders/
│   │       └── maestro/      # MAESTRO HDF5 kinds (xy_fine_4d, focus_xy_fine_5d, …)
│   ├── dft_engine.py         # MAIN ROUTER: Routes calculation to chinook_tb or qe_generator
│   ├── dft/                  # Nested folder for separated DFT physics engines
│   │   ├── __init__.py
│   │   ├── chinook_tb.py     # Tight Binding & Slater-Koster math engine
│   │   └── qe_generator.py   # Quantum Espresso & Wannier90 input file generator
│   ├── arpes_engine.py       # MAIN ROUTER: Routes calculation to three_step.py or one_step/
│   ├── arpes/                # Nested folder for separated physics engines
│   │   ├── __init__.py
│   │   ├── three_step.py     # Option A: Classic 3-step phenomenological calculations
│   │   └── one_step/         # Option B: Advanced 1-step solver submodules
│   │       ├── __init__.py
│   │       ├── chinook_wrapper.py  # B1: Chinook TB + free-electron final state
│   │       ├── kmap_solver.py      # B2: Plane-wave FFT from real-space DFT orbitals (planned)
│   │       └── kkr_wrapper.py      # B3: SPR-KKR kkrspec / oscarpes
│   ├── peem/                 # PEEM math (no Qt)
│   │   ├── engine.py         # Pairing, drift, channel separation orchestration helpers
│   │   ├── bg.py             # Background models (also used by XAS 1D)
│   │   ├── roi.py            # ROI helpers for drift / NCC
│   │   └── sumrule.py        # XMCD sum-rule integrals (also used by XAS 1D)
│   ├── transport_engine.py   # PLANNED: magnetoresistance, Hall, R-T scaling
│   └── ml/                   # SSL models, clustering, training workers (no Qt)
├── plotting/
│   ├── __init__.py
│   ├── backends/              # Low-level rendering engines
│   │   ├── __init__.py
│   │   ├── matplotlib_engine.py # Safe CPU rendering for 1D lines & static 2D maps
│   │   ├── pyvista_engine.py    # Fast GPU rendering for 3D crystal structures & volumes
│   │   └── pyqtgraph_engine.py  # High-speed real-time 2D image rendering (optional)
│   └── viewers/               # Reusable Qt widgets for suites to embed (where present)
└── gui/
    ├── __init__.py
    ├── main_browser.py       # THE BIG GUI: Global Data Workspace Explorer & Suite Launcher Ribbon
    ├── services/             # Non-HTTP service layer between panels and core
    │   ├── peem_service.py
    │   └── xas_service.py    # XAS uses shared peem BG/sum-rule (no core/xas_engine.py yet)
    ├── components/           # Reusable, isolated UI panels
    │   ├── __init__.py
    │   ├── crystal_panel.py
    │   ├── dft_panels.py / qe_generator_panel.py / sprkkr_panels.py
    │   ├── arpes_panel.py
    │   ├── peem_panel.py / xas_panel.py
    │   ├── data_viewer_panel.py   # Universal N-D viewer + ML domain overlays
    │   └── ml_tabs/               # Extracted ML suite tabs
    ├── ml/                   # MaestroAI workers, warehouse, session helpers
    └── suites/               # Independent roadmap suites + ML
        ├── crystal_suite.py
        ├── dft_suite.py
        ├── arpes_suite.py
        ├── peem_suite.py
        ├── xas_suite.py
        ├── transport_suite.py    # Shell until transport_engine ships
        └── ml_suite.py

8. **ARPES Multi-Engine Protocol**: 
   When writing physics solvers under `core/arpes/`, never let solver-specific parameters bleed into the main UI. 
   The `arpes_engine.py` must act as a unified Factory Router. It receives a configuration dictionary from the GUI containing the model choice (A, B1, B2, B3) along with experimental variables, routes it to the designated submodule, and parses the output back into an xarray.DataTree structure under `/simulated` .