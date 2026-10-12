# Pseudopotentials (`pseudo/` / `Pseudo/`)

This folder is **not** shipped with TensorSpec on GitHub. Quantum ESPRESSO needs
element `.upf` (or `.UPF`) files that **you** download. The GUI/QE writer looks
here by default (`./pseudo`).

On macOS the default disk is often case-insensitive, so `pseudo/` and `Pseudo/`
may be the same directory. Prefer lowercase `pseudo/` to match the code.

## Quick start

1. Create the folder if missing:
   ```bash
   mkdir -p pseudo
   ```
2. Download UPF files for the elements in your structure (C, H, …).
3. Drop them into `pseudo/` (filenames usually start with the element symbol,
   e.g. `C.us.pbe....upf`, `C_FR.upf` for fully relativistic / SOC).
4. Optional: point elsewhere without copying files:
   ```bash
   export TENSORSPEC_PSEUDO_DIR=/path/to/your/upf/library
   ```

## Where to get potentials (pick one library and stay consistent)

| Source | Notes |
|--------|--------|
| [SSSP](https://www.materialscloud.org/discover/sssp/table/efficiency) (Materials Cloud) | Common “efficiency / precision” PBE sets for QE |
| [PseudoDojo](http://www.pseudo-dojo.org/) | ONCVPSP / PAW tables |
| [QE.org pseudopotentials](https://pseudopotentials.quantum-espresso.org/) | Official / community UPF collections |

Do **not** mix random libraries in one calculation unless you know the cutoffs
and valence match. For SOC runs you typically need FR / REL files (names often
contain `FR` or `rel`).

## What TensorSpec does with them

`tensorspec.core.dft.qe_generator` copies matching UPFs from this directory
(or `TENSORSPEC_PSEUDO_DIR`) into each job’s `out_dir/pseudo/` and writes
`pseudo_dir = './pseudo/'` in the QE input. If an element is missing, you get
an explicit error asking you to place a UPF here — that is expected on a fresh
clone.

## What not to do

- Do not commit `.upf` / `.UPF` / `.upf.gz` into git (size + license + privacy).
- Do not assume Sandy’s lab `Pseudo/` tree is available after `git clone`.
