import os
import time
import numpy as np
import traceback
from pymatgen.core import Structure
from tensorspec.core.workspace import global_workspace

try:
    import chinook.build_lib as build_lib
    import chinook.klib as klib  # The officially documented module
except ImportError:
    build_lib = None
    klib = None

from tensorspec.core.arpes.one_step.chinook_arpes_kmesh import _grizzly_diagonalize_tb


def _load_hr_rows(w90_filepath, data_start, num_wann, hop_tol, *, chunk_rows=2_000_000):
    """Stream wannier90_hr.dat, keeping only rows that can survive the hop_tol filter.

    Returns (hop, r_idx, n_total): `hop` is (M, 7) float64 of the surviving rows in
    file order, `r_idx` the R-block index of each surviving row computed from its
    ORIGINAL position, and `n_total` the number of data rows in the file.

    Why: a 324-orbital spinor hr.dat is 324^2 * 1265 = 1.33e8 rows / 6.2 GB. The old
    np.loadtxt over the whole file needed ~7.4 GB for the result array alone, plus
    several times that transiently, and ran for hours. Typically ~99% of rows are
    below hop_tol, so we filter per chunk and never hold more than chunk_rows at once.

    Rows kept: |Re| > hop_tol or |Im| > hop_tol, PLUS every on-site row (R = 0 and
    i == j) whatever its magnitude, because the caller shifts those by -E_F afterwards
    and a tiny raw value there becomes a large one. Degeneracy weights are >= 1, so
    dividing by them can only shrink |t|; the kept set is therefore a superset of what
    the caller's own |t_ij| > hop_tol filter keeps, and the final result is identical
    to reading the whole file at once.
    """
    n_block = int(num_wann) * int(num_wann)
    kept, kept_r = [], []
    state = {"row0": 0}

    def _take(block):
        block = np.asarray(block, dtype=np.float64)
        if block.ndim == 1:
            block = block.reshape(1, -1)
        if block.size == 0:
            return
        if block.shape[1] < 7:
            raise ValueError(f"Unexpected wannier90_hr.dat columns: {block.shape}")
        idx = state["row0"] + np.arange(block.shape[0], dtype=np.int64)
        state["row0"] += block.shape[0]
        sel = (np.abs(block[:, 5]) > hop_tol) | (np.abs(block[:, 6]) > hop_tol)
        sel |= (
            (block[:, 0] == 0)
            & (block[:, 1] == 0)
            & (block[:, 2] == 0)
            & (block[:, 3] == block[:, 4])
        )
        if sel.any():
            kept.append(block[sel, :7])
            kept_r.append(idx[sel] // n_block)

    try:
        import pandas as pd

        # sep=r"\s+" is special-cased to the C whitespace tokenizer (fast, leading
        # blanks handled); the file is 7 numeric columns with no header.
        for chunk in pd.read_csv(
            w90_filepath,
            skiprows=data_start,
            sep=r"\s+",
            header=None,
            usecols=range(7),
            dtype=np.float64,
            chunksize=chunk_rows,
        ):
            _take(chunk.to_numpy(dtype=np.float64))
    except ImportError:
        from itertools import islice

        with open(w90_filepath, "r") as fh:
            for _ in range(data_start):
                fh.readline()
            while True:
                lines = list(islice(fh, chunk_rows))
                if not lines:
                    break
                _take(np.loadtxt(lines))

    n_total = state["row0"]
    if not kept:
        return np.empty((0, 7), dtype=np.float64), np.empty(0, dtype=np.int64), n_total
    return np.vstack(kept), np.concatenate(kept_r), n_total

class ChinookTightBindingEngine:
    """
    Core engine for Tight Binding calculations using the Chinook backend.
    Translates PyMatgen structures into Chinook TB models and calculates E(k).
    """
    def __init__(self):
        self.crystal_structure = None
        self.A_qe = None
        self._w90_parse_cache = {}
        self._source_cache_key = None
        self._cached_tb_dict = None
        self._cached_basis_args = None
        self._tb_build_cache_key = None
        self._tb_build_cache = None  # (basis, tb_model)
        self._tb_debug = os.environ.get("TENSORSPEC_TB_DEBUG", "").lower() in (
            "1",
            "true",
            "yes",
        )
        self._last_wannier_basis_quality = None

        # Materials Database for Slater-Koster hopping parameters
        self.materials_db = {
            'C': { # PyMatgen reduces C2 down to C
                'orbitals': ['C', 'C'],
                'sk_base': {
                    'C-C_nn': -2.7,      # Standard graphene t1 is ~ -2.7 eV
                    'C-C_nnn': -0.2,     # t2 is ~ -0.2 eV
                    'C-C_third': 0.0
                },
                # Carbon: p-orbitals form the Dirac cone at E=0, s-orbitals are deep
                'onsite': {'0': -8.0, '1': 0.0} 
            },
            'WTe2': {
                'orbitals': ['W', 'Te'],
                'sk_base': {
                    'W-W': -1.5, 'W-Te': -1.2, 'Te-Te_in_plane': -0.8,
                    'Te-Te_interlayer': -0.3 
                },
                'onsite': {'0': -10.0, '1': -2.0, '2': 0.0}
            },
            'TaIrTe4': {
                'orbitals': ['Ta', 'Ir', 'Te'],
                'sk_base': {'M-M': -1.3, 'M-Te': -1.1, 'Te-Te': -0.7}, # Proxy Slater-Koster bonds
                'onsite': {'0': -10.0, '1': -2.0, '2': 0.0} 
            },
            'MoTe2': {
                'orbitals': ['Mo', 'Te'],
                'sk_base': {'Mo-Mo': -1.4, 'Mo-Te': -1.2, 'Te-Te': -0.8},
                'onsite': {'0': -10.0, '1': -1.5, '2': 0.0}
            },
            'VTe2': {
                'orbitals': ['V', 'Te'],
                'sk_base': {'V-V': -1.3, 'V-Te': -1.1, 'Te-Te': -0.7},
                # VTe2: V d-bands at Fermi level, Te p-bands shifted down
                'onsite': {'0': -10.0, '1': -2.0, '2': 0.0}
            }
        }

    def load_structure_from_workspace(self, variable_name: str) -> bool:
        data = global_workspace.pull_crystal_structure(variable_name)
        if isinstance(data, Structure):
            # --- CRITICAL FIX: DO NOT STANDARDIZE THE LATTICE! ---
            # PyMatgen rotates the crystal, which misaligns the BZ and makes the K-path miss the Dirac cone!
            self.crystal_structure = data
            return True
        return False

    def get_default_hopping(self, formula: str) -> dict:
        """Returns the default hopping params for a formula, or fallback WTe2 values."""
        if formula in self.materials_db:
            return self.materials_db[formula]['sk_base']
        return {'M-M': -1.5, 'M-X': -1.2, 'X-X': -0.8} # Generic proxy names

    def _get_orbital_basis(self, element_symbol):
        """
        Dynamically returns the orbital basis based on the element.
        Format: [n][l][projection]
        CRITICAL: Must strictly match Wannier90's internal orbital ordering!
        """
        from pymatgen.core import Element
        
        # Strip out any oxidation states or numbers (e.g., convert "Te2-" to "Te")
        clean_symbol = ''.join([c for c in element_symbol if c.isalpha()])
        el = Element(clean_symbol)
        
        # Determine the principal quantum number (n) based on the periodic table row
        n = el.row
        
        # Wannier90 strictly orders s;p;d projections as: s, pz, px, py, dz2, dxz, dyz, dx2-y2, dxy
        s_orbs = [f"{n}0"]
        p_orbs = [f"{n}1z", f"{n}1x", f"{n}1y"]
        # d-orbitals are always (n-1) in the valence shell (e.g., 6s -> 5d)
        d_orbs = [f"{n-1}2ZR", f"{n-1}2xz", f"{n-1}2yz", f"{n-1}2XY", f"{n-1}2xy"]
        
        # Mirror the exact projection logic from qe_generator.py
        if el.is_transition_metal or el.number > 30:
            return s_orbs + p_orbs + d_orbs
        else:
            return s_orbs + p_orbs

    @staticmethod
    def assess_wannier_basis_quality(
        flat_pos, num_wann: int, basis_source: str = "unknown"
    ) -> dict:
        """Heuristic ARPES readiness check from WF centre spread."""
        pos = np.asarray(flat_pos, dtype=float)
        if pos.size == 0:
            return {
                "ok": False,
                "unique_centers": 0,
                "num_wann": int(num_wann),
                "basis_source": basis_source,
                "min_ok": 0,
                "message": "No orbital positions for Wannier basis.",
            }

        n_unique = len(np.unique(np.round(pos.reshape(-1, 3), decimals=4), axis=0))
        num_wann = int(num_wann)
        min_ok = (
            max(20, num_wann // 8) if num_wann >= 50 else max(5, num_wann // 4)
        )

        ok = n_unique >= min_ok
        if basis_source == "cif_fallback" and num_wann >= 50:
            ok = False
        elif basis_source == "wout_cif" and n_unique < min_ok:
            ok = False

        message = None
        if not ok:
            message = (
                f"ARPES basis suspect: {n_unique} unique WF centres for "
                f"num_wann={num_wann} (source={basis_source}; need ≥{min_ok}). "
                "Place wannier90.win + wannier90.wout beside hr.dat, load matching "
                "bulk CIF, then Prepare TB for ARPES. Circle-ish ARPES often means "
                "this mismatch."
            )

        return {
            "ok": ok,
            "unique_centers": n_unique,
            "num_wann": num_wann,
            "basis_source": basis_source,
            "min_ok": min_ok,
            "message": message,
        }

    @staticmethod
    def _resolve_wannier_wout_path(work_dir: str) -> str:
        """Prefer wannier90.wout — never scf.out/nscf.out (also end in .out)."""
        preferred = os.path.join(work_dir, "wannier90.wout")
        if os.path.isfile(preferred):
            return preferred
        for name in sorted(os.listdir(work_dir)):
            if name.endswith(".wout"):
                return os.path.join(work_dir, name)
        return preferred

    @staticmethod
    def _resolve_wannier_win_path(work_dir: str, w90_filepath: str) -> str:
        """Only look beside hr.dat — never a sibling QE folder (friend's Wannier safety)."""
        candidates = [
            os.path.join(work_dir, "wannier90.win"),
            os.path.join(
                work_dir,
                os.path.basename(w90_filepath).replace("_hr.dat", ".win"),
            ),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        return candidates[0]

    @staticmethod
    def _parse_wannier_final_centers(wout_path: str, num_wann: int) -> list:
        """WF centres from the Final State block only (skip Initial State duplicate)."""
        if not os.path.isfile(wout_path):
            return []
        try:
            with open(wout_path, "r") as f:
                lines = f.readlines()
        except OSError as exc:
            print(f"Failed to read Wannier centers: {exc}")
            return []

        start = 0
        for idx, line in enumerate(lines):
            if line.strip() == "Final State":
                start = idx + 1
                break

        centers = []
        for line in lines[start:]:
            if "WF centre and spread" not in line:
                continue
            parts = line.split("(")[1].split(")")[0].split(",")
            centers.append([float(x.strip()) for x in parts])

        if len(centers) >= num_wann:
            return centers[-num_wann:]
        return centers

    @staticmethod
    def _parse_wannier_win_metadata(win_path: str) -> dict:
        meta = {
            "spinors": False,
            "atoms": [],
            "projections": {},
        }
        if not os.path.isfile(win_path):
            return meta

        section = None
        with open(win_path, "r") as f:
            for raw in f:
                line = raw.strip()
                if line.startswith("spinors"):
                    meta["spinors"] = "true" in line.lower()
                elif line == "begin projections":
                    section = "projections"
                    continue
                elif line == "end projections":
                    section = None
                    continue
                elif line == "begin atoms_frac":
                    section = "atoms_frac"
                    continue
                elif line == "end atoms_frac":
                    section = None
                    continue

                if section == "projections" and line and ":" in line:
                    elem, shells = line.split(":", 1)
                    meta["projections"][elem.strip()] = [
                        s.strip() for s in shells.split(";") if s.strip()
                    ]
                elif section == "atoms_frac" and line:
                    meta["atoms"].append(line.split()[0])

        return meta

    def _orbitals_for_projection_shells(self, element_symbol: str, shells: list) -> list:
        full = self._get_orbital_basis(element_symbol)
        s_orbs = full[0:1]
        p_orbs = full[1:4] if len(full) > 3 else []
        d_orbs = full[4:9] if len(full) > 4 else []
        selected = []
        for shell in shells:
            key = shell.lower()
            if key == "s":
                selected.extend(s_orbs)
            elif key == "p":
                selected.extend(p_orbs)
            elif key == "d":
                selected.extend(d_orbs)
        return selected or list(full)

    def _build_wannier_flat_basis(
        self, win_path: str, centers: list, num_wann: int, use_soc: bool
    ):
        """Build Chinook basis rows in Wannier90 hr.dat index order."""
        from pymatgen.core import Element

        meta = self._parse_wannier_win_metadata(win_path)
        spinors = bool(meta.get("spinors"))
        flat_atoms = []
        flat_Z = {}
        flat_pos = []
        flat_orbs = []

        if meta.get("atoms") and meta.get("projections"):
            idx = 0
            for atom_sym in meta["atoms"]:
                clean = "".join(c for c in atom_sym if c.isalpha())
                shells = meta["projections"].get(clean) or meta["projections"].get(atom_sym)
                if not shells:
                    for key, val in meta["projections"].items():
                        if key.lower() == clean.lower():
                            shells = val
                            break
                if not shells:
                    shells = ["s", "p", "d"]
                z_num = int(Element(clean).Z)
                spatial_orbs = self._orbitals_for_projection_shells(clean, shells)
                repeats = 2 if spinors else (2 if use_soc else 1)
                for orb in spatial_orbs:
                    for _ in range(repeats):
                        flat_atoms.append(idx)
                        flat_Z[idx] = z_num
                        if idx < len(centers):
                            flat_pos.append(np.array(centers[idx], dtype=float))
                        else:
                            flat_pos.append(np.zeros(3, dtype=float))
                        flat_orbs.append([orb])
                        idx += 1
                        if idx >= num_wann:
                            break
                    if idx >= num_wann:
                        break
                if idx >= num_wann:
                    break

            if idx == num_wann:
                print(
                    f"Wannier basis from {os.path.basename(win_path)}: "
                    f"{num_wann} orbitals ({'spinor' if spinors else 'scalar'}), "
                    f"{len(set(tuple(np.round(p, 4)) for p in flat_pos))} unique WF centres",
                    flush=True,
                )
                return flat_atoms, flat_Z, flat_pos, flat_orbs, spinors

        return None

    def _build_cif_basis_with_wout_centers(
        self, centers: list, num_wann: int, use_soc: bool
    ):
        """
        Friend's bundle (CIF + hr + wout, no .win): CIF s/p/d labels + wout WF centres.
        Index order may still differ from Wannier90 without .win — ask for .win if ARPES looks wrong.
        """
        if not self.crystal_structure or len(centers) < num_wann:
            return None

        spatial = []
        for site in self.crystal_structure:
            for orb in self._get_orbital_basis(site.species_string):
                spatial.append((int(site.specie.number), orb))

        n_spatial = len(spatial)
        if n_spatial == 0:
            return None

        spinors = num_wann == 2 * n_spatial
        if spinors:
            expanded = []
            for z_num, orb in spatial:
                expanded.append((z_num, orb))
                expanded.append((z_num, orb))
            spatial = expanded
        elif num_wann != n_spatial:
            return None

        flat_atoms = []
        flat_Z = {}
        flat_pos = []
        flat_orbs = []
        for idx, (z_num, orb) in enumerate(spatial):
            flat_atoms.append(idx)
            flat_Z[idx] = z_num
            flat_pos.append(np.array(centers[idx], dtype=float))
            flat_orbs.append([orb])

        print(
            f"Wannier basis (CIF + wout centres, no .win): {num_wann} orbitals "
            f"({'spinor' if spinors else 'scalar'}), "
            f"{len(set(tuple(np.round(p, 4)) for p in flat_pos))} unique WF centres. "
            "Orbital order is guessed from CIF — add wannier90.win for exact mapping.",
            flush=True,
        )
        return flat_atoms, flat_Z, flat_pos, flat_orbs, spinors

    def export_chinook_dictionary(self, shells=None, onsite_e=0.0, use_soc=False, soc_strength=0.5, tb_mode="Slater-Koster (Rigorous)", orbital_shifts=None):
        if not self.crystal_structure:
            raise ValueError("No structure loaded in DFT engine to export.")

        basis_vectors = self.crystal_structure.lattice.matrix.tolist()
        
        spin_dict = {'bool': False, 'soc': False}
        if use_soc:
            # We are using Native Chinook SOC Generation!
            spin_dict = {'bool': True, 'soc': True, 'lam': {i: soc_strength for i in range(len(self.crystal_structure))}}

        # If user selected Simple Scalar but enabled SOC, we MUST override to SK to prevent the NoneType crash
        if "Scalar" in tb_mode and not use_soc:
            # --- OLD ISOTROPIC LIST MODE (FAST, NO SOC) ---
            atom_orb_indices = {}
            global_idx = 0
            for i, site in enumerate(self.crystal_structure):
                atom_orb_indices[i] = []
                for _ in self._get_orbital_basis(site.species_string):
                    atom_orb_indices[i].append(global_idx)
                    global_idx += 1

            explicit_hopping = []
            for i in range(len(self.crystal_structure)):
                for g_idx in atom_orb_indices[i]:
                    explicit_hopping.append([g_idx, g_idx, 0.0, 0.0, 0.0, complex(onsite_e, 0)])

            sorted_shells = sorted(shells, key=lambda x: x[1]) if shells else []
            for dR_a in [-1, 0, 1]:
                for dR_b in [-1, 0, 1]:
                    for dR_c in [-1, 0, 1]:
                        R_frac = np.array([dR_a, dR_b, dR_c])
                        R_cart = self.crystal_structure.lattice.get_cartesian_coords(R_frac)
                        for i in range(len(self.crystal_structure)):
                            for j in range(len(self.crystal_structure)):
                                if dR_a == 0 and dR_b == 0 and dR_c == 0 and i == j: continue  
                                dist = self.crystal_structure.get_distance(i, j, jimage=[dR_a, dR_b, dR_c])
                                t_val = 0.0
                                for t, r_max in sorted_shells:
                                    if dist <= r_max:
                                        t_val = t
                                        break  
                                if abs(t_val) < 1e-5: continue
                                for idx_a, idx_b in zip(atom_orb_indices[i], atom_orb_indices[j]):
                                    explicit_hopping.append([idx_a, idx_b, R_cart[0], R_cart[1], R_cart[2], complex(t_val, 0)])

            return {
                'type': 'list', 'list': explicit_hopping, 'H': explicit_hopping,
                'a': basis_vectors, 'cutoff': 100.0, 'renorm': 1.0, 'offset': 0.0,
                'tol': 1e-15, 'spin': spin_dict
            }

        # --- NATIVE CHINOOK SLATER-KOSTER MODE ---
        V_dict = {}
        
        formula = self.crystal_structure.composition.reduced_formula
        mat_props = self.materials_db.get(formula, self.materials_db['VTe2'])
        
        # Priority: UI Overrides -> Database -> Hardcoded defaults
        onsite_db = orbital_shifts if orbital_shifts else mat_props.get('onsite', {'0': -10.0, '1': -2.0, '2': 0.0})
        
        # 1. On-site energies dynamically mapped per angular momentum (l)
        for i, site in enumerate(self.crystal_structure):
            for orb in self._get_orbital_basis(site.species_string):
                n, l = orb[0], orb[1]
                
                # Fetch the specific orbital energy from the database, apply the global UI shift (onsite_e)
                base_energy = onsite_db.get(l, 0.0)
                V_dict[f"{i}{n}{l}"] = onsite_e + base_energy

        sorted_shells = sorted(shells, key=lambda x: x[1]) if shells else []
        cutoff_max = sorted_shells[-1][1] if sorted_shells else 10.0

        # 2. Map UI Distances to Pairwise SK Bond Strings
        for i, site_i in enumerate(self.crystal_structure):
            for j, site_j in enumerate(self.crystal_structure):
                if i == j:
                    try:
                        # Find distance to the closest adjacent unit cell for t2 self-interaction
                        neighbors = self.crystal_structure.get_neighbors(site_i, cutoff_max)
                        self_dists = [nn.nn_distance for nn in neighbors if nn.index == i and nn.nn_distance > 1e-4]
                        dist = min(self_dists) if self_dists else 100.0
                    except:
                        dist = self.crystal_structure.lattice.a
                else:
                    dist = self.crystal_structure.get_distance(i, j)

                t_val = 0.0
                for t, r_max in sorted_shells:
                    if dist <= r_max:
                        t_val = t
                        break

                if abs(t_val) < 1e-5: continue

                for orb_i in self._get_orbital_basis(site_i.species_string):
                    for orb_j in self._get_orbital_basis(site_j.species_string):
                        n_i, l_i = orb_i[0], orb_i[1]
                        n_j, l_j = orb_j[0], orb_j[1]
                        
                        # Populate Native Slater-Koster Bonds (Sigma, Pi, Delta)
                        V_dict[f"{i}{j}{n_i}{n_j}{l_i}{l_j}S"] = t_val * 1.5
                        if int(l_i) >= 1 and int(l_j) >= 1:
                            V_dict[f"{i}{j}{n_i}{n_j}{l_i}{l_j}P"] = t_val
                        if int(l_i) >= 2 and int(l_j) >= 2:
                            V_dict[f"{i}{j}{n_i}{n_j}{l_i}{l_j}D"] = t_val * 0.8

        print("\n--- CHINOOK DICTIONARY DEBUG ---")
        print(f"Mode: Native Slater-Koster")
        print(f"Total Atoms: {len(self.crystal_structure)}")
        print(f"V_dict Keys Generated: {len(V_dict)}")
        for key, val in V_dict.items(): print(f"  {key} : {val}")
        print("--------------------------------\n")

        return {
            'type': 'SK',               
            'V': V_dict,                
            'a': basis_vectors,                            # <--- Satisfies the Generic Top-Layer Validator
            'avec': np.array(basis_vectors, dtype=float),  # <--- Satisfies the Slater-Koster Math Engine
            'cutoff': float(cutoff_max),
            'renorm': 1.0,
            'offset': 0.0,
            'tol': 1e-15,
            'spin': spin_dict
        }

    def _structure_fingerprint(self):
        s = self.crystal_structure
        return (
            s.composition.reduced_formula,
            tuple(tuple(row) for row in s.lattice.matrix.tolist()),
        )

    def _w90_source_key(self, w90_filepath, use_soc, onsite_e, hop_tol=1e-6, qe_fermi=0.0):
        path = os.path.abspath(w90_filepath)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        return (
            path,
            mtime,
            use_soc,
            float(onsite_e),
            float(hop_tol),
            float(qe_fermi),
            self._structure_fingerprint(),
        )

    def _manual_source_key(
        self,
        custom_hopping,
        onsite_e,
        use_soc,
        soc_strength,
        cutoffs,
        tb_mode,
        orbital_shifts,
    ):
        hop_items = tuple(sorted((custom_hopping or {}).items()))
        shift_items = tuple(sorted((orbital_shifts or {}).items()))
        return (
            self._structure_fingerprint(),
            float(onsite_e),
            use_soc,
            float(soc_strength),
            tb_mode,
            hop_items,
            tuple(cutoffs or ()),
            shift_items,
        )

    def seed_w90_parsed(self, source_key, tb_dict, basis_args, A_qe=None):
        """Inject pre-parsed W90 TB (hybrid remote cache path)."""
        if A_qe is not None:
            self.A_qe = A_qe
        self._w90_parse_cache[source_key] = (tb_dict, basis_args)

    def _get_wannier_tb(
        self, w90_filepath, use_soc, onsite_e, hop_tol=1e-6, qe_fermi=0.0
    ):
        hop_tol = float(hop_tol)
        qe_fermi = float(qe_fermi)
        key = self._w90_source_key(
            w90_filepath, use_soc, onsite_e, hop_tol, qe_fermi=qe_fermi
        )
        if key in self._w90_parse_cache:
            tb_dict, basis_args = self._w90_parse_cache[key]
            num_wann = len(basis_args.get("atoms", [])) if basis_args else 0
            if num_wann:
                self._last_wannier_basis_quality = self.assess_wannier_basis_quality(
                    basis_args.get("pos", []),
                    num_wann,
                    (basis_args or {}).get("basis_source", "memory"),
                )
            return self._w90_parse_cache[key], key, True

        from tensorspec.core.dft.w90_tb_cache import load_parsed_tb, save_parsed_tb

        disk = load_parsed_tb(
            w90_filepath, use_soc, onsite_e, hop_tol, qe_fermi=qe_fermi
        )
        if disk is not None:
            tb_dict, basis_args, a_qe = disk
            if a_qe is not None:
                self.A_qe = a_qe
            self._wannier_spinors_flag = bool(
                (basis_args or {}).get("wannier_spinors", False)
            )
            num_wann = len(basis_args.get("atoms", [])) if basis_args else 0
            if num_wann:
                self._last_wannier_basis_quality = self.assess_wannier_basis_quality(
                    basis_args.get("pos", []),
                    num_wann,
                    (basis_args or {}).get("basis_source", "cached"),
                )
            parsed = (tb_dict, basis_args)
            self._w90_parse_cache[key] = parsed
            print(
                f"W90 TB disk cache HIT ({len(tb_dict.get('list', []))} hops, "
                f"hop_tol={hop_tol:g})",
                flush=True,
            )
            return parsed, key, True

        tb_dict, basis_args = self.export_wannier_dictionary(
            w90_filepath,
            use_soc,
            onsite_e,
            hop_tol=hop_tol,
            qe_fermi=qe_fermi,
        )
        save_parsed_tb(
            w90_filepath,
            use_soc,
            onsite_e,
            tb_dict,
            basis_args,
            getattr(self, "A_qe", None),
            hop_tol=hop_tol,
            qe_fermi=qe_fermi,
        )
        self._w90_parse_cache[key] = (tb_dict, basis_args)
        return (tb_dict, basis_args), key, False

    def _orbital_list_from_basis(self, basis):
        """Chinook gen_basis returns a dict with 'bulk'; TB_model may hold a list."""
        if isinstance(basis, dict):
            if "bulk" in basis:
                return list(basis["bulk"])
            if "orbitals" in basis:
                return list(basis["orbitals"])
            return []
        if isinstance(basis, (list, tuple)):
            return list(basis)
        return list(getattr(basis, "orbitals", []) or [])

    @staticmethod
    def _basis_looks_spinor(orbs_list) -> bool:
        """True when Wannier spinor doubling is interleaved (label pairs)."""
        n = len(orbs_list)
        if n < 2 or n % 2:
            return False
        checks = 0
        matches = 0
        for i in range(0, min(n, 64), 2):
            a = orbs_list[i]
            b = orbs_list[i + 1]
            la = getattr(a, "label", None)
            lb = getattr(b, "label", None)
            if la is None or lb is None:
                continue
            checks += 1
            if la == lb:
                matches += 1
        return checks > 0 and matches == checks

    def _assign_wannier_spins(self, orbs_list, *, use_soc: bool, wannier_spinors: bool):
        """Keep orbital.spin=+1 for Wannier spinor hr.dat (spin is in H, not ARPES tags).

        Chinook ARPES sets exp.spin=True if any orbital has spin=-1; GrizzlyME v1
        rejects that as 'spin-resolved'. SARPES uses --engine chinook, not Grizzly.
        """
        if not orbs_list:
            return
        if wannier_spinors or self._basis_looks_spinor(orbs_list):
            for b_obj in orbs_list:
                b_obj.spin = 1.0
            print(
                f"Wannier spinor hr.dat: {len(orbs_list)} orbitals, spin tag +1 each "
                f"(spinor physics in H matrix; GrizzlyME-compatible)",
                flush=True,
            )
            return
        if use_soc:
            half = len(orbs_list) // 2
            for idx, b_obj in enumerate(orbs_list):
                b_obj.spin = 1.0 if idx < half else -1.0

    def _build_chinook_tb(self, tb_dict, basis_args, w90_filepath, use_soc):
        if self._tb_debug:
            print("\n--- CHINOOK BUILD STEPS ---")
        basis = build_lib.gen_basis(basis_args)

        orbs_list = self._orbital_list_from_basis(basis)
        wannier_spinors = bool(
            getattr(self, "_wannier_spinors_flag", False)
            or (basis_args or {}).get("wannier_spinors", False)
        )
        if w90_filepath:
            self._assign_wannier_spins(
                orbs_list, use_soc=use_soc, wannier_spinors=wannier_spinors
            )
            self._wannier_spinors_flag = wannier_spinors or self._basis_looks_spinor(
                orbs_list
            )

        if self._tb_debug:
            print(
                f"1. Successfully built basis! "
                f"(Total Orbitals: {len(orbs_list)})"
            )
            print("\n--- DEEP ORBITAL DEBUG ---")
            print(f"Atomic Numbers (Z) passed: {basis_args['Z']}")
            print(f"Orbitals passed: {basis_args['orbs']}")
            print("Orbitals Chinook actually generated and kept:")
            for b_obj in orbs_list:
                try:
                    print(f"  Atom {getattr(b_obj, 'atom', '?')} -> {b_obj.__dict__}")
                except Exception:
                    print(f"  {b_obj}")
            print("--------------------------\n")

        tb_model = build_lib.gen_TB(basis, tb_dict)
        if self._tb_debug:
            print("2. Successfully built TB Hamiltonian!")
            print("---------------------------\n")

        self.basis = basis
        self.H_dict = tb_dict
        self.tb_model = tb_model
        return basis, tb_model

    def _diagonalize_tb(
        self,
        tb_model,
        k_points,
        *,
        need_eigenvectors: bool = True,
        use_soc: bool = False,
        w90_filepath=None,
        diag_engine: str = "chinook",
        diag_device: str = "cpu",
    ):
        """Diagonalize H(k). GrizzlyME only when explicitly requested in UI."""
        tb_model.Kobj = klib.kpath(k_points)
        nk = len(k_points)
        if self._tb_debug:
            print("3. Successfully built K-path!")
            print("\n--- K-PATH & BASIS DEBUG ---")
            print("K-points sample (First 3 points):")
            for kp in k_points[:3]:
                print(f"  {kp}")
            print("----------------------------\n")

        eonly = not need_eigenvectors
        engine = str(diag_engine or "chinook").strip().lower()
        device = str(diag_device or "cpu").strip().lower()

        if engine == "grizzly":
            if use_soc and not w90_filepath:
                raise ValueError(
                    "GrizzlyME band diag does not support native Chinook SOC. "
                    "Select Chinook, or use a Wannier90 spinor hr.dat."
                )
            try:
                import grizzly.hamiltonian  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "GrizzlyME is not installed (pip install grizzlyme). "
                    "Select Chinook in Band diag engine."
                ) from exc
            Eband, Evec = _grizzly_diagonalize_tb(
                tb_model, device, nk, Eonly=eonly
            )
            tb_model.Eband = Eband
            tb_model.Evec = Evec
            tag = f"grizzly/{device}"
            if eonly:
                tag += "+eonly"
            return tag

        if eonly:
            tb_model.solve_H(Eonly=True)
            tb_model.Evec = None
            return "chinook+eonly"
        tb_model.solve_H()
        return "chinook"

    def solve_bands(
        self,
        k_points,
        custom_hopping=None,
        onsite_e=0.0,
        use_soc=False,
        soc_strength=0.5,
        w90_filepath=None,
        cutoffs=None,
        tb_mode="Simple Scalar",
        orbital_shifts=None,
        need_eigenvectors: bool = True,
        diag_engine: str = "chinook",
        diag_device: str = "cpu",
        qe_fermi: float = 0.0,
    ):
        if build_lib is None or klib is None:
            raise ImportError("Chinook is not installed properly. Cannot calculate bands.")
        if not self.crystal_structure:
            raise ValueError("No structure loaded.")

        t_total = time.perf_counter()
        parse_hit = build_hit = False

        if not w90_filepath:
            self.A_qe = None

        if w90_filepath:
            (tb_dict, basis_args), source_key, parse_hit = self._get_wannier_tb(
                w90_filepath,
                use_soc,
                onsite_e,
                hop_tol=1e-6,
                qe_fermi=qe_fermi,
            )
        else:
            source_key = self._manual_source_key(
                custom_hopping,
                onsite_e,
                use_soc,
                soc_strength,
                cutoffs,
                tb_mode,
                orbital_shifts,
            )
            if source_key == self._source_cache_key and self._cached_tb_dict is not None:
                tb_dict, basis_args = self._cached_tb_dict, self._cached_basis_args
                parse_hit = True
            else:
                shells = []
                if custom_hopping:
                    distances = cutoffs if cutoffs else [1.6, 2.6, 3.1, 4.5]
                    for i, (key, t_val) in enumerate(custom_hopping.items()):
                        r_max = distances[i] if i < len(distances) else 10.0
                        shells.append((t_val, r_max))

                tb_dict = self.export_chinook_dictionary(
                    shells=shells,
                    onsite_e=onsite_e,
                    use_soc=use_soc,
                    soc_strength=soc_strength,
                    tb_mode=tb_mode,
                    orbital_shifts=orbital_shifts,
                )
                basis_args = {
                    "atoms": list(range(len(self.crystal_structure))),
                    "Z": {
                        i: site.specie.number
                        for i, site in enumerate(self.crystal_structure)
                    },
                    "pos": [
                        np.array(site.coords, dtype=float)
                        for site in self.crystal_structure
                    ],
                    "orbs": [
                        self._get_orbital_basis(site.species_string)
                        for site in self.crystal_structure
                    ],
                    "spin": tb_dict.get("spin", {"bool": False}),
                }
                self._source_cache_key = source_key
                self._cached_tb_dict = tb_dict
                self._cached_basis_args = basis_args

        t_parse = time.perf_counter()

        build_key = source_key
        if build_key == self._tb_build_cache_key and self._tb_build_cache is not None:
            basis, tb_model = self._tb_build_cache
            build_hit = True
        else:
            try:
                basis, tb_model = self._build_chinook_tb(
                    tb_dict, basis_args, w90_filepath, use_soc
                )
                self._tb_build_cache_key = build_key
                self._tb_build_cache = (basis, tb_model)
            except Exception as e:
                print("\n!!! FATAL CHINOOK CRASH !!!")
                traceback.print_exc()
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                raise RuntimeError(f"Chinook crashed during initialization: {e}") from e

        t_build = time.perf_counter()
        backend = self._diagonalize_tb(
            tb_model,
            k_points,
            need_eigenvectors=need_eigenvectors,
            use_soc=use_soc,
            w90_filepath=w90_filepath,
            diag_engine=diag_engine,
            diag_device=diag_device,
        )
        t_diag = time.perf_counter()

        parse_s = t_parse - t_total
        build_s = t_build - t_parse
        diag_s = t_diag - t_build
        total_s = t_diag - t_total
        cache_bits = []
        if parse_hit:
            cache_bits.append("parse")
        if build_hit:
            cache_bits.append("build")
        cache_tag = f" cache={'+'.join(cache_bits)}" if cache_bits else ""
        print(
            f"DFT bands ({backend}): {len(k_points)} k-pts, "
            f"parse={parse_s:.2f}s build={build_s:.2f}s diag={diag_s:.2f}s "
            f"total={total_s:.2f}s{cache_tag}"
        )

        eigenvalues = np.real(tb_model.Eband)
        eigenvectors = tb_model.Evec if need_eigenvectors else None

        raw_labels = []
        for site in self.crystal_structure:
            for orb_str in self._get_orbital_basis(site.species_string):
                if orb_str.endswith("0"):
                    orb_name = "s"
                elif orb_str[1] == "1":
                    orb_name = "p" + orb_str[2:]
                elif orb_str[1] == "2":
                    if orb_str.endswith("ZR"):
                        orb_name = "dz2"
                    elif orb_str.endswith("XY"):
                        orb_name = "dx2-y2"
                    else:
                        orb_name = "d" + orb_str[2:]
                raw_labels.append(f"{site.species_string}_{orb_name}")

        if use_soc and not w90_filepath:
            orb_labels = [lbl + "_up" for lbl in raw_labels] + [
                lbl + "_dn" for lbl in raw_labels
            ]
        else:
            orb_labels = raw_labels

        return eigenvalues, eigenvectors, orb_labels

    def prepare_tb_for_arpes(
        self,
        w90_filepath,
        *,
        use_soc: bool = False,
        onsite_e: float = 0.0,
        hop_tol: float = 1e-4,
        quick_diag: bool = True,
        qe_fermi: float = 0.0,
    ):
        """
        Parse + build TB for ARPES workspace push — skip full k-path diagonalization.

        hop_tol defaults stricter than band-plot path so huge third-party hr.dat
        (tens of millions of hops) becomes tractable.
        """
        if build_lib is None or klib is None:
            raise ImportError("Chinook is not installed properly.")
        if not self.crystal_structure:
            raise ValueError("No structure loaded.")
        if not w90_filepath:
            raise ValueError("Wannier90 hr.dat required for Prepare TB for ARPES.")

        hop_tol = float(hop_tol)
        t_total = time.perf_counter()
        (tb_dict, basis_args), source_key, parse_hit = self._get_wannier_tb(
            w90_filepath,
            use_soc,
            onsite_e,
            hop_tol=hop_tol,
            qe_fermi=qe_fermi,
        )
        t_parse = time.perf_counter()

        build_hit = False
        if source_key == self._tb_build_cache_key and self._tb_build_cache is not None:
            basis, tb_model = self._tb_build_cache
            build_hit = True
        else:
            basis, tb_model = self._build_chinook_tb(
                tb_dict, basis_args, w90_filepath, use_soc
            )
            self._tb_build_cache_key = source_key
            self._tb_build_cache = (basis, tb_model)
        t_build = time.perf_counter()

        eigenvalues = None
        if quick_diag:
            gamma = np.array([[0.0, 0.0, 0.0]], dtype=float)
            self._diagonalize_tb(
                tb_model,
                gamma,
                need_eigenvectors=False,
                use_soc=use_soc,
                w90_filepath=w90_filepath,
                diag_engine="chinook",
                diag_device="cpu",
            )
            eigenvalues = np.real(tb_model.Eband)

        n_hops = len(tb_dict.get("list", []))
        n_orbs = len(self._orbital_list_from_basis(basis))
        stats = {
            "n_hops": n_hops,
            "n_orbs": n_orbs,
            "hop_tol": hop_tol,
            "parse_s": t_parse - t_total,
            "build_s": t_build - t_parse,
            "total_s": time.perf_counter() - t_total,
            "parse_hit": parse_hit,
            "build_hit": build_hit,
        }
        print(
            f"Prepare TB for ARPES: hops={n_hops:,} orbs={n_orbs} "
            f"hop_tol={hop_tol:g} parse={stats['parse_s']:.2f}s "
            f"build={stats['build_s']:.2f}s total={stats['total_s']:.2f}s",
            flush=True,
        )
        return {
            "tb_dict": tb_dict,
            "basis": basis,
            "tb_model": tb_model,
            "eigenvalues": eigenvalues,
            "stats": stats,
        }

    
    def get_kpath_template(self, lattice_type="hexagonal", a=3.0, b=3.0):
        if lattice_type == "hexagonal":
            return np.array([[0,0,0], [1/3, 1/3, 0], [0.5, 0, 0], [0,0,0]]), ["$\Gamma$", "K", "M", "$\Gamma$"]
        else:
            return np.array([[0,0,0], [0.5, 0, 0], [0.5, 0.5, 0], [0,0,0]]), ["$\Gamma$", "X", "M", "$\Gamma$"]
    
    def get_auto_kpath(self):
        """Uses PyMatgen to automatically find the standard BZ high-symmetry path."""
        from pymatgen.symmetry.bandstructure import HighSymmKpath
        
        if not self.crystal_structure:
            raise ValueError("No structure loaded to detect k-path.")
            
        kpath = HighSymmKpath(self.crystal_structure)
        kpts_dict = kpath.kpath['kpoints']
        path_segments = kpath.kpath['path']
        
        # Heuristic: If c-axis is large (>10 A), it is a 2D slab/sheet
        is_2d = self.crystal_structure.lattice.c > 10.0
        
        high_sym_pts = []
        labels = []
        
        for segment in path_segments:
            for lbl in segment:
                # Filter out individual 3D points
                if is_2d and abs(kpts_dict[lbl][2]) > 1e-4:
                    continue
                # Prevent immediate duplicate points where segments join
                if not labels or labels[-1] != lbl:
                    labels.append(lbl)
                    # Use the fractional coordinates directly!
                    high_sym_pts.append(kpts_dict[lbl])
                    
        # Format Gamma for matplotlib's LaTeX engine
        labels = ["$\Gamma$" if "Gamma" in lbl or lbl == "\\Gamma" else lbl for lbl in labels]
        
        return np.array(high_sym_pts), labels

    def get_custom_kpath(self, coords_str, labels_str):
        """Parses custom arbitrary path from UI text boxes."""
        try:
            # Parse strings like '0,0,0 ; 0.5,0,0' into float arrays
            pts = [[float(x) for x in pt.split(',')] for pt in coords_str.split(';')]
            lbls = [lbl.strip() for lbl in labels_str.split(';')]
            
            if len(pts) != len(lbls):
                raise ValueError("Number of coordinate points must match number of labels.")
                
            return np.array(pts), lbls
        except Exception as e:
            raise ValueError(f"Custom K-Path format error. Please check your syntax.\n{e}")

    def generate_k_path(self, high_sym_pts, labels, points_per_segment=100):
        k_vecs, k_dist, node_idx = [], [], [0]
        current_dist = 0.0
        for i in range(len(high_sym_pts) - 1):
            start, end = high_sym_pts[i], high_sym_pts[i+1]
            segment = np.linspace(start, end, points_per_segment)[1:] if i > 0 else np.linspace(start, end, points_per_segment)
            k_vecs.append(segment)
            segment_len = np.linalg.norm(end - start)
            k_dist.append(np.linspace(current_dist, current_dist + segment_len, len(segment)))
            current_dist += segment_len
            node_idx.append(len(np.vstack(k_vecs)) - 1)
        return np.vstack(k_vecs), np.concatenate(k_dist), node_idx, labels

    def export_wannier_dictionary(
        self, w90_filepath, use_soc=False, onsite_e=0.0, hop_tol=1e-6, qe_fermi=None
    ):
        """
        Parses wannier90_hr.dat natively to bypass Chinook's strict/buggy W90 importer.
        Dynamically extracts QE's lattice to reverse basis-vector rotation.

        hop_tol: drop hoppings with |t| <= hop_tol (default 1e-6). Raise for large
        third-party hr.dat so Prepare-for-ARPES / band build stays tractable.

        qe_fermi: if not None, fold this EF into onsite hoppings (UI value).
        Otherwise auto-detect from nscf/scf/FERMI_ENERGY.txt beside hr.dat.
        """
        if not self.crystal_structure:
            raise ValueError("Please load a crystal structure first.")

        hop_tol = float(hop_tol)
        t0 = time.perf_counter()
        print(
            f"Parsing Wannier90 Hamiltonian natively: {w90_filepath} "
            f"(hop_tol={hop_tol:g})",
            flush=True,
        )

        # Header only (comment / num_wann / nrpts / degeneracy weights)
        with open(w90_filepath, "r") as f:
            f.readline()
            num_wann = int(f.readline().strip())
            nrpts = int(f.readline().strip())
            deg_lines = int(np.ceil(nrpts / 15.0))
            deg_weights = []
            for _ in range(deg_lines):
                deg_weights.extend(int(x) for x in f.readline().split())
        deg_weights = np.asarray(deg_weights, dtype=np.float64)
        if len(deg_weights) < nrpts:
            raise ValueError(
                f"wannier90_hr.dat degeneracy block short: got {len(deg_weights)}, need {nrpts}"
            )

        data_start = 3 + deg_lines
        # Columns: R_x R_y R_z i j Re(H) Im(H) — streamed in chunks and pre-filtered,
        # so a multi-GB spinor hr.dat does not have to fit in memory all at once.
        hop, hop_r_idx, n_file_rows = _load_hr_rows(
            w90_filepath, data_start, num_wann, hop_tol
        )
        if hop.shape[0] == 0:
            raise ValueError(
                f"wannier90_hr.dat has no hopping above |t|>{hop_tol:g}: {w90_filepath}"
            )

        a_mat = self.crystal_structure.lattice.matrix

        # --- NATIVE LATTICE ALIGNMENT & FERMI SHIFT ---
        # QE defines a different basis (a1, a2) than PyMatgen for hexagonal cells.
        # We find the integer transformation matrix T to perfectly map the W90 cells back to PyMatgen.
        T_mat = np.eye(3)
        work_dir = os.path.dirname(w90_filepath)
        scf_out = os.path.join(work_dir, "scf.out")
        A_qe_found = False

        from tensorspec.core.dft.qe_fermi import detect_qe_fermi_eV

        if qe_fermi is not None:
            ef = float(qe_fermi)
            print(f"QE Fermi from UI/caller: {ef:.4f} eV", flush=True)
        else:
            ef, ef_src = detect_qe_fermi_eV(work_dir)
            if ef_src != "none":
                print(f"QE Fermi from {ef_src}: {ef:.4f} eV", flush=True)
            else:
                print("QE Fermi not found beside hr.dat; using 0.0 eV", flush=True)

        if os.path.exists(scf_out):
            try:
                alat_ang = 1.0
                qe_a = []
                with open(scf_out, "r") as f:
                    lines_scf = f.readlines()
                for k_idx, line in enumerate(lines_scf):
                    if "lattice parameter (alat)" in line:
                        alat_bohr = float(line.split("=")[1].split()[0])
                        alat_ang = alat_bohr * 0.5291772109
                    elif "crystal axes: (cart. coord. in units of alat)" in line:
                        for m_idx in range(1, 4):
                            coords_str = (
                                lines_scf[k_idx + m_idx]
                                .split("=")[1]
                                .replace("(", "")
                                .replace(")", "")
                            )
                            parts = coords_str.split()
                            qe_a.append([float(x) * alat_ang for x in parts])
                        break
                if len(qe_a) == 3:
                    A_qe = np.array(qe_a)
                    self.A_qe = A_qe  # SAVE TO SELF FOR K-PATH CALCULATION
                    A_pm_inv = np.linalg.inv(a_mat)
                    T_mat = np.round(np.dot(A_qe, A_pm_inv)).astype(int)
                    A_qe_found = True
            except Exception as e:
                print(f"Lattice alignment failed: {e}")

        # --- FALLBACK: Use wannier90.wout lattice if scf.out is missing ---
        wout_path = self._resolve_wannier_wout_path(work_dir)
        win_path = self._resolve_wannier_win_path(work_dir, w90_filepath)

        if (not os.path.exists(scf_out) or not A_qe_found) and os.path.exists(wout_path):
            try:
                qe_a = []
                with open(wout_path, "r") as f:
                    lines_wout = f.readlines()
                for i, line in enumerate(lines_wout):
                    if "Lattice Vectors" in line:
                        for j in range(2, 5):
                            parts = lines_wout[i + j].replace("|", "").split()
                            if len(parts) >= 4 and parts[0].startswith("a_"):
                                qe_a.append(
                                    [float(parts[1]), float(parts[2]), float(parts[3])]
                                )
                        break
                if len(qe_a) == 3:
                    A_qe = np.array(qe_a)
                    self.A_qe = A_qe  # SAVE TO SELF FOR K-PATH CALCULATION
                    A_pm_inv = np.linalg.inv(a_mat)
                    T_mat = np.round(np.dot(A_qe, A_pm_inv)).astype(int)
            except Exception as e:
                print(f"wout lattice alignment failed: {e}")

        flat_atoms = []
        flat_Z = {}
        flat_pos = []
        flat_orbs = []
        wannier_spinors = False
        basis_source = "cif_fallback"

        centers = self._parse_wannier_final_centers(wout_path, num_wann)
        built = None
        if os.path.isfile(win_path):
            built = self._build_wannier_flat_basis(win_path, centers, num_wann, use_soc)
        if built is not None:
            flat_atoms, flat_Z, flat_pos, flat_orbs, wannier_spinors = built
            basis_source = "win"
        elif len(centers) >= num_wann:
            built = self._build_cif_basis_with_wout_centers(centers, num_wann, use_soc)
            if built is not None:
                flat_atoms, flat_Z, flat_pos, flat_orbs, wannier_spinors = built
                basis_source = "wout_cif"
            else:
                print(
                    f"Found {len(centers)} WF centres in {os.path.basename(wout_path)} "
                    f"but CIF orbital count does not match num_wann={num_wann}. "
                    "Load matching CIF or add wannier90.win.",
                    flush=True,
                )
        else:
            print(
                f"Warning: No WF centres in {os.path.basename(wout_path)}. "
                "Falling back to CIF atom sites (ARPES matrix elements may be wrong).",
                flush=True,
            )

        if not flat_atoms:
            idx = 0
            for i, site in enumerate(self.crystal_structure):
                for orb in self._get_orbital_basis(site.species_string):
                    flat_atoms.append(idx)
                    flat_Z[idx] = site.specie.number
                    flat_pos.append(np.array(site.coords, dtype=float))
                    flat_orbs.append([orb])
                    idx += 1
            if len(flat_atoms) != num_wann and not wannier_spinors:
                print(
                    f"Warning: CIF basis count ({len(flat_atoms)}) != num_wann ({num_wann}). "
                    "Wannier index ↔ orbital mapping may be wrong for ARPES.",
                    flush=True,
                )

        # --- Vectorized hoppings (same physics as former line loop) ---
        R_qe = hop[:, 0:3]
        i_idx = hop[:, 3].astype(np.int64) - 1
        j_idx = hop[:, 4].astype(np.int64) - 1
        t_real = hop[:, 5].astype(np.float64, copy=True)
        t_imag = hop[:, 6].astype(np.float64)

        r_idx = hop_r_idx
        if r_idx.size and int(r_idx.max()) >= len(deg_weights):
            raise ValueError(
                f"Hopping rows exceed degeneracy table "
                f"(r_idx max={int(r_idx.max())}, nrpts={len(deg_weights)})"
            )
        weights = deg_weights[r_idx]

        onsite_mask = (
            (R_qe[:, 0] == 0)
            & (R_qe[:, 1] == 0)
            & (R_qe[:, 2] == 0)
            & (i_idx == j_idx)
        )
        t_real[onsite_mask] = t_real[onsite_mask] - ef + float(onsite_e)

        t_ij = (t_real + 1j * t_imag) / weights
        keep = np.abs(t_ij) > hop_tol
        n_raw = int(n_file_rows)
        n_keep = int(np.count_nonzero(keep))
        print(
            f"Hop filter |t|>{hop_tol:g}: kept {n_keep:,} / {n_raw:,} "
            f"({100.0 * n_keep / max(n_raw, 1):.2f}%)",
            flush=True,
        )

        R_qe_k = R_qe[keep]
        i_k = i_idx[keep]
        j_k = j_idx[keep]
        t_k = t_ij[keep]

        T_mat = np.asarray(T_mat, dtype=np.float64)
        a_mat = np.asarray(a_mat, dtype=np.float64)
        R_pm = R_qe_k @ T_mat
        R_cart = R_pm @ a_mat

        pos = np.asarray(flat_pos, dtype=np.float64)
        n_pos = len(pos)
        if n_pos == 0:
            raise ValueError("No orbital positions for Wannier → Cartesian hop map.")
        tau_i = pos[i_k % n_pos]
        tau_j = pos[j_k % n_pos]
        dR = R_cart + tau_j - tau_i

        # Chinook list H expects Python rows [i, j, dx, dy, dz, complex]
        # Build from pre-cast arrays (much faster than per-row int/float/complex in a tight loop).
        ik = i_k.astype(np.int64, copy=False)
        jk = j_k.astype(np.int64, copy=False)
        dr = dR.astype(np.float64, copy=False)
        tk = np.asarray(t_k, dtype=np.complex128, copy=False)
        n_hop = tk.shape[0]
        explicit_hopping = [
            [int(ik[n]), int(jk[n]), dr[n, 0], dr[n, 1], dr[n, 2], tk[n]]
            for n in range(n_hop)
        ]

        if use_soc and not wannier_spinors:
            # Wannier90 spinor hr.dat is already 2× — do not duplicate again.
            n_orbs = len(flat_atoms)
            flat_atoms = flat_atoms + [a + n_orbs for a in flat_atoms]

            new_Z = {}
            for idx in range(len(flat_atoms)):
                new_Z[idx] = flat_Z[idx % n_orbs]
            flat_Z = new_Z

            flat_pos = flat_pos + flat_pos
            flat_orbs = flat_orbs + flat_orbs

            spin_dict = {"bool": False, "soc": False}
        else:
            spin_dict = {"bool": False, "soc": False}

        self._wannier_spinors_flag = wannier_spinors
        tb_dict = {
            "type": "list",
            "list": explicit_hopping,
            "H": explicit_hopping,
            "a": a_mat.tolist(),
            "cutoff": 100.0,
            "renorm": 1.0,
            "offset": 0.0,
            "tol": 1e-15,
            "spin": spin_dict,
        }

        basis_args = {
            "atoms": flat_atoms,
            "Z": flat_Z,
            "pos": flat_pos,
            "orbs": flat_orbs,
            "spin": spin_dict,
            "wannier_spinors": bool(wannier_spinors),
            "basis_source": basis_source,
        }

        quality = self.assess_wannier_basis_quality(flat_pos, num_wann, basis_source)
        self._last_wannier_basis_quality = quality
        if quality.get("message"):
            print(f"Warning: {quality['message']}", flush=True)
        else:
            print(
                f"Wannier basis OK for ARPES: {quality['unique_centers']} unique WF centres, "
                f"num_wann={num_wann}, source={basis_source}",
                flush=True,
            )

        print(
            f"Successfully extracted {len(explicit_hopping)} non-zero hopping elements "
            f"for {num_wann} bands in {time.perf_counter() - t0:.2f}s "
            f"(read {hop.shape[0]} rows)."
        )
        return tb_dict, basis_args