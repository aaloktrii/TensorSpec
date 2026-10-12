#!/usr/bin/env python3
"""
Temperature-dependent VTe2 (-2 0 1) one-step ARPES with Chinook (+ GrizzlyME when
installed), driven directly by Zefeng's QE / Wannier90 outputs — no GUI needed.

    TensorSpec_env/bin/python scripts/chinook/run_vte2_temperature_chinook.py

WHAT IT DOES (nested loop, exactly as requested)
------------------------------------------------
  for every temperature folder under  <qe-root>  (one wannier90_hr.dat + CIF each)
      for every beam / slit geometry (beam 55 deg and 20 deg, x two slit pairings)
          for every deflector  -10.3, -5, 0, +5, +10.3 deg
              -> dispersion cut  I(theta_slit, E)   200 slit angles x 200 energies
          -> one iso-energy map at E_F  I(theta_slit, phi_deflector)
             slit -15..15 (200 pts), deflector -15..15 step 0.1 deg (301 pts)

Each result is written as ONE .npz in the TensorSpec Data-Viewer contract
(tensorspec.core.io.simulated_loader.SimulatedARPESLoader):

    intensity : (kx, ky, E)   float32
    kx        : k_slit     [1/A]  = k sin(theta_slit),   k = 0.512316 sqrt(hv - W)
    ky        : k_defl     [1/A]  = k sin(phi_deflector)
    E         : E - E_F    [eV]
    metadata  : dict (angles in degrees, every physics knob, provenance)

so "Load ARPES Data" in the GUI opens them directly, and they are byte-compatible
with scripts/chinook/to_viewer_npz_chinook.py / viewer_export.stack_viewer_cubes.

GEOMETRY (TensorSpec convention, see scripts/chinook/make_physics_json.py)
-------------------------------------------------------------------------
Chinook's lab frame: surface normal = +y, plane of incidence = x-y, slit 0 along x,
lab z vertical. There is no separate beam-azimuth knob, so the two slit pairings
are expressed by rotating sample + slit together:

    tag        beam incidence  manip_azimuth  slit_angle   meaning
    th55_ph0        55            90             0        beam in the slit plane (as measured)
    th20_ph0        20            90             0
    th55_ph90       55             0            90        beam perpendicular to the slit plane
    th20_ph90       20             0            90

The deflector has no knob of its own either: it IS k_bounds['Y'] collapsed to a
single value (phi_min = phi_max = d, nphi = 1), exactly like the GUI / remote runner.

OUTPUT TREE   (<out-root> = <TensorSpec_GUI>/scratch/vte2_temperature_dependent_chinook)
-----------
    <out-root>/
      <T_label>/                              e.g. T300K/   (folder name under qe-root)
        <T_label>_dispersion_grid_4configs_x_5deflectors.png
        <T_label>_isoE_EF_4configs.png
        run_manifest.json, run_log.txt
        th55_ph0/
          dispersion/th55_ph0_defl_m10p3_viewer.npz   ... (5 cuts)
          dispersion/th55_ph0_stacked_5deflectors.npz (kx, ky[5], E) Fermi-map-style stack
          iso_energy/th55_ph0_isoE_EF_viewer.npz       (kx, ky[301], E[1])
          th55_ph0_dispersions_5deflectors.png
          th55_ph0_isoE_EF.png
        th20_ph0/ ...   th55_ph90/ ...   th20_ph90/ ...
      summary/
        dispersion_<cfg>_rows-temperature_cols-deflector.png      (one per config)
        isoE_EF_rows-config_cols-temperature.png

Already-finished .npz files are skipped on re-run (resume); --force recomputes.

PHYSICS DEFAULTS (= the production VTe2 (-201) choices documented in
docs/superpowers/results/CHINOOK_VTe2_Zeeman_0p1T_methods_and_code.md)
---------------------------------------------------------------------
hv 84 eV, LH (p-pol), W 4.5 eV, V0 12 eV, Full matrix elements, rad_type slater,
mfp 8.7 A, se_width 0.05 eV, res_E 0.005 eV, res_k 0.005 1/A, Fresnel off, no photon
momentum, no k_z broadening, hop_tol 1e-4. Energy axis is padded internally by
10 x se_width on both sides and cropped before saving (same as the remote runner).
Fermi-Dirac temperature: the temperature parsed from the folder name (e.g. "T300K"
-> 300 K); use --fd-temperature 10 to force a constant.

All of these are CLI flags; run with -h.

ENGINE
------
--engine auto (default) uses GrizzlyME for the matrix elements when `grizzlyme` is
importable (it is in requirements.txt), otherwise chinook's serial M_k. Both share the
same k-mesh / geometry code (chinook_arpes_kmesh.py) and give identical cubes; the
pure-chinook path still needs grizzlyme installed for its radial-integral cache.
On a Mac use --device cpu (torch MPS has no complex128 eigh; auto already picks
cpu when there is no CUDA). The big iso-energy maps are split into θ-blocks sized
from the free RAM (--theta-chunk overrides). Expect, for a 324-WF CDW model on a
laptop CPU: a few seconds to minutes per dispersion cut, ~1 h per iso-energy map.

SMOKE TEST FIRST
----------------
    python run_vte2_temperature_chinook.py --dry-run          # discovery + plan only
    python run_vte2_temperature_chinook.py --quick --temps T300K --configs th55_ph0
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import importlib.util
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# 0. Defaults (edit here or override on the command line)
# ----------------------------------------------------------------------------
DEFAULT_QE_ROOT = "/Users/sandyai/Documents/VTe2/zefeng/NEW_QE"
DEFAULT_TENSORSPEC_ROOT = "/Users/sandyai/Documents/GitHub/TensorSpec_GUI"
OUT_SUBDIR = os.path.join("scratch", "vte2_temperature_dependent_chinook")

K_PER_SQRT_EV = 0.512316  # 1/A per sqrt(eV)

# tag -> (beam incidence, manip_azimuth, slit_angle, human label)
CONFIGS = {
    "th55_ph0":  (55.0, 90.0, 0.0,  "beam 55°, ∥ slit plane (as measured)"),
    "th20_ph0":  (20.0, 90.0, 0.0,  "beam 20°, ∥ slit plane"),
    "th55_ph90": (55.0, 0.0,  90.0, "beam 55°, ⊥ slit plane"),
    "th20_ph90": (20.0, 0.0,  90.0, "beam 20°, ⊥ slit plane"),
}
DEFAULT_DEFLECTORS = [-10.3, -5.0, 0.0, 5.0, 10.3]
# intensity modes of the one-step kernel (chinook_arpes_kmesh): "Full" = radial integrals x dipole x Bloch coefficients,
# "dipole" = |A.k|^2 only, "bare" = Lorentzian spectral function of the eigenvalues at the probed (k_par, k_z(hv)) --
# no matrix element, no Fermi-Dirac cutoff, no depth weighting (gauge-invariant, so fine for the MLWF Hamiltonians)
ME_MODES = {"full": "Full Matrix Elements", "dipole": "Polarization Dipole Only", "bare": "Bare Spectral Function (ME Off)"}


def defl_tag(d: float) -> str:
    """-10.3 -> m10p3, 5.0 -> 5, 0.0 -> 0 (same tags as the existing VTe2 matrix)."""
    s = f"{d:g}".replace("-", "m").replace(".", "p")
    return s


# ----------------------------------------------------------------------------
# 1. Locate TensorSpec and import its core (no Qt needed)
# ----------------------------------------------------------------------------
def find_tensorspec_root(cli_root: str | None) -> Path:
    cands = []
    if cli_root:
        cands.append(Path(cli_root))
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        cands.append(p)
    cands.append(Path(DEFAULT_TENSORSPEC_ROOT))
    cands.append(Path.cwd())
    for c in cands:
        if (c / "tensorspec" / "core" / "dft" / "chinook_tb.py").is_file():
            return c.resolve()
    raise SystemExit(
        "Cannot find the TensorSpec_GUI repo (folder containing tensorspec/core/dft/"
        "chinook_tb.py). Pass --tensorspec-root /path/to/TensorSpec_GUI"
    )


_CHINOOK_CHATTER = {
    "Initiate diagonalization:", "Diagonalization Complete.",
    "Begin computing matrix elements:", "Done matrix elements",
}


class Tee:
    """Mirror stdout/stderr into a log file; drop chinook's progress bars / chatter."""

    def __init__(self, path: Path):
        self.file = open(path, "a", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        if s.startswith("\r") or (s.endswith("100%") and "|" in s):
            return  # chinook serial_Mk progress bar
        if s.strip() in _CHINOOK_CHATTER:
            return
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def isatty(self):
        return False


def log(msg: str = ""):
    stamp = _dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# 2. Discover temperature folders (hr.dat + CIF + E_F) under qe-root
# ----------------------------------------------------------------------------
_TEMP_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[kK](?![a-zA-Z])")
_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")


def parse_temperature_K(label: str):
    """'T300K' / '300K_relaxed' -> 300.0; a bare number is accepted only if 4..2000 K."""
    m = _TEMP_RE.search(label)
    if m:
        return float(m.group(1))
    m = _NUM_RE.search(label)
    if m and 4.0 <= float(m.group(1)) <= 2000.0:
        return float(m.group(1))
    return None


def _find_cif(hr_dir: Path, temp_dir: Path, qe_root: Path, label: str):
    """CIF beside the hr.dat, else walking up to the temperature folder, else anywhere
    under it, else anywhere under qe_root (preferring a name with this folder's T)."""
    d = hr_dir
    while True:
        cifs = sorted(glob.glob(str(d / "*.cif")))
        if cifs:
            return _pick_cif(cifs, label)
        if d in (temp_dir, qe_root) or d.parent == d:
            break
        d = d.parent
    for base in (temp_dir, qe_root):
        cifs = sorted(glob.glob(str(base / "**" / "*.cif"), recursive=True))
        if cifs:
            return _pick_cif(cifs, label)
    return None


_FERMI_LINE = re.compile(r"the Fermi energy is\s+(-?\d+\.\d+)\s*ev", re.IGNORECASE)


def _grep_fermi(path: str):
    """Last 'the Fermi energy is X ev' line in a QE output (last SCF step wins)."""
    val = None
    try:
        with open(path, "r", errors="ignore") as fh:
            for line in fh:
                m = _FERMI_LINE.search(line)
                if m:
                    val = float(m.group(1))
    except OSError:
        return None
    return val


def _pick_cif(cifs, label):
    tk = parse_temperature_K(label)
    if tk is not None and len(cifs) > 1:
        tok = f"{tk:g}"
        pref = [c for c in cifs if tok in os.path.basename(c)]
        if pref:
            return pref[0]
    return cifs[0]


def _find_fermi(hr_dir: Path, temp_dir: Path, qe_root: Path, detect_fn):
    """TensorSpec order (nscf.out > scf.out > FERMI_ENERGY.txt) beside the hr.dat and up
    to the temperature folder; then any QE *.out / FERMI_ENERGY.txt under it (nscf first)."""
    d = hr_dir
    while True:
        ef, src = detect_fn(str(d))
        if src != "none":
            return float(ef), os.path.join(str(d), src)
        if d in (temp_dir, qe_root) or d.parent == d:
            break
        d = d.parent
    cands = sorted(glob.glob(str(temp_dir / "**" / "*.out"), recursive=True))
    cands.sort(key=lambda p: (0 if "nscf" in os.path.basename(p).lower() else
                              1 if "scf" in os.path.basename(p).lower() else 2, p))
    for p in cands:
        v = _grep_fermi(p)
        if v is not None:
            return v, p
    for p in sorted(glob.glob(str(temp_dir / "**" / "FERMI_ENERGY.txt"), recursive=True)):
        ef, src = detect_fn(os.path.dirname(p))
        if src != "none":
            return float(ef), p
    return None, None


def discover_temperatures(qe_root: Path, detect_fermi_fn, cif_override: str | None,
                          fermi_override: float | None):
    hr_files = []
    for root, dirs, files in os.walk(qe_root):
        dirs[:] = [x for x in dirs if not x.startswith(".")]
        for f in files:
            if f.endswith("_hr.dat"):
                hr_files.append(Path(root) / f)
    if not hr_files:
        raise SystemExit(f"No *_hr.dat found under {qe_root}")

    by_label = {}
    for hr in sorted(hr_files):
        rel = hr.relative_to(qe_root)
        label = rel.parts[0] if len(rel.parts) > 1 else qe_root.name
        label = re.sub(r"[^A-Za-z0-9_.+-]", "_", label)
        by_label.setdefault(label, []).append(hr)

    temps = []
    for label, hrs in by_label.items():
        if len(hrs) > 1:
            pref = [h for h in hrs if h.name == "wannier90_hr.dat"]
            chosen = pref[0] if pref else hrs[0]
            log(f"WARNING: {label}: {len(hrs)} hr.dat files; using {chosen}")
        else:
            chosen = hrs[0]
        hr_dir = chosen.parent
        rel = chosen.relative_to(qe_root)
        temp_dir = qe_root / rel.parts[0] if len(rel.parts) > 1 else qe_root
        cif = cif_override or _find_cif(hr_dir, temp_dir, qe_root, label)
        if fermi_override is not None:
            ef, ef_src = float(fermi_override), "CLI --fermi"
        else:
            ef, ef_src = _find_fermi(hr_dir, temp_dir, qe_root, detect_fermi_fn)
        win = hr_dir / "wannier90.win"
        if not win.is_file():
            alt = hr_dir / chosen.name.replace("_hr.dat", ".win")
            win = alt if alt.is_file() else None
        wouts = sorted(glob.glob(str(hr_dir / "*.wout")))
        wout = hr_dir / "wannier90.wout" if (hr_dir / "wannier90.wout").is_file() else (
            Path(wouts[0]) if wouts else None)
        temps.append({
            "label": label,
            "temperature_K": parse_temperature_K(label),
            "hr": str(chosen),
            "hr_dir": str(hr_dir),
            "cif": cif,
            "fermi_eV": ef,
            "fermi_source": ef_src,
            "win": str(win) if win else None,
            "wout": str(wout) if wout else None,
        })

    def _sort_key(t):
        tk = t["temperature_K"]
        return (0, tk) if tk is not None else (1, t["label"])

    temps.sort(key=_sort_key)
    return temps


# ----------------------------------------------------------------------------
# 3. Physics dict (same keys as physics_from_experiment_kwargs / arpes_physics.json)
# ----------------------------------------------------------------------------
def make_physics(cfg_tag: str, args, fd_temperature: float, spinor: bool) -> dict:
    inc, azi, slit, note = CONFIGS[cfg_tag]
    return {
        "hv": float(args.hv),
        "work_function": float(args.workf),
        "inner_potential": float(args.v0),
        "temperature": float(fd_temperature),
        "incidence_angle": float(inc),
        "polarization": "Linear Horizontal",  # p-pol, A in the plane of incidence
        "lin_pol_angle": 45.0,                # unused for Linear Horizontal
        "matrix_element_mode": ME_MODES[getattr(args, "me_mode", "full")],
        "manip_theta": 0.0,
        "manip_azimuth": float(azi),
        "manip_tilt": 0.0,
        "hkl": tuple(int(x) for x in args.hkl),
        "slit_angle": float(slit),
        "se_width": float(args.se_width),
        "res_E": float(args.res_e),
        "res_k": float(args.res_k),
        "rad_type": "slater",
        "mfp": float(args.mfp),
        "kz_halfwidth": 0.0,
        "kz_npoints": 1,
        "fresnel_enabled": False,
        "optical_n": 1.0,
        "optical_k": 0.0,
        "include_photon_momentum": False,
        # up/down spinor components add in intensity (GrizzlyME path); chinook path
        # ignores this key and adds them coherently (Kramers cancellation, ~3e-4).
        "spinor_basis": "interleaved" if spinor else "none",
        "_note": note,
    }


# ----------------------------------------------------------------------------
# 4. Engine helpers
# ----------------------------------------------------------------------------
def resolve_engine(engine: str, device: str):
    """Return (engine, device). GrizzlyME is used when importable (spinless ME on torch)."""
    have_grizzly = importlib.util.find_spec("grizzly") is not None
    if engine == "grizzly" and not have_grizzly:
        raise SystemExit("--engine grizzly but 'grizzlyme' is not installed in this Python.")
    use = engine if engine != "auto" else ("grizzly" if have_grizzly else "chinook")
    dev = "n/a"
    if use == "grizzly":
        import torch  # noqa: F401

        if device == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            # NOTE: Apple MPS has no complex128 eigh -> GrizzlyME diag would fail; stay on CPU.
        else:
            dev = device
            if dev == "mps":
                log("WARNING: torch MPS does not support complex128 linalg; expect failures. "
                    "Use --device cpu on a Mac.")
    return use, dev


def plan_theta_chunk(n_basis: int, nphi: int, ntheta: int, requested: int, ram_fraction=0.25):
    """θ-rows per block so that H(k) + eigenvectors (+ torch copies) fit in RAM."""
    if requested and requested > 0:
        return max(1, min(int(requested), ntheta))
    try:
        import psutil

        avail = psutil.virtual_memory().available
    except Exception:
        avail = 8 * 1024**3
    bytes_per_k = 4 * 16 * max(n_basis, 1) ** 2  # Hmat + Evec + eigh work + torch copy
    nk_max = max(int(avail * ram_fraction / bytes_per_k), nphi)
    rows = max(1, nk_max // max(nphi, 1))
    return int(min(rows, ntheta))


def _normalize_cube(Ig, nt, npf, ne):
    Ig = np.asarray(Ig, dtype=np.float32)
    if Ig.shape == (nt, npf, ne):
        return Ig
    if Ig.shape == (npf, nt, ne):
        return np.transpose(Ig, (1, 0, 2))
    if Ig.size == nt * npf * ne:
        return Ig.reshape(nt, npf, ne)
    raise RuntimeError(f"unexpected cube shape {Ig.shape}, expected ({nt},{npf},{ne})")


# globals for the (optional) fork-based chinook worker pool
_G = {}


def _chinook_block_worker(item):
    i0, i1, th, phis, e_axis = item
    from tensorspec.core.arpes.one_step.chinook_arpes_kmesh import run_chinook_arpes

    kb = {"X": [float(th[0]), float(th[-1]), len(th)],
          "Y": [float(phis[0]), float(phis[-1]), len(phis)],
          "E": [float(e_axis[0]), float(e_axis[-1]), len(e_axis)]}
    t0 = time.perf_counter()
    Ig = run_chinook_arpes(_G["tb_model"], kb, _G["physics"], _G["B"], fermi_shift=0.0)
    return i0, i1, _normalize_cube(Ig, len(th), len(phis), len(e_axis)), time.perf_counter() - t0


def compute_cube(tb_model, B_matrix, physics, thetas, phis, e_axis, *, engine, device,
                 theta_chunk, workers, me_shell=None, tag=""):
    """Full (ntheta, nphi, ne) float32 cube, computed in θ-blocks."""
    nt, npf, ne = len(thetas), len(phis), len(e_axis)
    cube = np.zeros((nt, npf, ne), dtype=np.float32)
    blocks = []
    for i0 in range(0, nt, theta_chunk):
        i1 = min(i0 + theta_chunk, nt)
        blocks.append((i0, i1, thetas[i0:i1]))
    nb = len(blocks)
    log(f"  {tag}: {nt} θ x {npf} φ x {ne} E  ->  {nb} θ-block(s) of ≤{theta_chunk} rows "
        f"({engine}{'/' + device if engine == 'grizzly' else ''})")
    t_all = time.perf_counter()

    if engine == "grizzly":
        from tensorspec.core.arpes.one_step.chinook_arpes_kmesh import run_grizzly_arpes

        for bi, (i0, i1, th) in enumerate(blocks):
            kb = {"X": [float(th[0]), float(th[-1]), len(th)],
                  "Y": [float(phis[0]), float(phis[-1]), npf],
                  "E": [float(e_axis[0]), float(e_axis[-1]), ne]}
            t0 = time.perf_counter()
            Ig = run_grizzly_arpes(tb_model, kb, physics, B_matrix, fermi_shift=0.0,
                                   device=device, me_shell=me_shell, profile_stages=False)
            cube[i0:i1] = _normalize_cube(Ig, len(th), npf, ne)
            dt = time.perf_counter() - t0
            if nb > 1 and (bi == 0 or (bi + 1) % max(1, nb // 10) == 0 or bi == nb - 1):
                eta = (time.perf_counter() - t_all) / (bi + 1) * (nb - bi - 1)
                log(f"    block {bi + 1}/{nb} θ=[{th[0]:.2f},{th[-1]:.2f}] {dt:.1f}s  ETA {eta/60:.1f} min")
            if device == "cuda":
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass
    else:
        items = [(i0, i1, th, phis, e_axis) for (i0, i1, th) in blocks]
        _G.update({"tb_model": tb_model, "physics": physics, "B": B_matrix})
        if workers > 1 and nb > 1:
            import multiprocessing as mp

            try:
                ctx = mp.get_context("fork")
            except ValueError:
                ctx = None
            if ctx is None:
                log("    fork start method unavailable -> running θ-blocks sequentially")
                workers = 1
        if workers > 1 and nb > 1:
            for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
                os.environ.setdefault(v, "1")
            # Pre-warm the radial-integral cache in the parent (1 k-point, same energy
            # window) so every forked worker inherits it instead of recomputing it.
            tw = time.perf_counter()
            _chinook_block_worker((0, 1, thetas[:1], phis[:1], e_axis))
            log(f"    radial integrals warm ({time.perf_counter() - tw:.1f}s); forking {min(workers, nb)} workers")
            done = 0
            with ctx.Pool(processes=min(workers, nb)) as pool:
                for i0, i1, blk, dt in pool.imap_unordered(_chinook_block_worker, items):
                    cube[i0:i1] = blk
                    done += 1
                    if done == 1 or done % max(1, nb // 10) == 0 or done == nb:
                        eta = (time.perf_counter() - t_all) / done * (nb - done)
                        log(f"    block {done}/{nb} done ({dt:.1f}s)  ETA {eta/60:.1f} min")
        else:
            for bi, item in enumerate(items):
                i0, i1, blk, dt = _chinook_block_worker(item)
                cube[i0:i1] = blk
                if nb > 1 and (bi == 0 or (bi + 1) % max(1, nb // 10) == 0 or bi == nb - 1):
                    eta = (time.perf_counter() - t_all) / (bi + 1) * (nb - bi - 1)
                    log(f"    block {bi + 1}/{nb} {dt:.1f}s  ETA {eta/60:.1f} min")

    log(f"  {tag}: done in {(time.perf_counter() - t_all)/60:.2f} min, ΣI={float(cube.sum()):.4e}")
    return cube


def padded_energy_axis(e_min, e_max, ne, se_width):
    """Runner-style padding: 10 x se_width each side at the same step, cropped on save."""
    e_out = np.linspace(e_min, e_max, ne)
    if ne <= 1:
        return e_out, e_out, 0
    de = float(e_out[1] - e_out[0])
    n_pad = int(math.ceil(10.0 * se_width / de))
    e_calc = np.linspace(e_out[0] - n_pad * de, e_out[-1] + n_pad * de, ne + 2 * n_pad)
    return e_calc, e_out, n_pad


# ----------------------------------------------------------------------------
# 5. Data-viewer .npz writer
# ----------------------------------------------------------------------------
def save_viewer_npz(path: Path, cube, theta_deg, phi_deg, E, hv, workf, meta: dict):
    k = K_PER_SQRT_EV * math.sqrt(max(hv - workf, 0.1))
    theta_deg = np.asarray(theta_deg, dtype=float)
    phi_deg = np.asarray(phi_deg, dtype=float)
    kx = k * np.sin(np.radians(theta_deg))
    ky = k * np.sin(np.radians(phi_deg))
    meta = dict(meta)
    meta.setdefault("axes", "intensity(kx,ky,E); kx=k_slit, ky=k_defl [1/A], E rel. E_F [eV]")
    meta["theta_lab_deg"] = theta_deg.tolist()
    meta["phi_lab_deg"] = phi_deg.tolist()
    meta["k_photoelectron_1perA"] = float(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, intensity=np.asarray(cube, dtype=np.float32), kx=kx, ky=ky,
                        E=np.asarray(E, dtype=float), metadata=meta)
    return path


def load_viewer_npz(path):
    d = np.load(path, allow_pickle=True)
    meta = d["metadata"].item() if "metadata" in d else {}
    return d["intensity"], d["kx"], d["ky"], d["E"], meta


# ----------------------------------------------------------------------------
# 6. Figures
# ----------------------------------------------------------------------------
def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    return plt, PowerNorm


def _wrap(label: str, width: int = 22) -> str:
    import textwrap

    return "\n".join(textwrap.fill(part, width) for part in str(label).split("\n"))


def _missing(ax, text="missing"):
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=9, color="0.4")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_dispersion_grid(rows, deflectors, out_png, title, gamma, row_labels, col_title_fn=None,
                         per_panel=False):
    """rows: list of lists of npz paths (len == len(deflectors)); one colour scale per row."""
    plt, PowerNorm = _mpl()
    nr, nc = len(rows), len(deflectors)
    fig, axes = plt.subplots(nr, nc, figsize=(3.3 * nc + 1.6, 2.9 * nr + 1.2), squeeze=False,
                             sharex=True, sharey=True)
    for r, paths in enumerate(rows):
        panels = []
        for p in paths:
            if p and os.path.isfile(p):
                I, kx, ky, E, meta = load_viewer_npz(p)
                panels.append((np.asarray(I, float)[:, 0, :], kx, E))
            else:
                panels.append(None)
        valid = [pp[0] for pp in panels if pp is not None]
        vmax_row = np.percentile(np.concatenate([v.ravel() for v in valid]), 99.5) if valid else 1.0
        total = float(sum(v.sum() for v in valid)) if valid else 0.0
        for c in range(nc):
            ax = axes[r, c]
            pp = panels[c]
            if pp is None:
                _missing(ax)
            else:
                I2, kx, E = pp
                vm = np.percentile(I2, 99.5) if per_panel else vmax_row
                ax.imshow(I2.T, origin="lower", aspect="auto",
                          extent=[kx.min(), kx.max(), E.min(), E.max()], cmap="magma",
                          norm=PowerNorm(gamma, vmin=0.0, vmax=max(float(vm), 1e-30)))
                ax.axhline(0.0, color="w", lw=0.6, alpha=0.45)
                ax.axvline(0.0, color="w", lw=0.6, alpha=0.3, ls=":")
            if r == 0:
                ax.set_title(col_title_fn(c) if col_title_fn else f"deflector {deflectors[c]:+.1f}°",
                             fontsize=10)
            if c == 0:
                ax.set_ylabel("$E-E_F$ (eV)", fontsize=9.5)
            if r == nr - 1:
                ax.set_xlabel("$k_{slit}$ (Å$^{-1}$)", fontsize=9.5)
            ax.tick_params(labelsize=8)
        axes[r, 0].text(-0.42, 0.5, f"{_wrap(row_labels[r])}\nΣI={total:.2e}",
                        transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=9,
                        linespacing=1.4)
    scale_note = ("each panel scaled to its own 99.5th pct" if per_panel
                  else f"one colour scale per row (99.5th pct), PowerNorm γ={gamma:g}")
    fig.suptitle(f"{title}\n{scale_note}", fontsize=11.5, y=0.995)
    fig.tight_layout(rect=[0.10, 0, 1, 0.95 if nr > 1 else 0.86])
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    log(f"  figure: {out_png}")


def plot_iso_grid(grid, out_png, title, gamma, row_labels, col_labels, per_panel=True):
    """grid[r][c] = npz path of an iso-E map (kx, ky, E[1]) or None."""
    plt, PowerNorm = _mpl()
    nr, nc = len(grid), len(grid[0])
    fig, axes = plt.subplots(nr, nc, figsize=(3.6 * nc + 1.4, 3.4 * nr + 1.0), squeeze=False)
    for r in range(nr):
        panels = []
        for c in range(nc):
            p = grid[r][c]
            if p and os.path.isfile(p):
                I, kx, ky, E, meta = load_viewer_npz(p)
                ie = int(np.argmin(np.abs(E))) if len(E) > 1 else 0
                panels.append((np.asarray(I, float)[:, :, ie], kx, ky, float(E[ie])))
            else:
                panels.append(None)
        valid = [pp[0] for pp in panels if pp is not None]
        vmax_row = np.percentile(np.concatenate([v.ravel() for v in valid]), 99.5) if valid else 1.0
        for c in range(nc):
            ax = axes[r, c]
            pp = panels[c]
            if pp is None:
                _missing(ax)
            else:
                I2, kx, ky, e0 = pp
                vm = np.percentile(I2, 99.5) if per_panel else vmax_row
                ax.imshow(I2.T, origin="lower", aspect="equal",
                          extent=[kx.min(), kx.max(), ky.min(), ky.max()], cmap="magma",
                          norm=PowerNorm(gamma, vmin=0.0, vmax=max(float(vm), 1e-30)))
                ax.axhline(0.0, color="w", lw=0.5, alpha=0.35, ls=":")
                ax.axvline(0.0, color="w", lw=0.5, alpha=0.35, ls=":")
                ax.text(0.02, 0.97, f"E={e0:+.3f} eV", transform=ax.transAxes, color="w",
                        fontsize=7.5, va="top")
            if r == 0:
                ax.set_title(col_labels[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{row_labels[r]}\n$k_{{defl}}$ (Å$^{{-1}}$)", fontsize=9.5)
            if r == nr - 1:
                ax.set_xlabel("$k_{slit}$ (Å$^{-1}$)", fontsize=9.5)
            ax.tick_params(labelsize=8)
    scale_note = ("each panel scaled to its own 99.5th pct" if per_panel
                  else f"one colour scale per row (99.5th pct)") + f", PowerNorm γ={gamma:g}"
    fig.suptitle(f"{title}\n{scale_note}", fontsize=11.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    log(f"  figure: {out_png}")


# ----------------------------------------------------------------------------
# 7. Paths for one (temperature, config) and the plotting driver
# ----------------------------------------------------------------------------
def cfg_dir(out_root: Path, label: str, cfg: str) -> Path:
    return out_root / label / cfg


def disp_path(out_root, label, cfg, d) -> Path:
    return cfg_dir(out_root, label, cfg) / "dispersion" / f"{cfg}_defl_{defl_tag(d)}_viewer.npz"


def stack_path(out_root, label, cfg, n) -> Path:
    return cfg_dir(out_root, label, cfg) / "dispersion" / f"{cfg}_stacked_{n}deflectors.npz"


def iso_path(out_root, label, cfg) -> Path:
    return cfg_dir(out_root, label, cfg) / "iso_energy" / f"{cfg}_isoE_EF_viewer.npz"


def short_labels(labels):
    """Display names for figures: drop '_'-separated tokens shared by ALL labels
    (deliverable_VTe2_CDW_XRD100K_NM_6x23x9_MLWF2000 -> XRD100K). Folder names are untouched."""
    labels = list(labels)
    if len(labels) < 2:
        return {lab: lab for lab in labels}
    toks = [lab.split("_") for lab in labels]
    common = set(toks[0])
    for tk in toks[1:]:
        common &= set(tk)
    out = {}
    for lab, tk in zip(labels, toks):
        keep = [x for x in tk if x not in common]
        if not keep:  # label made only of shared tokens: strip the common prefix instead
            pre = os.path.commonprefix(labels)
            keep = [lab[len(pre):] or lab]
        out[lab] = "_".join(keep)
    if len(set(out.values())) < len(labels):  # collisions -> fall back to full names
        return {lab: lab for lab in labels}
    return out


def make_all_figures(out_root: Path, temps, cfgs, deflectors, args):
    header = (f"VTe$_2$ ({args.hkl[0]} {args.hkl[1]} {args.hkl[2]}) Chinook one-step ARPES  —  "
              f"hν={args.hv:g} eV LH, V$_0$={args.v0:g} eV, W={args.workf:g} eV"
              + (f"  —  {ME_MODES[args.me_mode]}" if getattr(args, "me_mode", "full") != "full" else ""))
    nice = short_labels([t["label"] for t in temps])
    for t in temps:
        label = t["label"]
        tdir = out_root / label
        if not tdir.is_dir():
            continue
        # per-config 1x5 + iso
        tname = nice[label] + (f" ({t['temperature_K']:g} K)" if t.get("temperature_K") else "")
        for cfg in cfgs:
            paths = [str(disp_path(out_root, label, cfg, d)) for d in deflectors]
            if any(os.path.isfile(p) for p in paths):
                plot_dispersion_grid([paths], deflectors,
                                     str(cfg_dir(out_root, label, cfg) / f"{cfg}_dispersions_{len(deflectors)}deflectors.png"),
                                     f"{header}\n{tname}  |  {cfg}: {CONFIGS[cfg][3]}",
                                     args.gamma, [cfg])
            ip = str(iso_path(out_root, label, cfg))
            if os.path.isfile(ip):
                plot_iso_grid([[ip]], str(cfg_dir(out_root, label, cfg) / f"{cfg}_isoE_EF.png"),
                              f"{header}\n{tname}  |  {cfg}: {CONFIGS[cfg][3]}  |  iso-energy at E_F",
                              args.gamma, [cfg], ["E = E$_F$"])
        # per-temperature 4x5 and iso 1x4
        rows = [[str(disp_path(out_root, label, cfg, d)) for d in deflectors] for cfg in cfgs]
        if any(os.path.isfile(p) for r in rows for p in r):
            plot_dispersion_grid(
                rows, deflectors,
                str(tdir / f"{label}_dispersion_grid_{len(cfgs)}configs_x_{len(deflectors)}deflectors.png"),
                f"{header}\n{tname}   [{label}]",
                args.gamma, [f"{c}\n{CONFIGS[c][3]}" for c in cfgs],
                col_title_fn=lambda c: (f"deflector {deflectors[c]:+.1f}°\n"
                                        f"$k_\\perp$={K_PER_SQRT_EV*math.sqrt(args.hv-args.workf)*math.sin(math.radians(deflectors[c])):+.3f} Å$^{{-1}}$"))
        iso_row = [str(iso_path(out_root, label, cfg)) for cfg in cfgs]
        if any(os.path.isfile(p) for p in iso_row):
            plot_iso_grid([iso_row], str(tdir / f"{label}_isoE_EF_{len(cfgs)}configs.png"),
                          f"{header}\n{tname}  |  iso-energy maps at E_F   [{label}]", args.gamma, [nice[label]],
                          [f"{c}\n{CONFIGS[c][3]}" for c in cfgs])
    # cross-temperature summaries
    sdir = out_root / "summary"
    tl = [t["label"] for t in temps]
    tn = [nice[lab] + (f"\n{t['temperature_K']:g} K" if t.get("temperature_K") else "") for lab, t in zip(tl, temps)]
    for cfg in cfgs:
        rows = [[str(disp_path(out_root, lab, cfg, d)) for d in deflectors] for lab in tl]
        if any(os.path.isfile(p) for r in rows for p in r):
            plot_dispersion_grid(rows, deflectors,
                                 str(sdir / f"dispersion_{cfg}_rows-temperature_cols-deflector.png"),
                                 f"{header}\n{cfg}: {CONFIGS[cfg][3]}  |  rows = temperature",
                                 args.gamma, tn)
    grid = [[str(iso_path(out_root, lab, cfg)) for lab in tl] for cfg in cfgs]
    if any(os.path.isfile(p) for r in grid for p in r):
        plot_iso_grid(grid, str(sdir / "isoE_EF_rows-config_cols-temperature.png"),
                      f"{header}\niso-energy at E_F  |  rows = geometry, columns = temperature",
                      args.gamma, cfgs, tn)


# ----------------------------------------------------------------------------
# 8. Main
# ----------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qe-root", default=DEFAULT_QE_ROOT, help="folder with one sub-folder per temperature")
    ap.add_argument("--tensorspec-root", default=None, help="TensorSpec_GUI repo (auto-detected)")
    ap.add_argument("--out-root", default=None, help=f"default <tensorspec-root>/{OUT_SUBDIR}")
    ap.add_argument("--temps", default="", help="comma list of temperature labels to run (substring match)")
    ap.add_argument("--configs", default=",".join(CONFIGS), help="comma list of geometry tags")
    ap.add_argument("--deflectors", default=",".join(f"{d:g}" for d in DEFAULT_DEFLECTORS))
    ap.add_argument("--cif", default=None, help="use this CIF for every temperature (default: auto-find)")
    ap.add_argument("--fermi", type=float, default=None, help="QE E_F in eV for every temperature (default: auto-detect)")
    ap.add_argument("--hkl", type=int, nargs=3, default=[-2, 0, 1], help="cleavage plane")
    # grids
    ap.add_argument("--theta-min", type=float, default=-15.0)
    ap.add_argument("--theta-max", type=float, default=15.0)
    ap.add_argument("--ntheta", type=int, default=200)
    ap.add_argument("--e-min", type=float, default=-1.2)
    ap.add_argument("--e-max", type=float, default=0.1)
    ap.add_argument("--ne", type=int, default=200)
    ap.add_argument("--iso-phi-min", type=float, default=-15.0)
    ap.add_argument("--iso-phi-max", type=float, default=15.0)
    ap.add_argument("--iso-phi-step", type=float, default=0.1)
    ap.add_argument("--iso-ntheta", type=int, default=None, help="slit points for the iso map (default = --ntheta)")
    ap.add_argument("--iso-e", type=float, default=0.0, help="iso-energy (eV rel. E_F)")
    ap.add_argument("--iso-e-halfwidth", type=float, default=0.25,
                    help="energy window computed around --iso-e so Lorentzian tails are included (chinook needs a finite window)")
    ap.add_argument("--iso-ne", type=int, default=11, help="energy points in that window (slice nearest --iso-e is saved)")
    ap.add_argument("--iso-save-window", action="store_true", help="also save the whole small energy window cube")
    # physics
    ap.add_argument("--hv", type=float, default=84.0)
    ap.add_argument("--workf", type=float, default=4.5)
    ap.add_argument("--v0", type=float, default=12.0)
    ap.add_argument("--fd-temperature", default="folder",
                    help="Fermi-Dirac T in K, or 'folder' = parse from the folder name (fallback 10 K)")
    ap.add_argument("--se-width", type=float, default=0.05)
    ap.add_argument("--res-e", type=float, default=0.005)
    ap.add_argument("--res-k", type=float, default=0.005)
    ap.add_argument("--mfp", type=float, default=8.7)
    ap.add_argument("--me-mode", choices=tuple(ME_MODES), default="full",
                    help="full = one-step matrix elements (default); dipole = |A.k|^2 only; bare = spectral function of the "
                         "bands at the probed (k_par, k_z), no matrix element and no Fermi-Dirac cutoff -- use with --out-root")
    ap.add_argument("--hop-tol", type=float, default=1e-4)
    ap.add_argument("--use-soc", action="store_true", help="only for a NON-spinor hr.dat (a spinor one is already doubled)")
    # engine
    ap.add_argument("--engine", choices=("auto", "grizzly", "chinook"), default="auto")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    ap.add_argument("--theta-chunk", type=int, default=0, help="θ rows per block (0 = auto from free RAM)")
    ap.add_argument("--workers", type=int, default=1, help="chinook engine only: fork workers over θ-blocks")
    # control
    ap.add_argument("--skip-dispersion", action="store_true")
    ap.add_argument("--skip-iso", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute even if the .npz exists")
    ap.add_argument("--quick", action="store_true", help="tiny grids into <out-root>/quick for a smoke test")
    ap.add_argument("--dry-run", action="store_true", help="discover folders, print the plan, exit")
    ap.add_argument("--plots-only", action="store_true", help="only (re)draw figures from existing .npz")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--gamma", type=float, default=0.5, help="PowerNorm exponent for the figures (1 = linear)")
    return ap


def main():
    args = build_parser().parse_args()
    ts_root = find_tensorspec_root(args.tensorspec_root)
    if str(ts_root) not in sys.path:
        sys.path.insert(0, str(ts_root))
    out_root = Path(args.out_root) if args.out_root else ts_root / OUT_SUBDIR
    if args.quick:
        out_root = out_root / "quick"
        args.ntheta, args.ne = min(args.ntheta, 40), min(args.ne, 60)
        args.iso_phi_step = max(args.iso_phi_step, 1.0)
        args.iso_ne = min(args.iso_ne, 5)
    out_root.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(out_root / "run_log.txt")
    sys.stderr = sys.stdout

    import collections
    import collections.abc

    if not hasattr(collections, "Iterable"):
        collections.Iterable = collections.abc.Iterable

    from pymatgen.core import Structure
    from tensorspec.core.dft.qe_fermi import detect_qe_fermi_eV
    from tensorspec.core.dft.sprkkr.viewer_export import stack_viewer_cubes

    cfgs = [c.strip() for c in args.configs.split(",") if c.strip()]
    bad = [c for c in cfgs if c not in CONFIGS]
    if bad:
        raise SystemExit(f"unknown config tag(s) {bad}; choose from {list(CONFIGS)}")
    deflectors = [float(x) for x in args.deflectors.split(",") if x.strip()]
    iso_ntheta = int(args.iso_ntheta or args.ntheta)
    if args.iso_ne < 3:
        # chinook selects states inside (E0 - 5dE, E1 + 5dE); a zero-width window finds none.
        print(f"note: --iso-ne {args.iso_ne} -> 3 (chinook needs a finite energy window)", flush=True)
        args.iso_ne = 3

    log("=" * 78)
    log("VTe2 temperature-dependent Chinook ARPES batch")
    log(f"TensorSpec root : {ts_root}")
    log(f"QE root         : {args.qe_root}")
    log(f"Output root     : {out_root}")
    temps = discover_temperatures(Path(args.qe_root), detect_qe_fermi_eV, args.cif, args.fermi)
    if args.temps:
        want = [w.strip() for w in args.temps.split(",") if w.strip()]
        temps = [t for t in temps if any(w in t["label"] for w in want)]
        if not temps:
            raise SystemExit(f"--temps {args.temps!r} matched nothing")
    log(f"Temperatures    : {[t['label'] for t in temps]}")
    for t in temps:
        log(f"  {t['label']:>12s}: T={t['temperature_K']} K  hr={t['hr']}")
        log(f"  {'':>12s}  cif={t['cif']}")
        log(f"  {'':>12s}  E_F={t['fermi_eV']} eV ({t['fermi_source']})  win={bool(t['win'])} wout={bool(t['wout'])}")
        if t["cif"] is None:
            log("  !! no .cif found for this temperature -> pass --cif")
        if t["fermi_eV"] is None:
            log("  !! no Fermi energy found (nscf.out / scf.out / FERMI_ENERGY.txt) -> pass --fermi")
    log(f"Geometries      : {cfgs}")
    log(f"Deflectors      : {deflectors}")
    log(f"Dispersion grid : θ[{args.theta_min},{args.theta_max}] x {args.ntheta}, E[{args.e_min},{args.e_max}] x {args.ne}")
    n_phi_iso = int(round((args.iso_phi_max - args.iso_phi_min) / args.iso_phi_step)) + 1
    log(f"Iso-E grid      : θ x {iso_ntheta}, φ[{args.iso_phi_min},{args.iso_phi_max}] step {args.iso_phi_step} "
        f"({n_phi_iso} pts), E={args.iso_e:+.3f} eV (window ±{args.iso_e_halfwidth} eV, {args.iso_ne} pts)")

    if args.dry_run:
        log("dry run: exiting before any calculation")
        return
    if args.plots_only:
        make_all_figures(out_root, temps, cfgs, deflectors, args)
        return

    missing = [t["label"] for t in temps if t["cif"] is None or t["fermi_eV"] is None]
    if missing:
        raise SystemExit(f"cannot run {missing}: CIF or Fermi energy missing (see above; use --cif / --fermi)")

    engine, device = resolve_engine(args.engine, args.device)
    log(f"Engine          : {engine}" + (f" on {device}" if engine == "grizzly" else ""))
    if engine == "grizzly":
        from tensorspec.core.arpes.one_step.chinook_arpes_kmesh import build_grizzly_me_shell
    from tensorspec.core.dft.chinook_tb import ChinookTightBindingEngine

    thetas = np.linspace(args.theta_min, args.theta_max, args.ntheta)
    e_calc, e_out, n_pad = padded_energy_axis(args.e_min, args.e_max, args.ne, args.se_width)
    thetas_iso = np.linspace(args.theta_min, args.theta_max, iso_ntheta)
    phis_iso = np.linspace(args.iso_phi_min, args.iso_phi_max, n_phi_iso)
    e_iso = (np.linspace(args.iso_e - args.iso_e_halfwidth, args.iso_e + args.iso_e_halfwidth, args.iso_ne)
             if args.iso_ne > 1 else np.array([args.iso_e]))
    i_iso = int(np.argmin(np.abs(e_iso - args.iso_e)))
    log(f"Energy padding  : {n_pad} steps each side ({n_pad * (e_out[1]-e_out[0] if args.ne > 1 else 0):.3f} eV), cropped on save")

    t_batch = time.perf_counter()
    for t in temps:
        label = t["label"]
        tdir = out_root / label
        tdir.mkdir(parents=True, exist_ok=True)
        log("")
        log("#" * 78)
        log(f"# TEMPERATURE {label}   (T = {t['temperature_K']} K)")
        log("#" * 78)

        # what still needs computing?
        todo = []
        for cfg in cfgs:
            if not args.skip_dispersion:
                for d in deflectors:
                    p = disp_path(out_root, label, cfg, d)
                    if args.force or not p.is_file():
                        todo.append(("disp", cfg, d))
            if not args.skip_iso:
                p = iso_path(out_root, label, cfg)
                if args.force or not p.is_file():
                    todo.append(("iso", cfg, None))
        if not todo:
            log("  everything already on disk -> skipping (use --force to recompute)")
            continue

        # ---- tight-binding model (parsed once per temperature; disk-cached by TensorSpec) ----
        t0 = time.perf_counter()
        eng = ChinookTightBindingEngine()
        eng.crystal_structure = Structure.from_file(t["cif"])
        st = eng.crystal_structure
        log(f"  structure: {st.composition.reduced_formula}, {st.num_sites} sites, "
            f"a,b,c = {st.lattice.a:.3f} {st.lattice.b:.3f} {st.lattice.c:.3f} Å, "
            f"α,β,γ = {st.lattice.alpha:.2f} {st.lattice.beta:.2f} {st.lattice.gamma:.2f}°")
        log(f"  parsing {t['hr']}  (E_F = {t['fermi_eV']:.4f} eV from {t['fermi_source']}, hop_tol={args.hop_tol:g})")
        prep = eng.prepare_tb_for_arpes(t["hr"], use_soc=args.use_soc, onsite_e=0.0,
                                        hop_tol=args.hop_tol, quick_diag=True, qe_fermi=float(t["fermi_eV"]))
        tb_model = prep["tb_model"]
        n_orb = prep["stats"]["n_orbs"]
        n_hop = prep["stats"]["n_hops"]
        spinor = bool(getattr(eng, "_wannier_spinors_flag", False))
        quality = getattr(eng, "_last_wannier_basis_quality", None) or {}
        B_matrix = np.asarray(st.lattice.reciprocal_lattice.matrix, dtype=float)  # CIF frame (same as GUI local path)
        log(f"  TB ready: {n_orb} orbitals ({'spinor' if spinor else 'scalar'}), {n_hop:,} hops, "
            f"basis_source={quality.get('basis_source')}, unique WF centres={quality.get('unique_centers')} "
            f"[{(time.perf_counter()-t0):.1f}s]")
        if quality.get("message"):
            log(f"  !! {quality['message']}")
        a_qe = getattr(eng, "A_qe", None)
        if a_qe is not None:
            T_mat = np.asarray(a_qe, float) @ np.linalg.inv(st.lattice.matrix)
            if not np.allclose(T_mat, np.round(T_mat), atol=5e-2):
                log("  !! QE/Wannier lattice is NOT an integer multiple of the CIF lattice "
                    f"(T = {np.round(T_mat, 3).tolist()}) — the CIF probably is a different cell setting; "
                    "hoppings may be mis-mapped. Check the CIF.")
        ev = prep.get("eigenvalues")
        if ev is not None:
            ev = np.asarray(ev, float).ravel()
            log(f"  Γ-point bands: {np.count_nonzero((ev > -1.5) & (ev < 0.5))} within [-1.5, 0.5] eV, "
                f"closest to E_F: {ev[np.argmin(np.abs(ev))]:+.3f} eV")
        if t["temperature_K"] is None and args.fd_temperature == "folder":
            log("  note: no temperature in folder name -> Fermi-Dirac T = 10 K")
        fd_T = (float(args.fd_temperature) if args.fd_temperature != "folder"
                else float(t["temperature_K"] if t["temperature_K"] is not None else 10.0))
        log(f"  Fermi-Dirac temperature for the cutoff: {fd_T:g} K")

        base_meta = {
            "engine": f"Chinook one-step + {'GrizzlyME' if engine == 'grizzly' else 'chinook serial ME'}",
            "device": device,
            "temperature_label": label,
            "temperature_K": t["temperature_K"],
            "fd_temperature_K": fd_T,
            "hr_dat": t["hr"], "cif": t["cif"],
            "qe_fermi_eV": float(t["fermi_eV"]), "qe_fermi_source": t["fermi_source"],
            "num_orbitals": int(n_orb), "n_hops": int(n_hop), "hop_tol": float(args.hop_tol),
            "wannier_spinors": spinor, "basis_source": quality.get("basis_source"),
            "hv_eV": float(args.hv), "ework_eV": float(args.workf), "inner_potential_eV": float(args.v0),
            "polarization": "Linear Horizontal (p)", "hkl": [int(x) for x in args.hkl],
            "se_width_eV": float(args.se_width), "res_E_eV": float(args.res_e), "res_k_1perA": float(args.res_k),
            "mfp_A": float(args.mfp), "rad_type": "slater", "matrix_element_mode": ME_MODES[args.me_mode],
            "fresnel": False, "photon_momentum": False, "kz_broadening": False,
            "fermi_edge_applied": args.me_mode != "bare", "script": os.path.abspath(__file__),
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
        }
        manifest = {"temperature": t, "args": vars(args), "engine": engine, "device": device,
                    "results": []}

        for cfg in cfgs:
            inc, azi, slit, note = CONFIGS[cfg]
            physics = make_physics(cfg, args, fd_T, spinor)
            cfg_meta = {**base_meta, "config": cfg, "note": note, "incidence_angle_deg": inc,
                        "manip_azimuth_deg": azi, "slit_angle_deg": slit, "manip_theta_deg": 0.0,
                        "manip_tilt_deg": 0.0}
            log("")
            log(f"  === {label} | {cfg}: {note}  (incidence {inc:g}°, azimuth {azi:g}°, slit {slit:g}°)")

            # ---------------- dispersion cuts, one per deflector ----------------
            disp_items = [d for (k, c, d) in todo if k == "disp" and c == cfg]
            if disp_items:
                chunk = plan_theta_chunk(n_orb, 1, args.ntheta, args.theta_chunk)
                shell = None
                if engine == "grizzly" and args.me_mode == "full":
                    log("  hoisting Full-ME radial-integral shell (dispersion energy window) ...")
                    shell = build_grizzly_me_shell(tb_model, physics, B_matrix, e_calc)
                for d in disp_items:
                    tag = f"{cfg} deflector {d:+.1f}°"
                    t1 = time.perf_counter()
                    cube = compute_cube(tb_model, B_matrix, physics, thetas, np.array([d]), e_calc,
                                        engine=engine, device=device, theta_chunk=chunk,
                                        workers=args.workers, me_shell=shell, tag=tag)
                    if n_pad:
                        cube = cube[:, :, n_pad:n_pad + len(e_out)]
                    p = disp_path(out_root, label, cfg, d)
                    meta = {**cfg_meta, "kind": "dispersion", "deflector_deg": float(d),
                            "wall_s": round(time.perf_counter() - t1, 2),
                            "e_pad_steps": int(n_pad), "theta_chunk": int(chunk)}
                    save_viewer_npz(p, cube, thetas, [d], e_out, args.hv, args.workf, meta)
                    manifest["results"].append({"kind": "dispersion", "config": cfg, "deflector": d,
                                                "npz": str(p), "wall_s": meta["wall_s"]})
                    log(f"  saved {p}")
            # stacked (kx, ky[ndefl], E) cube when all cuts exist
            cuts = [str(disp_path(out_root, label, cfg, d)) for d in deflectors]
            if all(os.path.isfile(c) for c in cuts) and not args.skip_dispersion:
                sp = stack_path(out_root, label, cfg, len(deflectors))
                if args.force or not sp.is_file():
                    stack_viewer_cubes(cuts, str(sp), metadata={**cfg_meta, "kind": "dispersion_stack",
                                                                  "deflectors_deg": deflectors})
                    log(f"  saved {sp}")

            # ---------------- iso-energy map at E_F ----------------
            if any(k == "iso" and c == cfg for (k, c, d) in todo):
                chunk = plan_theta_chunk(n_orb, n_phi_iso, iso_ntheta, args.theta_chunk)
                shell = None
                if engine == "grizzly" and args.me_mode == "full":
                    log("  hoisting Full-ME radial-integral shell (iso-energy window) ...")
                    shell = build_grizzly_me_shell(tb_model, physics, B_matrix, e_iso)
                t1 = time.perf_counter()
                cube = compute_cube(tb_model, B_matrix, physics, thetas_iso, phis_iso, e_iso,
                                    engine=engine, device=device, theta_chunk=chunk,
                                    workers=args.workers, me_shell=shell, tag=f"{cfg} iso-E map")
                p = iso_path(out_root, label, cfg)
                meta = {**cfg_meta, "kind": "iso_energy", "iso_energy_eV": float(e_iso[i_iso]),
                        "iso_window_eV": [float(e_iso[0]), float(e_iso[-1]), int(len(e_iso))],
                        "wall_s": round(time.perf_counter() - t1, 2), "theta_chunk": int(chunk)}
                save_viewer_npz(p, cube[:, :, i_iso:i_iso + 1], thetas_iso, phis_iso, e_iso[i_iso:i_iso + 1],
                                args.hv, args.workf, meta)
                log(f"  saved {p}")
                if args.iso_save_window:
                    pw = p.with_name(p.name.replace("_viewer.npz", "_window_viewer.npz"))
                    save_viewer_npz(pw, cube, thetas_iso, phis_iso, e_iso, args.hv, args.workf,
                                    {**meta, "kind": "iso_energy_window"})
                    log(f"  saved {pw}")
                manifest["results"].append({"kind": "iso_energy", "config": cfg, "npz": str(p),
                                            "wall_s": meta["wall_s"]})

        with open(tdir / "run_manifest.json", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(manifest, default=str) + "\n")
        del tb_model, eng, prep
        import gc

        gc.collect()

    log("")
    log(f"all calculations finished in {(time.perf_counter() - t_batch)/60:.1f} min")
    if not args.no_plots:
        log("drawing figures ...")
        make_all_figures(out_root, temps, cfgs, deflectors, args)
    log(f"DONE  ->  {out_root}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted — finished .npz files are kept; re-run to resume.", flush=True)
        sys.exit(130)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
