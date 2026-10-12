"""Orchestration: structure -> SCF -> converged pot -> ARPES -> tensor/tree.

Pure glue over jobs.py / fanout.py / inputs.py / outputs.py / progress.py.
Zero Qt (sandy_rule.md layer 1). No GUI imports here -- panels call into
this module, never the reverse.

`launcher` is either a `jobs.LocalLauncher` (has `.bin_dir`) or a
`jobs.RemoteLauncher` (has `.cluster`); `_is_remote()` tells them apart by
duck-typing so a test fake only needs to mirror whichever shape it plays.

Cross-host paths: `workdir` is ALWAYS the LOCAL staging directory -- inputs
are built here (`build_scf_inputs` / `build_arpes_inputs`) and, for a
remote launcher, outputs are fetched back here. A remote launcher also
needs `remote_workdir`, a path valid on the cluster, where the job
actually runs (upload target, `JobSpec.workdir`); omitting it on a remote
call raises `ValueError`. For ARPES, if the starting `pot_path` is not a
file that exists locally, it is treated as a cluster-only path and pulled
down first (`RemoteLauncher.download`) since `build_arpes_inputs` needs it
locally via ase2sprkkr; the `.inp`'s `POTFIL=` is then rewritten to that
pot's bare filename before uploading both into the remote subdir.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import xarray as xr

from tensorspec.core.data_models import TensorData

from .fanout import plan_jobs
from .geometry import (
    ANGSTROM_PER_BOHR,
    atoms_per_plane_max as _atoms_per_plane_max,
    hkl_to_abas_frame,
    layer_stack,
    parse_pot_geometry,
    pick_surface_site,
)
from .inputs import build_arpes_inputs, build_scf_inputs
from .jobs import JobSpec, LocalLauncher, RemoteLauncher, resolve_binary
from .outputs import ScfStatus, parse_spc, spc_to_datatree, spc_to_tensor, stitch_spc, stitch_points, write_points_json
from .pointwise import AnglePoint
from .params import ArpesParams, ScfParams
from .progress import arpes_rows_done, scf_progress

PathLike = Union[str, Path]

_STEM = {"scf": "kkrscf", "arpes": "kkrspec"}
_SUFFIX = "9.7"


def _is_remote(launcher: Any) -> bool:
    """True for a RemoteLauncher (or anything shaped like one): has
    `.cluster`, not `.bin_dir`. Real class check first, duck-type fallback
    so fakes in tests need not subclass RemoteLauncher."""
    if isinstance(launcher, RemoteLauncher):
        return True
    if isinstance(launcher, LocalLauncher):
        return False
    return hasattr(launcher, "cluster") and not hasattr(launcher, "bin_dir")


def _resolve_job_binary(kind: str, nproc: int, launcher: Any, bin_dir: Optional[PathLike]) -> str:
    if _is_remote(launcher):
        from tensorspec.core.compute import cluster_paths

        stem = _STEM[kind]
        name = f"{stem}{_SUFFIX}MPI" if nproc > 1 else f"{stem}{_SUFFIX}"
        return cluster_paths.sprkkr_binary(launcher.cluster, name)
    bd = bin_dir if bin_dir is not None else getattr(launcher, "bin_dir")
    return resolve_binary(kind, nproc, bd)


def _remote_rows_done(handle: Any, remote_path: str) -> int:
    """Count data rows of a remote .spc via ssh (rows = lines not starting with
    a letter/'#'; header lines all start with a letter or '#')."""
    try:
        ssh = handle.launcher.connect()
        out, _ = handle.launcher._run(
            ssh, f"grep -cE '^ *-?[0-9]' {remote_path} 2>/dev/null || echo 0"
        )
        return int(out.strip().splitlines()[-1]) if out.strip() else 0
    except Exception:
        return 0


def _sync_remote_log(handle: Any, local_log: Path, n: int = 400) -> None:
    """Pull remote log progress into the local staging log so the local parsers
    (scf_progress) see live progress.

    SPR-KKR reprints long core-state tables every SCF iteration, so a plain
    tail window may never contain the `N ERR ... EF ...` summary lines. We
    therefore fetch: all iteration summary / convergence lines (grep) + the
    last ``n`` lines.
    """
    try:
        ssh = handle.launcher.connect()
        lp = handle.log_path
        cmd = (
            f"grep -E '^ *[0-9]+ ERR |cycle converged|SCF - cycle|STOP|rror' {lp} 2>/dev/null | tail -n 60; "
            f"echo '#--- tail ---'; tail -n {n} {lp} 2>/dev/null"
        )
        text, _ = handle.launcher._run(ssh, cmd)
        if text:
            local_log.parent.mkdir(parents=True, exist_ok=True)
            local_log.write_text(text)
    except Exception:
        pass


def _is_running(handle: Any) -> bool:
    if hasattr(handle, "is_running"):
        return bool(handle.is_running())
    return handle.poll() is None


def _log_tail(path: PathLike, n_chars: int = 4000) -> str:
    p = Path(path)
    if not p.exists():
        return f"<no log at {p}>"
    text = p.read_text(errors="ignore")
    return text[-n_chars:]


def _set_potfil(inp_path: PathLike, name: str) -> None:
    """Rewrite the ``POTFIL=...`` line of an already-written .inp to the bare
    filename ``name``. Used for a remote run: ``build_arpes_inputs`` writes
    the *local* pot path into POTFIL, but on the cluster the pot sits next
    to the .inp in the same subdir, so it must be referenced by name only.
    """
    inp_path = Path(inp_path)
    lines = inp_path.read_text().splitlines()
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("POTFIL") and "=" in stripped:
            prefix = line[: line.index("POTFIL")]
            out.append(f"{prefix}POTFIL={name}")
        else:
            out.append(line)
    inp_path.write_text("\n".join(out) + "\n")


# ---------------------------------------------------------------------------
# SCF
# ---------------------------------------------------------------------------


@dataclass
class ScfResult:
    pot_path: str
    status: ScfStatus
    workdir: str
    log_path: str
    wall_s: float


def run_scf(
    structure: Any,
    params: ScfParams,
    workdir: PathLike,
    launcher: Any,
    nproc: int = 1,
    bin_dir: Optional[PathLike] = None,
    wait: bool = True,
    poll_cb: Optional[Callable[[ScfStatus], None]] = None,
    remote_workdir: Optional[str] = None,
):
    """Build SCF inputs, launch kkrscf, (optionally) wait for it, return ScfResult.

    ``wait=False`` returns the raw launcher handle (LocalHandle/RemoteHandle)
    instead -- caller polls with ``progress.scf_progress(log_path)`` itself.

    ``workdir`` is the LOCAL staging dir (inputs written here, outputs
    fetched back here). For a remote launcher, ``remote_workdir`` (a path
    valid on the cluster) is required -- raises ``ValueError`` if None.
    """
    workdir = Path(workdir)
    scf_inputs = build_scf_inputs(structure, params, workdir)

    binary = _resolve_job_binary("scf", nproc, launcher, bin_dir)
    log_name = f"{params.dataset}_SCF.out"

    remote = _is_remote(launcher)
    if remote and remote_workdir is None:
        raise ValueError("remote launcher requires remote_workdir= (cluster-side run dir)")
    job_workdir = remote_workdir if remote else str(workdir)

    job = JobSpec(
        kind="scf",
        workdir=job_workdir,
        inp_name=scf_inputs.inp_path.name,
        binary=binary,
        nproc=nproc,
        mpi=nproc > 1,
        log_name=log_name,
    )

    if remote:
        launcher.upload([scf_inputs.inp_path, scf_inputs.pot_path], remote_workdir)

    # Timer starts BEFORE launch: a remote launch may block until the job ends
    # (ssh channel held open), and that time is real wall time.
    start = time.time()
    handle = launcher.launch(job)
    log_path = workdir / log_name

    if not wait:
        return handle

    poll_s = 3.0 if remote else 1.0
    while True:
        if remote:
            _sync_remote_log(handle, log_path)
        status = scf_progress(log_path)
        if poll_cb is not None:
            poll_cb(status)
        if not _is_running(handle):
            break
        time.sleep(poll_s)
    wall_s = time.time() - start

    if remote:
        handle.fetch([log_name, f"{params.dataset}.pot_new"], str(workdir))

    status = scf_progress(log_path)
    pot_path = workdir / f"{params.dataset}.pot_new"
    if not pot_path.exists():
        raise RuntimeError(
            f"SPR-KKR SCF did not produce {pot_path.name} in {workdir}. "
            f"Log tail:\n{_log_tail(log_path)}"
        )

    return ScfResult(
        pot_path=str(pot_path),
        status=status,
        workdir=str(workdir),
        log_path=str(log_path),
        wall_s=wall_s,
    )


# ---------------------------------------------------------------------------
# Surface geometry (shared by GUI B3 + CLI e2e -- design doc §0)
# ---------------------------------------------------------------------------


@dataclass
class SurfaceGeometry:
    hkl_abas: Tuple[int, int, int]
    iq_at_surf: int
    planes_summary: str
    atoms_per_plane_max: int


def resolve_surface_geometry(
    pot_path: PathLike,
    hkl: Tuple[float, float, float],
    cif_lattice_matrix: Any = None,
    iq_at_surf: Optional[int] = None,
) -> SurfaceGeometry:
    """Work out the raw-ABAS-frame Miller index + surface IQ from a .pot.

    ``hkl`` is read in the CIF conventional-cell frame when
    ``cif_lattice_matrix`` is given (converted via ``hkl_to_abas_frame``);
    otherwise ``hkl`` is assumed to already be the raw ABAS frame.
    ``iq_at_surf``: an int (nonzero) pins the surface site; ``None``/``0``
    auto-picks the topmost non-vacancy site (``pick_surface_site``).
    """
    pot_geom = parse_pot_geometry(pot_path)
    if cif_lattice_matrix is not None:
        abas_cart = pot_geom.abas * pot_geom.alat_bohr * ANGSTROM_PER_BOHR
        hkl_abas = hkl_to_abas_frame(cif_lattice_matrix, hkl, abas_cart)
    else:
        hkl_abas = tuple(int(round(v)) for v in hkl)

    iq = int(iq_at_surf) if iq_at_surf else pick_surface_site(pot_geom, hkl_abas)

    planes = layer_stack(pot_geom, hkl_abas)
    lines = []
    for i, pl in enumerate(planes[:5]):
        species = ",".join(sorted(set(pl.species)))
        lines.append(f"plane {i}: proj={pl.proj_alat:.4f} n={len(pl.iqs)} species=[{species}]")
    planes_summary = "; ".join(lines)

    return SurfaceGeometry(
        hkl_abas=hkl_abas,
        iq_at_surf=iq,
        planes_summary=planes_summary,
        atoms_per_plane_max=_atoms_per_plane_max(pot_geom, hkl_abas),
    )


# ---------------------------------------------------------------------------
# ARPES
# ---------------------------------------------------------------------------


@dataclass
class ArpesResult:
    dataset: xr.Dataset
    tensor: TensorData
    tree: Any
    params: ArpesParams
    wall_s: float
    spc_paths: List[str] = field(default_factory=list)
    geometry: Optional[SurfaceGeometry] = None


@dataclass
class _SubJob:
    handle: Any
    job: JobSpec
    subdir: Path
    sub_params: ArpesParams
    expected_spc: str
    expected_log: str


class ArpesRunHandle:
    """Handle to 1..N launched ARPES jobs (mpi single job, or energy chunks)."""

    def __init__(
        self,
        subjobs: List[_SubJob],
        params: ArpesParams,
        launcher: Any,
        start_time: float,
        progress_cb: Optional[Callable[[float], None]] = None,
        geometry: Optional[SurfaceGeometry] = None,
        points: Optional[List[AnglePoint]] = None,
        deflector_deg: float = 0.0,
    ):
        self.handles = subjobs
        self.params = params
        self._launcher = launcher
        self._start_time = start_time
        self._progress_cb = progress_cb
        self._geometry = geometry
        self._points = points
        self._deflector_deg = deflector_deg

    def fraction_done(self) -> float:
        expected = sum(sj.sub_params.n_points for sj in self.handles)
        if expected <= 0:
            return 0.0
        if _is_remote(self._launcher):
            done = sum(
                _remote_rows_done(sj.handle, f"{sj.job.workdir}/{sj.expected_spc}")
                for sj in self.handles
            )
        else:
            done = sum(
                arpes_rows_done(sj.subdir / sj.expected_spc) for sj in self.handles
            )
        frac = min(1.0, done / expected)
        if self._progress_cb is not None:
            self._progress_cb(frac)
        return frac

    def is_done(self) -> bool:
        return all(not _is_running(sj.handle) for sj in self.handles)

    def kill(self) -> None:
        for sj in self.handles:
            if hasattr(sj.handle, "kill"):
                sj.handle.kill()

    def eta_s(self, eta_model) -> float:
        nproc_total = max(1, sum(getattr(sj.job, "nproc", 1) for sj in self.handles))
        total = eta_model.estimate_seconds(self.params, nproc_total)
        return max(0.0, total * (1.0 - self.fraction_done()))

    def collect(self) -> ArpesResult:
        poll_s = 3.0 if _is_remote(self._launcher) else 1.0
        while not self.is_done():
            self.fraction_done()
            time.sleep(poll_s)
        self.fraction_done()
        wall_s = time.time() - self._start_time

        remote = _is_remote(self._launcher)
        datasets = []
        spc_paths: List[str] = []
        inputs_text = ""
        for sj in self.handles:
            if remote:
                sj.handle.fetch([sj.expected_spc, sj.expected_log], str(sj.subdir))
            spc_path = sj.subdir / sj.expected_spc
            ds = parse_spc(str(spc_path), phi_range=sj.sub_params.phi_e)
            datasets.append(ds)
            spc_paths.append(str(spc_path))
            if not inputs_text:
                inp_path = sj.subdir / f"{sj.sub_params.dataset}.inp"
                if inp_path.exists():
                    inputs_text = inp_path.read_text()

        if self._points:
            merged = stitch_points(datasets, self._points, self._deflector_deg)
        else:
            merged = stitch_spc(datasets) if len(datasets) > 1 else datasets[0]

        host = getattr(self._launcher, "cluster", {}).get("host", "local") if remote else "local"
        meta = {
            "dataset": self.params.dataset,
            "hv_eV": self.params.hv_eV,
            "pol_p": self.params.pol_p,
            "ework_eV": self.params.ework_eV,
            "n_layer": self.params.n_layer,
            "EF_Ry": merged.attrs.get("EF_Ry"),
            "host": host,
            "wall_s": wall_s,
            "inputs_text": inputs_text,
            "pointwise": bool(self._points),
            "deflector_deg": self._deflector_deg,
        }

        tensor = spc_to_tensor(merged, meta)
        tree = spc_to_datatree(merged, meta)

        return ArpesResult(
            dataset=merged,
            tensor=tensor,
            tree=tree,
            params=self.params,
            wall_s=wall_s,
            spc_paths=spc_paths,
            geometry=self._geometry,
        )


def run_arpes(
    pot_path: PathLike,
    params: ArpesParams,
    workdir: PathLike,
    launcher: Any,
    nproc: int = 1,
    mode: str = "auto",
    mpi_available: bool = True,
    wait: bool = True,
    progress_cb: Optional[Callable[[float], None]] = None,
    remote_workdir: Optional[str] = None,
    cif_lattice: Any = None,
    extra_raw: Optional[dict] = None,
    angle_points: Optional[List[AnglePoint]] = None,
    deflector_deg: float = 0.0,
):
    """Fan out (mpi | energy-chunks), launch every job, then parse+stitch.

    ``wait=True`` (default) blocks and returns ``ArpesResult``; ``wait=False``
    returns an ``ArpesRunHandle`` immediately (call ``.collect()`` later).

    ``workdir`` is the LOCAL staging dir (each chunk's inputs are built in
    ``workdir/<subdir>``, outputs fetched back there). For a remote
    launcher, ``remote_workdir`` is required (``ValueError`` if None) and
    each chunk runs in ``remote_workdir/<subdir>``. If ``pot_path`` is not a
    file that exists locally, it is assumed to be a cluster-only path and is
    downloaded to ``workdir`` first -- ``build_arpes_inputs`` needs a local
    pot (ase2sprkkr reads it directly).

    ``params.hkl_frame == "cif"``: ``params.hkl`` is read as a Miller index
    in the CIF conventional-cell frame (requires ``cif_lattice`` -- a 3x3
    matrix, rows = direct lattice vectors, cartesian Angstrom, from the CIF
    exactly as loaded, before any primitive reduction) and is converted here
    to the raw ABAS frame + CRYS_VECS via ``resolve_surface_geometry``;
    ``params.iq_at_surf`` (None/0 -> auto-pick) is resolved the same way.

    ``angle_points``: when set, overrides ``mode`` and energy-chunking logic.
    Launches N independent jobs (one per point), each at all energies, with
    ``(Theta_i, Phi_i)`` set per point. Each job has ``nproc=1``; the N jobs
    themselves are the parallelism. One job per (Theta, Phi) at all energies
    is incompatible with energy-chunk fan-out.
    """
    workdir = Path(workdir)
    remote = _is_remote(launcher)
    start_time = time.time()  # before any launch (remote launch may block)
    if remote and remote_workdir is None:
        raise ValueError("remote launcher requires remote_workdir= (cluster-side run dir)")

    local_pot_path: PathLike = pot_path
    if remote and not Path(pot_path).is_file():
        local_pot_path = launcher.download([str(pot_path)], str(workdir))[0]

    # Resolve surface geometry (design doc §0) BEFORE plan_jobs -- plan_jobs
    # fans `params` out into per-chunk sub_params, so hkl/iq_at_surf must
    # already be final (raw ABAS + a resolved int) by the time it runs.
    geometry: Optional[SurfaceGeometry] = None
    if params.hkl_frame == "cif":
        if cif_lattice is None:
            raise ValueError("hkl_frame='cif' needs cif_lattice=")
        geometry = resolve_surface_geometry(
            local_pot_path, params.hkl, cif_lattice_matrix=cif_lattice, iq_at_surf=params.iq_at_surf
        )
        params = replace(params, hkl=geometry.hkl_abas, hkl_frame="abas", iq_at_surf=geometry.iq_at_surf)
    elif params.hkl_frame == "abas" and not params.iq_at_surf:
        geometry = resolve_surface_geometry(local_pot_path, params.hkl, iq_at_surf=None)
        params = replace(params, iq_at_surf=geometry.iq_at_surf)
    if geometry is not None:
        print(
            f"[geom] hkl_abas={geometry.hkl_abas} IQ_AT_SURF={geometry.iq_at_surf} "
            f"atoms_per_plane_max={geometry.atoms_per_plane_max}"
        )

    if angle_points:
        job_list = [
            (
                replace(params,
                        theta_e=(p.theta_e_deg, p.theta_e_deg), nt=1,
                        phi_e=(p.phi_e_deg, p.phi_e_deg), np_=1,
                        dataset=f"{params.dataset}_p{p.index:04d}"),
                f"{params.dataset}_p{p.index:04d}",
                1,
            )
            for p in angle_points
        ]
    else:
        job_list = list(plan_jobs(params, nproc, mode=mode, mpi_available=mpi_available).jobs)

    workdir.mkdir(parents=True, exist_ok=True)
    if angle_points:
        write_points_json(
            workdir / "pointwise_points.json", angle_points,
            meta={
                "deflector_deg": deflector_deg,
                "hv_eV": params.hv_eV,
                "ework_eV": params.ework_eV,
                "dataset": params.dataset,
                "nt": params.nt,
            }
        )
        min_k_par = min(p.k_par for p in angle_points)
        max_k_par = max(p.k_par for p in angle_points)
        min_theta = min(p.theta_e_deg for p in angle_points)
        max_theta = max(p.theta_e_deg for p in angle_points)
        min_phi = min(p.phi_e_deg for p in angle_points)
        max_phi = max(p.phi_e_deg for p in angle_points)
        print(f"[pointwise] N={len(angle_points)} Theta=[{min_theta:.2f},{max_theta:.2f}] Phi=[{min_phi:.2f},{max_phi:.2f}] min|k_par|={min_k_par:.4f}")
        if remote:
            # Sidecar must live next to the arpes_<ts>/ subdirs on the cluster too,
            # so GUI Fetch (arpes_panel.fetch_sprkkr_results) can rebuild lab axes.
            launcher.upload([workdir / "pointwise_points.json"], remote_workdir)

    subjobs: List[_SubJob] = []
    for sub_params, subdir_name, nproc_i in job_list:
        subdir = workdir / subdir_name
        arpes_inputs = build_arpes_inputs(local_pot_path, sub_params, subdir, extra_raw=extra_raw)
        binary = _resolve_job_binary("arpes", nproc_i, launcher, bin_dir=None)
        job_workdir = f"{remote_workdir}/{subdir_name}" if remote else str(subdir)
        job = JobSpec(
            kind="arpes",
            workdir=job_workdir,
            inp_name=arpes_inputs.inp_path.name,
            binary=binary,
            nproc=nproc_i,
            mpi=nproc_i > 1,
            log_name=arpes_inputs.expected_log,
            expected_outputs=[arpes_inputs.expected_spc],
        )
        if remote:
            _set_potfil(arpes_inputs.inp_path, Path(local_pot_path).name)
            launcher.upload([arpes_inputs.inp_path, local_pot_path], job_workdir)
        handle = launcher.launch(job)
        subjobs.append(
            _SubJob(
                handle=handle,
                job=job,
                subdir=subdir,
                sub_params=sub_params,
                expected_spc=arpes_inputs.expected_spc,
                expected_log=arpes_inputs.expected_log,
            )
        )

    run_handle = ArpesRunHandle(
        subjobs, params, launcher, start_time=start_time, progress_cb=progress_cb, geometry=geometry,
        points=angle_points, deflector_deg=deflector_deg
    )

    if not wait:
        return run_handle
    return run_handle.collect()
