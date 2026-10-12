"""SPR-KKR core subpackage. Zero Qt (sandy_rule.md layer split).

Runs the real Munich SPR-KKR 9.7 binaries (kkrscf / kkrspec) from TensorSpec.
Spec: docs/superpowers/specs/2026-09-08-sprkkr-in-gui-design.md

Modules
- params    dataclasses ScfParams / ArpesParams (real SPR-KKR keywords)
- inputs    build .inp/.pot via ase2sprkkr (lazy import; no run)
- outputs   parse *_data.spc -> xarray / TensorData / DataTree; SCF log parse
- pointwise  lab detector angles -> SPR-KKR (Theta,Phi) single-point jobs
- jobs      JobSpec + LocalLauncher / RemoteLauncher (we launch binaries ourselves)
- progress  fraction done, SCF status, ETA model
- fanout    energy-chunk / hv-list fan-out plans + full energy axis
- vault     converged-potential registry keyed by structure + ScfParams
"""
from .params import ArpesParams, ScfParams
from .inputs import (
    ArpesInputs,
    ScfInputs,
    build_arpes_inputs,
    build_scf_inputs,
    read_inp_keywords,
)
from .outputs import (
    ScfStatus,
    parse_scf_log,
    parse_spc,
    spc_to_datatree,
    spc_to_tensor,
    stitch_spc,
    read_points_json,
    stitch_points,
    write_points_json,
)
from .spc_results import spc_paths_to_results
from .pointwise import (
    AnglePoint,
    angle_points,
    k_vacuum,
    sprkkr_angles_to_lab_k,
)
from .jobs import (
    JobSpec,
    LocalHandle,
    LocalLauncher,
    RemoteHandle,
    RemoteLauncher,
    resolve_binary,
)
from .progress import (
    EtaModel,
    arpes_fraction_done,
    arpes_rows_done,
    format_eta,
    scf_progress,
)
from .fanout import FanoutPlan, full_energy_axis, plan_jobs, split_energy, split_hv
from .vault import Vault, VaultEntry, pot_key
from .workflow import (
    ArpesResult,
    ArpesRunHandle,
    ScfResult,
    SurfaceGeometry,
    resolve_surface_geometry,
    run_arpes,
    run_scf,
)
from .settings import SprkkrSettings, load_settings, save_settings, local_binaries_present

__all__ = [
    # params / inputs
    "ScfParams", "ArpesParams", "ScfInputs", "ArpesInputs",
    "build_scf_inputs", "build_arpes_inputs", "read_inp_keywords",
    # outputs
    "ScfStatus", "parse_scf_log", "parse_spc", "spc_to_datatree", "spc_to_tensor", "stitch_spc",
    "read_points_json", "stitch_points", "write_points_json", "spc_paths_to_results",
    # pointwise
    "AnglePoint", "angle_points", "k_vacuum", "sprkkr_angles_to_lab_k",
    # jobs
    "JobSpec", "LocalLauncher", "LocalHandle", "RemoteLauncher", "RemoteHandle", "resolve_binary",
    # progress
    "EtaModel", "arpes_fraction_done", "arpes_rows_done", "format_eta", "scf_progress",
    # fanout
    "FanoutPlan", "full_energy_axis", "plan_jobs", "split_energy", "split_hv",
    # vault
    "Vault", "VaultEntry", "pot_key",
    # workflow
    "ScfResult", "run_scf", "ArpesResult", "ArpesRunHandle", "run_arpes",
    "SurfaceGeometry", "resolve_surface_geometry",
    # settings
    "SprkkrSettings", "load_settings", "save_settings", "local_binaries_present",
]
