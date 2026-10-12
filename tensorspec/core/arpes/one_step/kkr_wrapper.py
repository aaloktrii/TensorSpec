"""B3 engine wrapper: real Munich SPR-KKR one-step ARPES, via workflow.run_arpes.

Zero Qt (sandy_rule.md layer 1). Router (arpes_engine.py) calls
``KKRWrapper.run_simulation(crystal_data, experiment_kwargs)`` unchanged.

``experiment_kwargs`` unit convention for this engine (differs from B1
Chinook, which works in k-space kx/ky): ``k_bounds['X']``/``['Y']`` here are
detector ANGLES in degrees -- theta (slit) and phi (deflection) -- because
that is what SPR-KKR's SPEC_EL THETA/PHI grid wants. ``photon_energy`` is
still eV, ``work_function`` still eV.

In pointwise mode (pointwise=True), ``k_bounds['X']`` is the lab SLIT angle
sweep and ``k_bounds['Y']`` is ignored in favor of ``deflector_angle`` (which
falls back to ``k_bounds['Y'][0]`` if not set). Each lab slit angle becomes
its own SPR-KKR job with unique (THETA_i, PHI_i) per point. The result theta
axis is lab slit angles, phi axis is the deflector value. SPEC_EL PHI is then
per-point, never the raw deflector.
"""
from __future__ import annotations

import dataclasses
import tempfile
from typing import Any, Dict, Optional

import numpy as np

from tensorspec.core.dft.sprkkr.jobs import LocalLauncher
from tensorspec.core.dft.sprkkr.params import ArpesParams
from tensorspec.core.dft.sprkkr.workflow import run_arpes

_ARPES_FIELDS = {f.name for f in dataclasses.fields(ArpesParams)}
_EXPLICIT_FIELDS = {
    "hv_eV", "pol_p", "theta_e", "nt", "phi_e", "np_",
    "e_min_eV", "e_max_eV", "ne", "ework_eV", "hkl",
    "iq_at_surf", "hkl_frame",
}


def _map_iq_at_surf(value) -> Optional[int]:
    """GUI/CLI iq_at_surf -> Optional[int]. None/0/"auto" -> None (auto-pick,
    resolved later by workflow.resolve_surface_geometry)."""
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() == "auto":
            return None
        return int(value)
    ivalue = int(value)
    return ivalue if ivalue else None


def _map_polarization(value: Optional[str]) -> str:
    """GUI polarization text -> SPR-KKR POL_P (P | S | C+ | C-)."""
    if not value:
        return "P"
    text = str(value).strip()
    if text in ("P", "S", "C+", "C-"):
        return text
    low = text.lower()
    if "vertical" in low or "s-pol" in low or low == "s":
        return "S"
    if "horizontal" in low or "p-pol" in low or low == "p":
        return "P"
    if "right" in low or low in ("cr", "c+"):
        return "C+"
    if "left" in low or low in ("cl", "c-"):
        return "C-"
    return "P"  # "Linear Arbitrary" or unknown -> linear P default


def _map_k_bounds(k_bounds: Optional[Dict[str, Any]], defaults: ArpesParams):
    theta_e, nt, phi_e, np_ = defaults.theta_e, defaults.nt, defaults.phi_e, defaults.np_
    if k_bounds:
        x = k_bounds.get("X")
        if x is not None:
            theta_e = (float(x[0]), float(x[1]))
            if len(x) > 2:
                nt = int(x[2])
        y = k_bounds.get("Y")
        if y is not None:
            phi_e = (float(y[0]), float(y[1]))
            if len(y) > 2:
                np_ = int(y[2])
    # min == max -> single sample (same rule as chinook_wrapper). Otherwise
    # SPR-KKR would compute NT/NP identical points (seen: PHI=0.0 with NP=10).
    if theta_e[0] == theta_e[1]:
        nt = 1
    if phi_e[0] == phi_e[1]:
        np_ = 1
    return theta_e, nt, phi_e, np_


def _map_energy(kwargs: Dict[str, Any], defaults: ArpesParams):
    e_min, e_max, ne = defaults.e_min_eV, defaults.e_max_eV, defaults.ne
    k_bounds = kwargs.get("k_bounds") or {}
    e = k_bounds.get("E")
    if e is not None:
        e_min, e_max = float(e[0]), float(e[1])
        if len(e) > 2:
            ne = int(e[2])
    e_range = kwargs.get("e_range")
    if e_range is not None:
        e_min, e_max = float(e_range[0]), float(e_range[1])
        if len(e_range) > 2:
            ne = int(e_range[2])
    if "e_min" in kwargs:
        e_min = float(kwargs["e_min"])
    if "e_max" in kwargs:
        e_max = float(kwargs["e_max"])
    if "e_steps" in kwargs:
        ne = int(kwargs["e_steps"])
    if e_min == e_max:
        ne = 1  # single energy (Fermi map) -- same rule as chinook_wrapper
    return e_min, e_max, ne


def _map_deflector(kwargs: Dict[str, Any], params: ArpesParams) -> float:
    """Map GUI deflector angle, falling back to phi_e[0] for back-compat."""
    deflector = kwargs.get("deflector_angle")
    if deflector is not None:
        return float(deflector)
    return params.phi_e[0]


def _build_angle_points(kwargs: Dict[str, Any], params: ArpesParams):
    """Build AnglePoint list when pointwise mode is on, else return []."""
    from tensorspec.core.dft.sprkkr.pointwise import angle_points as build_angle_points

    if not kwargs.get("pointwise"):
        return []

    return build_angle_points(
        params.theta_e, params.nt,
        hv_eV=params.hv_eV, work_function_eV=params.ework_eV,
        deflector_deg=_map_deflector(kwargs, params),
        slit_rot_deg=float(kwargs.get("slit_angle", 0.0)),
        manip_theta_deg=float(kwargs.get("manip_theta", 0.0)),
        manip_azimuth_deg=float(kwargs.get("manip_azimuth", 0.0)),
        manip_tilt_deg=float(kwargs.get("manip_tilt", 0.0)),
        ref_energy_eV=float(kwargs.get("ref_energy_eV", 0.0)),
        phi_offset_deg=float(kwargs.get("phi_offset_deg", 0.0)),
    )


def _build_arpes_params(kwargs: Dict[str, Any]) -> ArpesParams:
    """Map GUI ``experiment_kwargs`` -> ``ArpesParams``. Every key optional."""
    defaults = ArpesParams()
    theta_e, nt, phi_e, np_ = _map_k_bounds(kwargs.get("k_bounds"), defaults)
    e_min, e_max, ne = _map_energy(kwargs, defaults)
    pol_p = _map_polarization(kwargs.get("polarization", defaults.pol_p))

    passthrough = {
        k: v for k, v in kwargs.items() if k in _ARPES_FIELDS and k not in _EXPLICIT_FIELDS
    }

    return dataclasses.replace(
        defaults,
        hv_eV=float(kwargs.get("photon_energy", defaults.hv_eV)),
        pol_p=pol_p,
        theta_e=theta_e,
        nt=nt,
        phi_e=phi_e,
        np_=np_,
        e_min_eV=e_min,
        e_max_eV=e_max,
        ne=ne,
        ework_eV=float(kwargs.get("work_function", defaults.ework_eV)),
        hkl=tuple(kwargs.get("hkl", defaults.hkl)),
        hkl_frame=kwargs.get("hkl_frame", defaults.hkl_frame),
        iq_at_surf=_map_iq_at_surf(kwargs.get("iq_at_surf", defaults.iq_at_surf)),
        **passthrough,
    )


def _fermi_dirac_weights(energy_eV: np.ndarray, temperature_K: float) -> np.ndarray:
    """1/(exp(E/kT)+1); E in eV rel. to EF. T<=0 -> hard step at E=0."""
    energy_eV = np.asarray(energy_eV, dtype=float)
    if temperature_K is None or temperature_K <= 0:
        return np.where(energy_eV < 0, 1.0, 0.0)
    kT = 8.617333e-5 * float(temperature_K)
    with np.errstate(over="ignore"):
        return 1.0 / (np.exp(energy_eV / kT) + 1.0)


class KKRWrapper:
    """
    Engine wrapper for SPR-KKR Multiple Scattering ARPES calculations (B3).
    Translates TensorSpec experimental parameters into a real kkrspec9.7 run.
    """

    def __init__(self):
        self.engine_name = "SPR-KKR (Multiple Scattering)"

    def run_simulation(self, crystal_data, experiment_kwargs):
        kwargs = dict(experiment_kwargs or {})

        pot_path = kwargs.get("pot_path")
        if not pot_path:
            raise ValueError("need converged potential; run SPR-KKR SCF first")

        params = _build_arpes_params(kwargs)
        params.validate()

        points = _build_angle_points(kwargs, params)
        if points and params.np_ != 1:
            raise ValueError("pointwise mode needs nt >= 1 and np_ == 1")

        workdir = kwargs.get("workdir")
        if workdir is None:
            workdir = tempfile.mkdtemp(prefix="kkr_arpes_")

        launcher = kwargs.get("launcher")
        if launcher is None:
            bin_dir = kwargs.get("bin_dir")
            if bin_dir is None:
                raise ValueError("need launcher= or bin_dir= to run SPR-KKR ARPES")
            launcher = LocalLauncher(bin_dir=bin_dir)

        nproc = int(kwargs.get("nproc", 1))
        mode = kwargs.get("mode", "auto")
        mpi_available = bool(kwargs.get("mpi_available", True))
        wait = not bool(kwargs.get("async_", False))
        remote_workdir = kwargs.get("remote_workdir")
        cif_lattice = kwargs.get("cif_lattice")

        outcome = run_arpes(
            pot_path=pot_path,
            params=params,
            workdir=workdir,
            launcher=launcher,
            nproc=nproc,
            mode=mode,
            mpi_available=mpi_available,
            wait=wait,
            remote_workdir=remote_workdir,
            cif_lattice=cif_lattice,
            angle_points=points or None,
            deflector_deg=_map_deflector(kwargs, params) if points else 0.0,
        )

        if not wait:
            return {"handle": outcome, "params": params}

        result = outcome
        # SARPES: spin_filter "up"|"down" picks I_up / I_dn (spin axis set via
        # ArpesParams.pol_e = PX|PY|PZ); None -> I_tot.
        spin_filter = kwargs.get("spin_filter")
        var = {"up": "I_up", "down": "I_dn"}.get(spin_filter or "", "I_tot")
        # dataset dims are (energy, theta, phi); B1/GUI layout is (theta, phi, energy)
        intensity = np.transpose(result.dataset[var].values, (1, 2, 0))

        temperature_K = kwargs.get("temperature_K")
        apply_fermi = bool(kwargs.get("apply_fermi", True))
        if temperature_K is not None and apply_fermi:
            energy_axis = result.dataset["energy"].values
            fermi = _fermi_dirac_weights(energy_axis, float(temperature_K))
            # intensity is (theta, phi, energy) here -- fermi rides the last axis.
            intensity = intensity * fermi[np.newaxis, np.newaxis, :]
            # tensor.value keeps its own (energy-first) shape convention; do NOT
            # touch dataset/tree raw arrays -- build a fresh cut array only.
            cut = result.dataset[var].values * fermi[:, np.newaxis, np.newaxis]
            if np.ndim(result.tensor.value) == 2:
                cut = cut[:, :, 0]
            result.tensor.value = cut
            result.tensor.metadata["fermi_cutoff_K"] = float(temperature_K)

        return {
            "intensity_broadened": intensity,
            # 1-D axes straight from the dataset coords, for GUI on_simulation_finished
            "energy": result.dataset["energy"].values,
            "theta": result.dataset["theta"].values,
            "phi": result.dataset["phi"].values,
            "tensor": result.tensor,
            "datatree": result.tree,
            "dataset": result.dataset,
            "params": result.params,
            "result": result,
            "geometry": result.geometry,
            "angle_points": points,
        }
