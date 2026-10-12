"""
Shared θ,φ → K_bulk kinematics and Chinook ARPES execution.

Used by local chinook_wrapper and remote runner so both paths
apply the same hkl projection, V0 refraction, and bulk-frame polarization.
"""
from __future__ import annotations

import io
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import importlib.util
import os

import numpy as np

try:
    from tensorspec.core.arpes.one_step.fresnel import apply_fresnel_to_A_lab
    from tensorspec.core.arpes.one_step.photon_momentum import photon_q_lab
except ImportError:
    # Remote jobs upload flat .py files beside this module (no package).
    def _load_co_uploaded(name: str):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{name}.py")
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    apply_fresnel_to_A_lab = _load_co_uploaded("fresnel").apply_fresnel_to_A_lab
    photon_q_lab = _load_co_uploaded("photon_momentum").photon_q_lab

_CHINOOK_PATCHED = False


def apply_chinook_runtime_patches() -> None:
    """Idempotent Chinook compatibility patches (Wannier Z=1, radint cutoff)."""
    global _CHINOOK_PATCHED
    if _CHINOOK_PATCHED:
        return
    import collections
    import collections.abc

    if not hasattr(collections, "Iterable"):
        collections.Iterable = collections.abc.Iterable

    import chinook.electron_configs as econ
    import chinook.radint_lib as radint_lib

    _orig_shield_split = econ.shield_split

    def safe_shield_split(shield_string):
        if not shield_string:
            return []
        return _orig_shield_split(shield_string)

    econ.shield_split = safe_shield_split

    original_Z_eff = econ.Z_eff
    _dummy_stream = io.StringIO()

    def patched_Z_eff(Z, orb):
        old_stdout = sys.stdout
        sys.stdout = _dummy_stream
        try:
            result = original_Z_eff(Z, orb)
        finally:
            sys.stdout = old_stdout
        if result is None:
            z_eff_db = {
                37: 2.20,
                38: 2.85,
                39: 3.00,
                40: 3.65,
                41: 4.30,
                42: 4.95,
                43: 5.60,
                44: 6.25,
                45: 6.90,
                46: 7.55,
                47: 8.20,
                48: 8.85,
                49: 5.00,
                50: 5.65,
                51: 6.30,
                52: 6.95,
                53: 7.60,
                54: 8.25,
                55: 2.20,
                56: 2.85,
                **{z: 3.00 + (z - 57) * 0.35 for z in range(57, 72)},
                72: 3.65,
                73: 4.30,
                74: 4.95,
                75: 5.60,
                76: 6.25,
                77: 6.90,
                78: 7.55,
                79: 8.20,
                80: 8.85,
                81: 5.00,
                82: 5.65,
                83: 6.30,
                84: 6.95,
                85: 7.60,
                86: 8.25,
            }
            return z_eff_db.get(Z, 4.5)
        return result

    econ.Z_eff = patched_Z_eff

    original_find_cutoff = radint_lib.find_cutoff

    def safe_find_cutoff(integrand):
        try:
            return original_find_cutoff(integrand)
        except UnboundLocalError:
            return 30.0

    radint_lib.find_cutoff = safe_find_cutoff
    _CHINOOK_PATCHED = True


def get_hkl_surface_frame(hkl, recip_matrix, azimuthal_ref=None):
    """Orthogonal surface basis for cleavage plane (h,k,l)."""
    h, k, l = hkl
    normal = h * recip_matrix[0] + k * recip_matrix[1] + l * recip_matrix[2]
    dist = np.linalg.norm(normal)
    if dist == 0:
        return np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])
    normal = normal / dist

    if azimuthal_ref is not None:
        up_vector = azimuthal_ref - np.dot(azimuthal_ref, normal) * normal
    else:
        b_star = recip_matrix[1]
        a_star = recip_matrix[0]
        b_star_norm = b_star / np.linalg.norm(b_star)
        default_up = b_star if abs(np.dot(b_star_norm, normal)) < 0.99 else a_star
        up_vector = default_up - np.dot(default_up, normal) * normal

    up_vector = up_vector / np.linalg.norm(up_vector)
    return normal, up_vector


def normalize_k_bounds(k_bounds: Mapping[str, list]) -> Dict[str, list]:
    kb = {key: list(val) for key, val in k_bounds.items()}
    if kb["X"][0] == kb["X"][1]:
        kb["X"][2] = 1
    if kb["Y"][0] == kb["Y"][1]:
        kb["Y"][2] = 1
    if kb["E"][0] == kb["E"][1]:
        kb["E"][2] = 1
    return kb


def compute_A_lab(pol_str: str, incidence_angle: float, lin_pol_angle: float = 45.0) -> np.ndarray:
    inc_rad = np.radians(incidence_angle)
    lin_ang = np.radians(lin_pol_angle)
    if "Horizontal" in pol_str:
        return np.array([np.cos(inc_rad), -np.sin(inc_rad), 0.0])
    if "Vertical" in pol_str:
        return np.array([0.0, 0.0, 1.0])
    if "Arbitrary" in pol_str:
        return np.cos(lin_ang) * np.array(
            [np.cos(inc_rad), -np.sin(inc_rad), 0.0]
        ) + np.sin(lin_ang) * np.array([0.0, 0.0, 1.0])
    if "Right" in pol_str:
        return (
            np.array([np.cos(inc_rad), -np.sin(inc_rad), 0.0])
            + 1j * np.array([0.0, 0.0, 1.0])
        ) / np.sqrt(2)
    return (
        np.array([np.cos(inc_rad), -np.sin(inc_rad), 0.0])
        - 1j * np.array([0.0, 0.0, 1.0])
    ) / np.sqrt(2)


def sample_to_bulk_frame(
    A_lab: np.ndarray,
    manip_theta: float,
    manip_azimuth: float,
    manip_tilt: float,
) -> np.ndarray:
    R_base = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    t_rad, a_rad, tilt_rad = (
        np.radians(manip_theta),
        np.radians(manip_azimuth),
        np.radians(manip_tilt),
    )
    R_z = np.array(
        [
            [np.cos(t_rad), -np.sin(t_rad), 0],
            [np.sin(t_rad), np.cos(t_rad), 0],
            [0, 0, 1],
        ]
    )
    R_y = np.array(
        [
            [np.cos(a_rad), 0, np.sin(a_rad)],
            [0, 1, 0],
            [-np.sin(a_rad), 0, np.cos(a_rad)],
        ]
    )
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(tilt_rad), -np.sin(tilt_rad)],
            [0, np.sin(tilt_rad), np.cos(tilt_rad)],
        ]
    )
    R_total = R_z @ R_y @ R_x @ R_base
    R_inv = np.linalg.inv(R_total)
    return R_inv @ A_lab


def build_k_bulk_mesh(
    k_bounds: Mapping[str, list],
    *,
    hv: float,
    work_function: float,
    inner_potential: float,
    slit_angle: float,
    manip_theta: float,
    manip_azimuth: float,
    manip_tilt: float,
    incidence_angle: float,
    polarization: str,
    hkl: Tuple[int, int, int],
    B_matrix: np.ndarray,
    lin_pol_angle: float = 45.0,
    fresnel_enabled: bool = True,
    optical_n: float = 1.0,
    optical_k: float = 0.0,
    include_photon_momentum: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, list], int, int, int, np.ndarray]:
    """Return K_BULK (3,Nk), A_bulk (3,), normalized kb, grid sizes, energy_axis.

    ``A`` path: vacuum ``compute_A_lab`` → optional Fresnel local field
    (``n = optical_n + 1j*optical_k``; ``n=1`` is vacuum parity) → manip + hkl.

    ``K`` order:
    1. vacuum photoelectron k from angles
    2. inner-potential refraction
    3. manip + hkl → ``K_BULK``
    4. if ``include_photon_momentum``: ``K_BULK -= q_bulk`` (full 3-vector;
       ``q_lab`` rotated with the same manip + hkl maps as ``K`` / ``A``)
    """
    kb = normalize_k_bounds(k_bounds)
    num_x, num_y, num_e = int(kb["X"][2]), int(kb["Y"][2]), int(kb["E"][2])
    energy_axis = np.linspace(kb["E"][0], kb["E"][1], num_e)

    A_vac = compute_A_lab(polarization, incidence_angle, lin_pol_angle)
    if fresnel_enabled:
        n_opt = complex(float(optical_n), float(optical_k))
        A_lab = apply_fresnel_to_A_lab(A_vac, incidence_angle, n=n_opt)
    else:
        A_lab = A_vac
    A_sample = sample_to_bulk_frame(A_lab, manip_theta, manip_azimuth, manip_tilt)

    E_kin = max(hv - work_function, 0.1)
    k_radius = 0.512316 * np.sqrt(E_kin)

    theta_arr = np.radians(np.linspace(kb["X"][0], kb["X"][1], num_x))
    phi_arr = np.radians(np.linspace(kb["Y"][0], kb["Y"][1], num_y))
    THETA, PHI = np.meshgrid(theta_arr, phi_arr, indexing="ij")

    K_SLIT = k_radius * np.sin(THETA)
    K_DEFL = k_radius * np.sin(PHI)
    k_slit_flat = K_SLIT.flatten(order="C")
    k_defl_flat = K_DEFL.flatten(order="C")

    slit_rad = np.radians(slit_angle)
    k_lab_x = k_slit_flat * np.cos(slit_rad) - k_defl_flat * np.sin(slit_rad)
    k_lab_z = k_slit_flat * np.sin(slit_rad) + k_defl_flat * np.cos(slit_rad)

    k_vac_sq = 0.262465 * E_kin
    k_lab_y_sq = k_vac_sq - (k_lab_x**2 + k_lab_z**2)
    k_lab_y = np.sqrt(np.clip(k_lab_y_sq, 0.0, None))
    K_LAB_VAC = np.vstack([k_lab_x, k_lab_y, k_lab_z])

    R_base = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    t_rad, a_rad, tilt_rad = (
        np.radians(manip_theta),
        np.radians(manip_azimuth),
        np.radians(manip_tilt),
    )
    R_z = np.array(
        [
            [np.cos(t_rad), -np.sin(t_rad), 0],
            [np.sin(t_rad), np.cos(t_rad), 0],
            [0, 0, 1],
        ]
    )
    R_y = np.array(
        [
            [np.cos(a_rad), 0, np.sin(a_rad)],
            [0, 1, 0],
            [-np.sin(a_rad), 0, np.cos(a_rad)],
        ]
    )
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(tilt_rad), -np.sin(tilt_rad)],
            [0, np.sin(tilt_rad), np.cos(tilt_rad)],
        ]
    )
    R_total = R_z @ R_y @ R_x @ R_base
    R_inv = np.linalg.inv(R_total)
    K_SAMPLE = R_inv @ K_LAB_VAC
    K_SAMPLE[2] = np.sqrt(K_SAMPLE[2] ** 2 + 0.262465 * inner_potential)

    B_matrix = np.asarray(B_matrix, dtype=float)
    Z_surf, Y_surf = get_hkl_surface_frame(
        hkl, B_matrix, azimuthal_ref=np.array([0.0, 1.0, 0.0])
    )
    X_surf = np.cross(Y_surf, Z_surf)
    R_hkl_to_bulk = np.column_stack((X_surf, Y_surf, Z_surf))
    K_BULK = R_hkl_to_bulk @ K_SAMPLE
    A_bulk = R_hkl_to_bulk @ A_sample

    if include_photon_momentum:
        q_lab = photon_q_lab(hv, incidence_angle)
        q_sample = sample_to_bulk_frame(q_lab, manip_theta, manip_azimuth, manip_tilt)
        q_bulk = R_hkl_to_bulk @ q_sample
        K_BULK = K_BULK - np.asarray(q_bulk, dtype=float).reshape(3, 1)

    return K_BULK, A_bulk, kb, num_x, num_y, num_e, energy_axis


def shift_k_bulk_along_surface_normal(
    K_BULK: np.ndarray,
    kz_shift: float,
    hkl: Tuple[int, int, int],
    B_matrix: np.ndarray,
) -> np.ndarray:
    """Shift bulk-frame k-mesh by Δk_z along surface normal ``Z_surf``.

    Uses the same ``get_hkl_surface_frame`` call as ``build_k_bulk_mesh`` so the
    escape-depth axis matches the cleavage normal for tilted hkl (e.g. (-2,0,1)).
    """
    if abs(float(kz_shift)) <= 0.0:
        return K_BULK
    Z_surf, _ = get_hkl_surface_frame(
        tuple(hkl),
        np.asarray(B_matrix, dtype=float),
        azimuthal_ref=np.array([0.0, 1.0, 0.0]),
    )
    K = np.asarray(K_BULK, dtype=float).copy()
    K += float(kz_shift) * Z_surf[:, None]
    return K


def lorentzian_kz_weights(halfwidth: float, npoints: int) -> Tuple[np.ndarray, np.ndarray]:
    """Discrete Lorentzian weights for incoherent k_z broadening.

    Critic wants Δk_z ≈ 1/λ_mfp ≈ 0.2 Å⁻¹ HWHM. Default off (``halfwidth<=0``)
    preserves legacy single-cut behavior. When ``halfwidth>0`` and ``npoints<=1``,
    uses an odd grid of 7 points so broadening is not a silent no-op.

    Escape-depth sampling shifts ``K_BULK`` along the surface normal; each pass
    also recomputes emission angles from the shifted mesh (detector-angle smear
    included). **Not** coherent LEED final-state interference.
    """
    hw = float(halfwidth)
    n = int(npoints)
    if hw <= 0.0:
        return np.array([0.0]), np.array([1.0])
    if n <= 1:
        n = 7  # halfwidth>0 with omitted/legacy npoints=1 → odd grid ≥3
    n = n if n % 2 == 1 else n + 1  # odd grid, include 0
    span = 3.0 * hw  # ±3 HWHM
    dk = np.linspace(-span, span, n)
    w = (hw / np.pi) / (dk**2 + hw**2)
    w = w / w.sum()
    return dk, w


def physics_from_experiment_kwargs(experiment_kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """Map GUI experiment_kwargs → physics dict for run_chinook_arpes."""
    kz_hw = float(experiment_kwargs.get("kz_halfwidth", 0.0))
    kz_np_default = 7 if kz_hw > 0.0 else 1
    return {
        "hv": float(experiment_kwargs.get("photon_energy", 21.2)),
        "work_function": float(experiment_kwargs.get("work_function", 4.5)),
        "inner_potential": float(experiment_kwargs.get("inner_potential", 15.0)),
        "temperature": float(experiment_kwargs.get("temperature", 10.0)),
        "incidence_angle": float(experiment_kwargs.get("incidence_angle", 55.0)),
        "polarization": str(experiment_kwargs.get("polarization", "Linear Horizontal")),
        "lin_pol_angle": float(experiment_kwargs.get("lin_pol_angle", 45.0)),
        "matrix_element_mode": str(
            experiment_kwargs.get("matrix_element_mode", "Full Matrix Elements")
        ),
        "manip_theta": float(experiment_kwargs.get("manip_theta", 0.0)),
        "manip_azimuth": float(experiment_kwargs.get("manip_azimuth", 0.0)),
        "manip_tilt": float(experiment_kwargs.get("manip_tilt", 0.0)),
        "hkl": tuple(experiment_kwargs.get("hkl", (0, 0, 1))),
        "slit_angle": float(experiment_kwargs.get("slit_angle", 0.0)),
        "se_width": float(experiment_kwargs.get("se_width", 0.01)),
        "res_E": float(experiment_kwargs.get("res_E", 0.02)),
        "res_k": float(experiment_kwargs.get("res_k", 0.02)),
        "rad_type": str(experiment_kwargs.get("rad_type", "slater")),
        "mfp": float(experiment_kwargs.get("mfp", 10.0)),
        "kz_halfwidth": kz_hw,
        "kz_npoints": int(experiment_kwargs.get("kz_npoints", kz_np_default)),
        "fresnel_enabled": bool(experiment_kwargs.get("fresnel_enabled", True)),
        "optical_n": float(experiment_kwargs.get("optical_n", 1.0)),
        "optical_k": float(experiment_kwargs.get("optical_k", 0.0)),
        "include_photon_momentum": bool(
            experiment_kwargs.get("include_photon_momentum", False)
        ),
    }


def build_arpes_dict_from_physics(physics, *, cube, energy_axis, A_bulk, is_full: bool):
    """Single place for chinook ARPES_dict keys (local + Grizzly shell).

    ``pol`` keeps the dtype of ``A_bulk`` (complex for CR/CL). For bulk (non-slab)
    bases, ``mfp`` only reweights intra-cell orbital depths; escape-depth physics
    comes from ``kz_halfwidth``.
    """
    return {
        "cube": cube,
        "ang": 0.0,
        "E": energy_axis,
        "hv": float(physics["hv"]),
        "W": float(physics["work_function"]),
        "Vo": float(physics["inner_potential"]),  # chinook key is Vo, not V0
        "T": float(physics.get("temperature", 10.0)),
        "pol": np.asarray(A_bulk),  # preserve complex128 for CR/CL
        "ME": bool(is_full),
        "SE": ["constant", float(physics.get("se_width", 0.01))],
        "resolution": {
            "E": float(physics.get("res_E", 0.02)),
            "k": float(physics.get("res_k", 0.02)),
        },
        "rad_type": str(physics.get("rad_type", "slater")),
        "mfp": float(physics.get("mfp", 10.0)),
    }


def _bare_intensity(ctx) -> np.ndarray:
    """Lorentzian spectral weight only — no Fermi–Dirac cutoff.

    Occupied-state cutoff belongs on real ME / Chinook–Grizzly ``spectral()``
    paths, not bare-band visualization.
    """
    exp = ctx["exp"]
    energy_axis = ctx["energy_axis"]
    num_x, num_y, num_e = ctx["num_x"], ctx["num_y"], ctx["num_e"]
    se_width = ctx["se_width"]
    gamma = se_width / 2.0
    diff = exp.val[:, :, np.newaxis] - energy_axis[np.newaxis, np.newaxis, :]
    spectral_weight = (gamma / np.pi) / (diff**2 + gamma**2)
    intensity_flat = np.sum(spectral_weight, axis=1)
    return intensity_flat.reshape((num_x, num_y, num_e), order="C")


def _reshape_me_intensity(output_maps, num_x, num_y, num_e) -> np.ndarray:
    expected_size = num_x * num_y * num_e
    if output_maps.size == 2 * expected_size:
        output_maps = np.sum(output_maps.reshape((2, expected_size)), axis=0)
    elif output_maps.ndim == 4 and output_maps.shape[0] == 2:
        output_maps = np.sum(output_maps, axis=0)
    return output_maps.reshape((num_x, num_y, num_e), order="C")


def _apply_dipole(intensity_3d, K_BULK, A_bulk, num_x, num_y):
    dipole_dot = (
        A_bulk[0] * K_BULK[0]
        + A_bulk[1] * K_BULK[1]
        + A_bulk[2] * K_BULK[2]
    )
    dipole_factor = np.abs(dipole_dot) ** 2
    dipole_factor = dipole_factor.reshape(num_x, num_y, order="C")
    return intensity_3d * dipole_factor[:, :, np.newaxis]


def _diag_phase_budget_bytes(device: str) -> int:
    """VRAM budget for inner diag Fourier phase buffer (per batch)."""
    try:
        import torch

        if str(device).startswith("cuda") and torch.cuda.is_available():
            free_b, _ = torch.cuda.mem_get_info()
            return max(256 * 1024**2, int(free_b * 0.35))
    except Exception:
        pass
    return 2 * 1024**3


def _grizzly_diagonalize_tb(
    tb_model, device: str, nk: int, *, Eonly: bool = False
):
    """GPU/CPU batched diag via GrizzlyME (replaces chinook TB.solve_H on large kmesh)."""
    from grizzly.hamiltonian import solve_H as grizzly_solve_H
    from grizzly.utils import to_numpy

    n_hops = sum(len(me.H) for me in tb_model.mat_els)
    n_hops = max(int(n_hops), 1)
    n_basis = int(getattr(tb_model, "n_basis", 0) or 0)
    if n_basis <= 0 and getattr(tb_model, "basis", None) is not None:
        n_basis = len(tb_model.basis)
    n_basis = max(n_basis, 1)

    phase_budget_bytes = _diag_phase_budget_bytes(device)
    # phases + contrib are both (Nk, Nh); add H(k) storage per k-point.
    bytes_per_k = n_hops * 16 * 2 + n_basis * n_basis * 16
    hop_limited = max(1, phase_budget_bytes // max(bytes_per_k, 1))
    nk_limited = max(256, (nk + 39) // 40) if nk > 2048 else nk
    chunk_size = min(hop_limited, nk_limited)
    if str(device).startswith("cuda"):
        if n_hops > 2_000_000:
            chunk_size = min(chunk_size, 4)
        elif n_hops > 500_000:
            chunk_size = min(chunk_size, 12)
    if chunk_size >= nk:
        chunk_size = 0
    if chunk_size > 1:
        print(
            f"GrizzlyME diag chunk: nk={nk} n_hops={n_hops} n_basis={n_basis} "
            f"chunk_size={chunk_size} phase_budget_MiB={phase_budget_bytes // (1024 * 1024)}",
            flush=True,
        )

    def _run(dev: str, cs: int):
        return grizzly_solve_H(tb_model, device=dev, chunk_size=cs, Eonly=Eonly)

    try:
        Eband, Evec = _run(device, chunk_size)
    except Exception as exc:
        oom = "out of memory" in str(exc).lower()
        if not (str(device).startswith("cuda") and oom):
            raise
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        if chunk_size > 1:
            cs_retry = max(1, chunk_size // 4)
            print(
                f"WARN: CUDA OOM at chunk_size={chunk_size}; retry chunk_size={cs_retry}",
                flush=True,
            )
            try:
                Eband, Evec = _run(device, cs_retry)
            except Exception:
                print("WARN: CUDA OOM — falling back to Grizzly CPU", flush=True)
                Eband, Evec = _run("cpu", 0)
        else:
            print("WARN: CUDA OOM — falling back to Grizzly CPU", flush=True)
            Eband, Evec = _run("cpu", 0)

    if Eonly:
        return to_numpy(Eband), None
    return to_numpy(Eband), to_numpy(Evec)


def _setup_kmesh_experiment(
    tb_model,
    k_bounds: Mapping[str, list],
    physics: Mapping[str, Any],
    B_matrix: np.ndarray,
    *,
    fermi_shift: float = 0.0,
    experiment_fn=None,
    diag_device: Optional[str] = None,
    kz_shift: float = 0.0,
):
    apply_chinook_runtime_patches()
    if experiment_fn is None:
        from chinook.ARPES_lib import experiment as experiment_fn

    K_BULK, A_bulk, kb, num_x, num_y, num_e, energy_axis = build_k_bulk_mesh(
        k_bounds,
        hv=float(physics["hv"]),
        work_function=float(physics["work_function"]),
        inner_potential=float(physics["inner_potential"]),
        slit_angle=float(physics["slit_angle"]),
        manip_theta=float(physics["manip_theta"]),
        manip_azimuth=float(physics["manip_azimuth"]),
        manip_tilt=float(physics["manip_tilt"]),
        incidence_angle=float(physics["incidence_angle"]),
        polarization=str(physics["polarization"]),
        hkl=tuple(physics["hkl"]),
        B_matrix=B_matrix,
        lin_pol_angle=float(physics.get("lin_pol_angle", 45.0)),
        fresnel_enabled=bool(physics.get("fresnel_enabled", True)),
        optical_n=float(physics.get("optical_n", 1.0)),
        optical_k=float(physics.get("optical_k", 0.0)),
        include_photon_momentum=bool(
            physics.get("include_photon_momentum", False)
        ),
    )
    if abs(float(kz_shift)) > 0.0:
        # Escape-depth Δk_z along surface normal (same Z_surf as build_k_bulk_mesh),
        # not bulk cartesian index 2 — required for tilted hkl e.g. (-2,0,1).
        K_BULK = shift_k_bulk_along_surface_normal(
            K_BULK, float(kz_shift), tuple(physics["hkl"]), B_matrix
        )

    me_mode = str(physics.get("matrix_element_mode", "Full Matrix Elements"))
    is_bare = "Off" in me_mode
    is_full = "Full" in me_mode
    T = float(physics.get("temperature", 10.0))
    se_width = float(physics.get("se_width", 0.01))

    arpes_dict = build_arpes_dict_from_physics(
        physics,
        cube=kb,
        energy_axis=energy_axis,
        A_bulk=A_bulk,
        is_full=is_full,
    )

    exp = experiment_fn(tb_model, arpes_dict)
    exp.ME = is_full

    if hasattr(exp.TB, "H"):
        del exp.TB.H
    if hasattr(exp.TB, "Eband"):
        del exp.TB.Eband
    if hasattr(exp.TB, "evec"):
        del exp.TB.evec

    exp.basis = exp.rot_basis()

    class CustomMesh:
        def __init__(self, k):
            self.kpts = k

    exp.TB.Kobj = CustomMesh(K_BULK.T)
    exp.k = K_BULK.T

    nk = int(K_BULK.shape[1])
    if diag_device and str(diag_device).lower() in ("cuda", "cpu", "mps"):
        print(
            f"GrizzlyME diagonalize: Nk={nk} device={diag_device}",
            flush=True,
        )
        exp.val, exp.vec = _grizzly_diagonalize_tb(exp.TB, str(diag_device), nk)
    else:
        exp.val, exp.vec = exp.TB.solve_H()
    if abs(fermi_shift) > 1e-12:
        exp.val = exp.val - fermi_shift

    # Uncoupled Wannier orbitals sit at exactly E=0 after EF alignment and flood
    # the Fermi edge with a flat bright line. Keep them out of Chinook dig_range.
    val = np.asarray(exp.val, dtype=float)
    zero_mask = np.abs(val) < 1e-10
    n_zero = int(np.count_nonzero(zero_mask))
    if n_zero:
        val = val.copy()
        val[zero_mask] = 1e6
        exp.val = val
        print(
            f"NOTE: excluded {n_zero} exact-zero eigenvalues from ARPES window "
            "(uncoupled orbitals after EF shift).",
            flush=True,
        )

    return {
        "exp": exp,
        "K_BULK": K_BULK,
        "A_bulk": A_bulk,
        "num_x": num_x,
        "num_y": num_y,
        "num_e": num_e,
        "energy_axis": energy_axis,
        "is_bare": is_bare,
        "is_full": is_full,
        "se_width": se_width,
        "T": T,
    }


def _kbulk_emission_angles(K_BULK: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-k emission (th, ph) from custom bulk final-state momenta."""
    k_bulk_para = np.sqrt(K_BULK[0] ** 2 + K_BULK[1] ** 2)
    k_bulk_norm = np.sqrt(K_BULK[0] ** 2 + K_BULK[1] ** 2 + K_BULK[2] ** 2)
    k_bulk_norm = np.where(k_bulk_norm == 0, 1e-10, k_bulk_norm)
    th_k = np.arcsin(np.clip(k_bulk_para / k_bulk_norm, -1.0, 1.0))
    ph_k = np.arctan2(K_BULK[1], K_BULK[0])
    return th_k, ph_k


def _finalize_me_geometry(ctx):
    exp = ctx["exp"]
    K_BULK = ctx["K_BULK"]
    num_x, num_y = ctx["num_x"], ctx["num_y"]

    exp.Eb = exp.val.flatten()
    exp.Ev = exp.vec
    # Placeholder grids: kinematics live in K_BULK. Chinook datacube() will
    # recompute th from X,Y — those values are restored after datacube.
    exp.X = np.zeros((num_y, num_x))
    exp.Y = np.zeros((num_y, num_x))

    th_k, ph_k = _kbulk_emission_angles(K_BULK)
    exp.ph = ph_k
    exp.th = th_k
    exp.diagonalize = lambda *args, **kwargs: None


def _restore_emission_angles_after_datacube(ctx) -> None:
    """Chinook datacube() overwrites th using X=Y=0 → all th=0. Restore from K_BULK.

    Chinook M_compute uses th[peak] and ph[k_index]; ph stays length Nk.
    """
    exp = ctx["exp"]
    th_k, ph_k = _kbulk_emission_angles(ctx["K_BULK"])
    exp.ph = ph_k
    nstates = len(exp.basis)
    k_idx = (np.asarray(exp.pks[:, 0], dtype=np.int64) // nstates)
    exp.th = th_k[k_idx]


def _datacube_without_mk(ctx) -> None:
    """Run Chinook datacube peak/radint setup; skip serial matrix elements."""
    exp = ctx["exp"]
    orig_serial = exp.serial_Mk
    orig_thread = exp.thread_Mk

    def _noop_mk(*_args, **_kwargs):
        return None

    exp.serial_Mk = _noop_mk
    exp.thread_Mk = _noop_mk
    try:
        _datacube_call_with_radint_cache(exp)
    finally:
        exp.serial_Mk = orig_serial
        exp.thread_Mk = orig_thread
    _restore_emission_angles_after_datacube(ctx)


def _datacube_call_with_radint_cache(exp) -> None:
    """``datacube()`` with GrizzlyME radint cache (memory + optional disk)."""
    import chinook.radint_lib as radint_lib

    from grizzly.radint_cache import DEFAULT_RADINT_CACHE, make_radint_cache_key

    cache_key = make_radint_cache_key(exp)
    cached = DEFAULT_RADINT_CACHE.get(cache_key)
    orig_make = radint_lib.make_radint_pointer
    if cached is not None:
        bfuncs_c, pointers_c = cached

        def _cached_make(_rad_dict, _basis, _Eb):
            return bfuncs_c, pointers_c

        radint_lib.make_radint_pointer = _cached_make
    try:
        exp.datacube()
    finally:
        radint_lib.make_radint_pointer = orig_make
    if cached is None and hasattr(exp, "Bfuncs") and hasattr(exp, "radint_pointers"):
        DEFAULT_RADINT_CACHE.put(cache_key, exp.Bfuncs, exp.radint_pointers)


@dataclass
class GrizzlyMeShell:
    """Basis-level Chinook ME tables reused across θ-blocks (Tier 1 hoist)."""

    basis: Any
    prefactors: np.ndarray
    Largs: Any
    Margs: Any
    Gbasis: Any
    orbital_pointers: Any
    proj_arr: Any
    Bfuncs: Any
    radint_pointers: Any
    dig_range: Tuple[float, float]
    nstates: int
    spin: bool
    hv: float
    W: float
    mfp: float


def build_grizzly_me_shell(
    tb_model,
    physics: Mapping[str, Any],
    B_matrix: np.ndarray,
    e_axis: np.ndarray,
    *,
    A_bulk: Optional[np.ndarray] = None,
) -> GrizzlyMeShell:
    """
    One-time Full-ME setup: radial integrals + Ylm/Gaunt tables for fixed TB/physics.

    Safe to reuse across θ-chunks when φ, E, hv, and matrix-element mode are unchanged.
    """
    apply_chinook_runtime_patches()
    from chinook.ARPES_lib import all_Y, experiment as experiment_fn
    from grizzly import prepare_me_shell
    from grizzly.me_shell import resolve_radint_mode
    from grizzly.radint_cache import dig_range_from_cube

    e_axis = np.asarray(e_axis, dtype=float)
    ne = max(1, len(e_axis))
    if A_bulk is None:
        _, A_bulk, _, _, _, _, _ = build_k_bulk_mesh(
            {"X": [0.0, 0.0, 1], "Y": [0.0, 0.0, 1], "E": [float(e_axis[0]), float(e_axis[-1]), ne]},
            hv=float(physics["hv"]),
            work_function=float(physics["work_function"]),
            inner_potential=float(physics["inner_potential"]),
            slit_angle=float(physics.get("slit_angle", 0.0)),
            manip_theta=float(physics.get("manip_theta", 0.0)),
            manip_azimuth=float(physics.get("manip_azimuth", 0.0)),
            manip_tilt=float(physics.get("manip_tilt", 0.0)),
            incidence_angle=float(physics.get("incidence_angle", 55.0)),
            polarization=str(physics.get("polarization", "Linear Horizontal (p-pol)")),
            hkl=tuple(physics.get("hkl", (0, 0, 1))),
            B_matrix=B_matrix,
            lin_pol_angle=float(physics.get("lin_pol_angle", 45.0)),
            fresnel_enabled=bool(physics.get("fresnel_enabled", True)),
            optical_n=float(physics.get("optical_n", 1.0)),
            optical_k=float(physics.get("optical_k", 0.0)),
            include_photon_momentum=bool(
                physics.get("include_photon_momentum", False)
            ),
        )

    me_mode = str(physics.get("matrix_element_mode", "Full Matrix Elements"))
    is_full = "Full" in me_mode
    stub_cube = {
        "X": [0.0, 0.0, 1],
        "Y": [0.0, 0.0, 1],
        "E": [float(e_axis[0]), float(e_axis[-1]), ne],
    }
    arpes_dict = build_arpes_dict_from_physics(
        physics,
        cube=stub_cube,
        energy_axis=e_axis,
        A_bulk=A_bulk,
        is_full=is_full,
    )

    exp = experiment_fn(tb_model, arpes_dict)
    exp.ME = is_full
    exp.diagonalize = lambda *args, **kwargs: None
    radint_started = time.perf_counter()
    prepared = prepare_me_shell(exp)
    radint_wall = time.perf_counter() - radint_started
    radint_status = "HIT" if prepared.radint_source == "cache" else "MISS"
    print(
        f"Radint {radint_status} backend={resolve_radint_mode()} "
        f"source={prepared.radint_source} wall={radint_wall:.2f}s",
        flush=True,
    )
    dig_range = dig_range_from_cube(exp.cube)

    # PreparedMeShell exposes indexed Gbasis but not the corresponding pointers.
    _, _, _, orbital_pointers = all_Y(exp.basis)
    return GrizzlyMeShell(
        basis=prepared.basis,
        prefactors=prepared.prefactors,
        Largs=prepared.Largs,
        Margs=prepared.Margs,
        Gbasis=prepared.Gbasis,
        orbital_pointers=orbital_pointers,
        proj_arr=prepared.proj_arr,
        Bfuncs=prepared.Bfuncs,
        radint_pointers=prepared.radint_pointers,
        dig_range=dig_range,
        nstates=prepared.nstates,
        spin=prepared.spin,
        hv=float(exp.hv),
        W=float(exp.W),
        mfp=prepared.mfp,
    )


def _apply_me_shell_peaks(ctx, shell: GrizzlyMeShell) -> None:
    """Attach hoisted ME tables and build block-specific peak list.

    Empty peak lists are allowed (θ-block with no bands in the energy window);
    callers must treat ``exp.pks`` length 0 as zero intensity.
    """
    exp = ctx["exp"]
    num_x, num_y = ctx["num_x"], ctx["num_y"]
    Eb = np.asarray(exp.val, dtype=float).reshape(-1)
    lo, hi = shell.dig_range
    nstates = shell.nstates
    rows = []
    for i in range(len(Eb)):
        e = Eb[i]
        if lo <= e <= hi:
            k_idx = i // nstates
            rows.append([i, k_idx // num_x, k_idx % num_x, e])
    if not rows:
        print(
            f"WARN: no states in energy window dig_range=({lo:.4f},{hi:.4f}) eV; "
            f"Eband=[{float(np.nanmin(Eb)):.3f},{float(np.nanmax(Eb)):.3f}] eV "
            f"(Nk×Nb={Eb.size}). Returning empty peaks for this θ-block.",
            flush=True,
        )
        exp.pks = np.zeros((0, 4), dtype=float)
        exp.Mk = np.zeros((0, 2, 3), dtype=complex)
        exp.basis = shell.basis
        exp.prefactors = shell.prefactors
        exp.Largs = shell.Largs
        exp.Margs = shell.Margs
        exp.Gbasis = shell.Gbasis
        exp.orbital_pointers = shell.orbital_pointers
        exp.proj_arr = shell.proj_arr
        exp.Bfuncs = shell.Bfuncs
        exp.radint_pointers = shell.radint_pointers
        exp.spin = shell.spin
        exp.hv = shell.hv
        exp.W = shell.W
        exp.mfp = shell.mfp
        th_k, ph_k = _kbulk_emission_angles(ctx["K_BULK"])
        exp.ph = ph_k
        exp.th = th_k
        return

    exp.pks = np.asarray(rows, dtype=float)
    exp.Mk = np.zeros((len(exp.pks), 2, 3), dtype=complex)
    exp.basis = shell.basis
    exp.prefactors = shell.prefactors
    exp.Largs = shell.Largs
    exp.Margs = shell.Margs
    exp.Gbasis = shell.Gbasis
    exp.orbital_pointers = shell.orbital_pointers
    exp.proj_arr = shell.proj_arr
    exp.Bfuncs = shell.Bfuncs
    exp.radint_pointers = shell.radint_pointers
    exp.spin = shell.spin
    exp.hv = shell.hv
    exp.W = shell.W
    exp.mfp = shell.mfp
    th_k, ph_k = _kbulk_emission_angles(ctx["K_BULK"])
    exp.ph = ph_k
    k_idx = (np.asarray(exp.pks[:, 0], dtype=np.int64) // nstates)
    exp.th = th_k[k_idx]


def _chinook_serial_mk(exp) -> None:
    valid = np.array(
        [i for i in range(len(exp.pks)) if exp.th[i] >= 0], dtype=int
    )
    if len(valid) == 0:
        return
    exp.serial_Mk(valid)


def _intensity_with_kz_broaden(physics: Mapping[str, Any], run_once):
    """Weight-sum intensities over Lorentzian Δk_z along the surface normal.

    Each pass shifts ``K_BULK`` by ``Δk_z * Z_surf`` and recomputes emission
    angles from the shifted mesh (so the sum also smears final-state direction).
    Default ``kz_halfwidth=0`` → single unshifted run (legacy). If ``halfwidth>0``
    and ``kz_npoints<=1``, ``lorentzian_kz_weights`` upgrades to an odd 7-point
    grid. **Not** coherent LEED final-state interference.
    """
    hw = float(physics.get("kz_halfwidth", 0.0))
    npoints = int(physics.get("kz_npoints", 7))
    dk, w = lorentzian_kz_weights(hw, npoints)
    acc = None
    for dki, wi in zip(dk, w):
        intensity = run_once(kz_shift=float(dki))
        acc = intensity * wi if acc is None else acc + intensity * wi
    return acc


def run_chinook_arpes(
    tb_model,
    k_bounds: Mapping[str, list],
    physics: Mapping[str, Any],
    B_matrix: np.ndarray,
    *,
    fermi_shift: float = 0.0,
    experiment_fn=None,
) -> np.ndarray:
    """
    Run Chinook ARPES for one detector cube (num_x × num_y × num_e).

    When ``physics["kz_halfwidth"] > 0``, returns a Lorentzian-weighted sum of
    intensities with ``K_BULK`` shifted by each Δk_z along the surface normal
    ``Z_surf``. Emission angles are recomputed from each shifted mesh (detector
    smear included) — **not** coherent LEED final-state interference.
    Default ``kz_halfwidth=0`` preserves legacy single-cut behavior.

    Returns intensity array shaped (num_x, num_y, num_e).
    """

    def _once(*, kz_shift: float = 0.0) -> np.ndarray:
        ctx = _setup_kmesh_experiment(
            tb_model,
            k_bounds,
            physics,
            B_matrix,
            fermi_shift=fermi_shift,
            experiment_fn=experiment_fn,
            kz_shift=kz_shift,
        )
        if ctx["is_bare"]:
            return _bare_intensity(ctx)

        _finalize_me_geometry(ctx)
        # Must not let datacube compute Mk with th overwritten from X=Y=0.
        _datacube_without_mk(ctx)
        _chinook_serial_mk(ctx["exp"])
        _spectral = ctx["exp"].spectral()
        # Chinook returns (I_raw, Ig); use broadened Ig (matches legacy chinook_wrapper).
        output_maps = np.real(_spectral[1] if isinstance(_spectral, tuple) else _spectral)
        intensity_3d = _reshape_me_intensity(
            output_maps, ctx["num_x"], ctx["num_y"], ctx["num_e"]
        )
        if not ctx["is_full"]:
            intensity_3d = _apply_dipole(
                intensity_3d, ctx["K_BULK"], ctx["A_bulk"], ctx["num_x"], ctx["num_y"]
            )
        return intensity_3d

    return _intensity_with_kz_broaden(physics, _once)


def _spinor_masks(exp, spinor_basis: str):
    """(up, down) boolean masks over the TB basis, or None for a spinless model.

    Wannier90 spinor hr.dat carries spin inside H, with every orbital tag spin=+1
    (GrizzlyME is spinless). Its single matrix-element row then adds the up and
    down spinor components COHERENTLY, but the photoelectron's up and down
    channels are orthogonal and must add in intensity. ``spinor_basis`` says how
    the components are ordered: 'interleaved' (orb0 up, orb0 down, orb1 up, ...;
    Wannier90 spinors=true) or 'blocked' (all up, then all down).
    """
    n = len(exp.basis)
    labels = [getattr(o, "label", None) for o in exp.basis]
    if spinor_basis in ("", "none", None):
        if n % 2 == 0 and n > 0 and labels[0::2] == labels[1::2]:
            print(
                "WARNING: basis labels come in identical pairs (looks like a Wannier "
                "spinor model) but physics['spinor_basis'] is 'none' — up/down will "
                "add coherently. Set spinor_basis='interleaved' if this is a spinor hr.dat.",
                flush=True,
            )
        return None
    if n % 2:
        raise ValueError(f"spinor_basis={spinor_basis!r} needs an even basis, got {n}")
    idx = np.arange(n)
    if spinor_basis == "interleaved":
        if labels[0::2] != labels[1::2]:
            raise ValueError("spinor_basis='interleaved' but labels are not paired (2i, 2i+1)")
        up = idx % 2 == 0
    elif spinor_basis == "blocked":
        if labels[: n // 2] != labels[n // 2 :]:
            raise ValueError("spinor_basis='blocked' but the two halves have different labels")
        up = idx < n // 2
    else:
        raise ValueError(f"unknown spinor_basis {spinor_basis!r}")
    return up, ~up


def run_grizzly_arpes(
    tb_model,
    k_bounds: Mapping[str, list],
    physics: Mapping[str, Any],
    B_matrix: np.ndarray,
    *,
    fermi_shift: float = 0.0,
    device: str = "cpu",
    experiment_fn=None,
    use_grizzly_spectral: Optional[bool] = None,
    profile_stages: bool = False,
    me_shell: Optional[GrizzlyMeShell] = None,
) -> np.ndarray:
    """
    Same kmesh physics as run_chinook_arpes; Mk via GrizzlyME.

    When ``device`` is cuda, defaults to Grizzly GPU diagonalization + spectral
    (hybrid fast path). Set ``use_grizzly_spectral=False`` to keep Chinook CPU
    ``exp.spectral()`` for A/B timing.

    When ``physics["kz_halfwidth"] > 0``, returns a Lorentzian-weighted sum of
    intensities with ``K_BULK`` shifted by each Δk_z along the surface normal
    ``Z_surf``. Emission angles are recomputed per pass — **not** coherent LEED
    final-state interference. Default off preserves legacy.
    """
    dev = str(device).lower()
    if use_grizzly_spectral is None:
        use_grizzly_spectral = dev in ("cuda", "mps")
    diag_dev = device if dev in ("cuda", "mps", "cpu") and dev != "auto" else None
    if dev == "cuda":
        diag_dev = "cuda"

    def _once(*, kz_shift: float = 0.0) -> np.ndarray:
        t0 = time.perf_counter()
        ctx = _setup_kmesh_experiment(
            tb_model,
            k_bounds,
            physics,
            B_matrix,
            fermi_shift=fermi_shift,
            experiment_fn=experiment_fn,
            diag_device=diag_dev if dev in ("cuda", "mps") else None,
            kz_shift=kz_shift,
        )
        t_setup = time.perf_counter()
        if ctx["is_bare"]:
            return _bare_intensity(ctx)

        from grizzly.engine import compute_all_Mk
        from grizzly.future import require_spinless

        exp = ctx["exp"]
        require_spinless(exp, feature="run_grizzly_arpes")
        _finalize_me_geometry(ctx)
        if me_shell is not None:
            t_dc0 = time.perf_counter()
            _apply_me_shell_peaks(ctx, me_shell)
            t_datacube = time.perf_counter()
            if profile_stages and t_datacube - t_dc0 > 0.05:
                print(
                    f"  ME shell peaks wall={t_datacube - t_dc0:.2f}s (hoisted radint/Ylm)",
                    flush=True,
                )
        else:
            _datacube_without_mk(ctx)
            t_datacube = time.perf_counter()

        # θ-block with no bands in the ARPES energy window → silent zeros
        if getattr(exp, "pks", None) is None or len(exp.pks) == 0:
            if profile_stages:
                print(
                    f"  grizzly stages: setup={t_setup - t0:.2f}s "
                    f"datacube={t_datacube - t_setup:.2f}s "
                    f"mk=0.00s spectral=0.00s (empty window) "
                    f"total={time.perf_counter() - t0:.2f}s",
                    flush=True,
                )
            return np.zeros(
                (ctx["num_x"], ctx["num_y"], ctx["num_e"]), dtype=float
            )

        masks = _spinor_masks(exp, str(physics.get("spinor_basis", "none")))
        if masks is None:
            exp.Mk = compute_all_Mk(exp, device=str(device))
        else:
            # Row 0 = up channel, row 1 = down channel; spectral() sums |pol.M|^2
            # over rows, i.e. the two spin channels add incoherently.
            ev_full = exp.Ev
            rows = []
            for m in masks:
                exp.Ev = ev_full * m[None, :, None]
                rows.append(np.asarray(compute_all_Mk(exp, device=str(device)))[:, 0, :])
            exp.Ev = ev_full
            exp.Mk = np.stack(rows, axis=1)
        t_mk = time.perf_counter()

        if use_grizzly_spectral:
            from grizzly.spectral import spectral_maps_from_experiment

            _, output_maps = spectral_maps_from_experiment(exp, device=str(device))
            output_maps = np.real(output_maps)
        else:
            _spectral = exp.spectral()
            output_maps = np.real(
                _spectral[1] if isinstance(_spectral, tuple) else _spectral
            )
        t_spec = time.perf_counter()

        if profile_stages:
            print(
                f"  grizzly stages: setup={t_setup - t0:.2f}s "
                f"datacube={t_datacube - t_setup:.2f}s "
                f"mk={t_mk - t_datacube:.2f}s "
                f"spectral={t_spec - t_mk:.2f}s "
                f"total={t_spec - t0:.2f}s",
                flush=True,
            )

        intensity_3d = _reshape_me_intensity(
            output_maps, ctx["num_x"], ctx["num_y"], ctx["num_e"]
        )
        if not ctx["is_full"]:
            intensity_3d = _apply_dipole(
                intensity_3d, ctx["K_BULK"], ctx["A_bulk"], ctx["num_x"], ctx["num_y"]
            )
        return intensity_3d

    return _intensity_with_kz_broaden(physics, _once)

