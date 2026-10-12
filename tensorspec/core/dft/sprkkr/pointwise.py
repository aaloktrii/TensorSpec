"""Lab detector angles (slit, deflector) -> SPR-KKR (Theta, Phi) single-point jobs.

Phase 1a geometry:
- Slit at azimuth 0 is assumed parallel to SPR-KKR's in-plane x axis.
- phi_offset_deg is the UNCALIBRATED constant offset; it is unknown and fixed
  per (material, surface, IQ_AT_SURF) until Task 8 calibration runs.
- Sample azimuth (manip_azimuth) is a physical manipulator rotation that adds
  to phi_offset_deg; all three manip angles pass through sample_to_bulk_frame.

Energy reference assumption (residual error):
- dTheta/dE ~ tan(Theta) / (2 E_kin); at Theta=18 deg, E_kin=79.5 eV ~ 0.13 deg/eV.
- |dk_par|/k_par ~ dE / (2 E_kin); ~ 0.7 %/eV.
- Angles are evaluated ONCE at ref_energy_eV and held fixed: over a -1.0..0.1 eV
  window the worst-case drift is <= 0.15 deg in Theta and <= 0.8 % in k_par.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


K_PER_SQRT_EV: float = 0.512316  # same constant as chinook_arpes_kmesh.build_k_bulk_mesh


@dataclass(frozen=True)
class AnglePoint:
    index: int
    slit_deg: float  # lab slit angle (the swept detector axis)
    theta_e_deg: float  # SPR-KKR SPEC_EL THETA, >= 0
    phi_e_deg: float  # SPR-KKR SPEC_EL PHI, wrapped to (-180, 180]
    k_par: float  # |k_par| at ref energy, 1/A, >= 0
    k_slit: float  # signed lab slit-axis k, 1/A
    k_defl: float  # signed lab deflector-axis k, 1/A


def k_vacuum(hv_eV: float, work_function_eV: float, energy_eV: float = 0.0) -> float:
    """Vacuum k-magnitude from photon energy and work function.

    Args:
        hv_eV: photon energy in eV
        work_function_eV: work function in eV
        energy_eV: binding energy relative to E_F in eV (default 0.0 = E_F).
                   Same sign rule as ArpesParams.e_min_eV: negative below Fermi.

    Returns:
        k magnitude in 1/Angstrom.
    """
    E_kin = max(hv_eV - work_function_eV + energy_eV, 0.1)
    return K_PER_SQRT_EV * np.sqrt(E_kin)


def wrap_deg(angle_deg: float) -> float:
    """Wrap angle to (-180, 180]."""
    while angle_deg > 180.0:
        angle_deg -= 360.0
    while angle_deg <= -180.0:
        angle_deg += 360.0
    return angle_deg


def lab_k_vectors(
    theta_slit_deg,
    deflector_deg: float,
    slit_rot_deg: float,
    k: float,
) -> np.ndarray:
    """Lab-frame k vectors from slit angles, deflector, and slit orientation.

    Args:
        theta_slit_deg: array of lab slit angles in degrees (swept detector axis).
        deflector_deg: deflector angle in degrees (fixed per point).
        slit_rot_deg: slit orientation angle in degrees (rotation about surface normal).
        k: vacuum k-magnitude at reference energy, 1/A.

    Returns:
        (3, N) array: k_lab_x, k_lab_y, k_lab_z (outward is +y).
    """
    theta_slit_rad = np.radians(theta_slit_deg)
    deflector_rad = np.radians(deflector_deg)
    slit_rot_rad = np.radians(slit_rot_deg)

    # Chinook ARPES conventions: K_SLIT = k sin(theta), K_DEFL = k sin(deflector)
    K_SLIT = k * np.sin(theta_slit_rad)
    K_DEFL = k * np.sin(deflector_rad)

    # Lab-frame x, z from Chinook's formula (lines 280-281 of chinook_arpes_kmesh.py)
    k_lab_x = K_SLIT * np.cos(slit_rot_rad) - K_DEFL * np.sin(slit_rot_rad)
    k_lab_z = K_SLIT * np.sin(slit_rot_rad) + K_DEFL * np.cos(slit_rot_rad)

    # y-component from energy conservation (+ outward surface normal)
    k_sq = k**2
    k_lab_y = np.sqrt(np.clip(k_sq - k_lab_x**2 - k_lab_z**2, 0.0, None))

    return np.vstack([k_lab_x, k_lab_y, k_lab_z])


def angle_points(
    theta_range_deg: Tuple[float, float],
    nt: int,
    *,
    hv_eV: float,
    work_function_eV: float,
    deflector_deg: float = 0.0,
    slit_rot_deg: float = 0.0,
    manip_theta_deg: float = 0.0,
    manip_azimuth_deg: float = 0.0,
    manip_tilt_deg: float = 0.0,
    ref_energy_eV: float = 0.0,
    phi_offset_deg: float = 0.0,
) -> List[AnglePoint]:
    """Generate one AnglePoint per lab slit angle; rotate all to sample frame.

    Args:
        theta_range_deg: (min, max) lab slit angles in degrees.
        nt: number of points (if 1, use the midpoint).
        hv_eV: photon energy in eV.
        work_function_eV: work function in eV.
        deflector_deg: deflector angle in degrees.
        slit_rot_deg: slit orientation in degrees.
        manip_theta_deg: manipulator theta in degrees.
        manip_azimuth_deg: manipulator azimuth (sample rotation) in degrees.
        manip_tilt_deg: manipulator tilt in degrees.
        ref_energy_eV: reference energy (rel. E_F) for angle evaluation.
        phi_offset_deg: unknown SPR-KKR x-axis offset (UNCALIBRATED).

    Returns:
        List of AnglePoint, one per slit angle.

    Raises:
        ValueError: if any theta_e_deg > 90 deg (emission into crystal).
    """
    if nt == 1:
        theta_slit_deg = np.array([np.mean(theta_range_deg)])
    else:
        theta_slit_deg = np.linspace(theta_range_deg[0], theta_range_deg[1], nt)

    k = k_vacuum(hv_eV, work_function_eV, ref_energy_eV)
    K_lab = lab_k_vectors(theta_slit_deg, deflector_deg, slit_rot_deg, k)

    # Lazy import to avoid package cycle and keep module load light
    from tensorspec.core.arpes.one_step.chinook_arpes_kmesh import sample_to_bulk_frame

    K_sample = sample_to_bulk_frame(K_lab, manip_theta_deg, manip_azimuth_deg, manip_tilt_deg)

    points = []
    for i, theta_i in enumerate(theta_slit_deg):
        k_x = K_sample[0, i]
        k_y = K_sample[1, i]
        k_z = K_sample[2, i]

        k_par = np.hypot(k_x, k_y)
        # sample_to_bulk_frame returns R_inv @ K_lab where R_inv is orthogonal with det=1.
        # For outward K_lab[1] (y=+outward), K_sample[2] is positive. Trivial case (slit=0,
        # deflector=0, manip=0) gives K_sample=[0,0,k] so theta_e=atan2(0,k)=0 deg. Correct.
        theta_e = np.degrees(np.arctan2(k_par, k_z))
        phi_e = np.degrees(np.arctan2(k_y, k_x)) + phi_offset_deg
        phi_e = wrap_deg(phi_e)

        k_slit = k * np.sin(np.radians(theta_i))
        k_defl = k * np.sin(np.radians(deflector_deg))

        if theta_e > 90.0:
            raise ValueError(
                f"AnglePoint {i}: theta_e_deg={theta_e:.2f} > 90, emission into crystal"
            )

        points.append(
            AnglePoint(
                index=i,
                slit_deg=float(theta_i),
                theta_e_deg=theta_e,
                phi_e_deg=phi_e,
                k_par=k_par,
                k_slit=k_slit,
                k_defl=k_defl,
            )
        )

    return points


def sprkkr_angles_to_lab_k(
    theta_e_deg,
    phi_e_deg,
    *,
    k: float,
    slit_rot_deg: float = 0.0,
    manip_theta_deg: float = 0.0,
    manip_azimuth_deg: float = 0.0,
    manip_tilt_deg: float = 0.0,
    phi_offset_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Round-trip: SPR-KKR (Theta, Phi) angles -> lab k components.

    Inverse of angle_points: given SPR-KKR angles, return lab slit and deflector k.
    Used for validation tests.

    Args:
        theta_e_deg: SPR-KKR SPEC_EL THETA in degrees.
        phi_e_deg: SPR-KKR SPEC_EL PHI in degrees.
        k: vacuum k-magnitude, 1/A.
        slit_rot_deg: slit orientation in degrees.
        manip_theta_deg: manipulator theta in degrees.
        manip_azimuth_deg: manipulator azimuth in degrees.
        manip_tilt_deg: manipulator tilt in degrees.
        phi_offset_deg: SPR-KKR x-axis offset in degrees.

    Returns:
        (k_slit, k_defl): arrays of same shape as theta_e_deg, k_slit and k_defl in 1/A.
    """
    theta_e_rad = np.radians(theta_e_deg)
    phi_e_rad = np.radians(phi_e_deg - phi_offset_deg)  # Remove offset to get sample frame phi

    from tensorspec.core.arpes.one_step.chinook_arpes_kmesh import sample_to_bulk_frame

    # Reconstruct sample-frame k from spherical angles
    k_par = k * np.sin(theta_e_rad)
    k_z = k * np.cos(theta_e_rad)
    k_x_sample = k_par * np.cos(phi_e_rad)
    k_y_sample = k_par * np.sin(phi_e_rad)
    K_sample = np.vstack([k_x_sample, k_y_sample, k_z])

    # Inverse transformation: sample -> lab
    # sample_to_bulk_frame computes R_inv @ K_lab = K_sample, so K_lab = R @ K_sample
    # We need R = inverse of R_inv
    t_rad = np.radians(manip_theta_deg)
    a_rad = np.radians(manip_azimuth_deg)
    tilt_rad = np.radians(manip_tilt_deg)

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
    R_base = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    R_total = R_z @ R_y @ R_x @ R_base
    R = R_total  # Forward transformation (sample_to_bulk uses R_inv)

    K_lab = R @ K_sample

    # Now extract k_slit and k_defl from lab frame
    slit_rot_rad = np.radians(slit_rot_deg)
    # Inverse of the Chinook formulas:
    # k_lab_x = k_slit * cos(slit_rot) - k_defl * sin(slit_rot)
    # k_lab_z = k_slit * sin(slit_rot) + k_defl * cos(slit_rot)
    # Solve for k_slit and k_defl:
    cos_s = np.cos(slit_rot_rad)
    sin_s = np.sin(slit_rot_rad)
    k_lab_x = K_lab[0]
    k_lab_z = K_lab[2]
    k_slit = k_lab_x * cos_s + k_lab_z * sin_s
    k_defl = -k_lab_x * sin_s + k_lab_z * cos_s

    return k_slit, k_defl
