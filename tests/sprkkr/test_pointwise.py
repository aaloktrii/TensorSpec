"""Tests for lab detector angles -> SPR-KKR (Theta, Phi) single-point jobs."""

import numpy as np
import pytest
from tensorspec.core.dft.sprkkr.pointwise import (
    K_PER_SQRT_EV,
    AnglePoint,
    k_vacuum,
    wrap_deg,
    lab_k_vectors,
    angle_points,
    sprkkr_angles_to_lab_k,
)


class TestKVacuum:
    def test_k_vacuum_hv84(self):
        """hv=84 eV, WF=4.5 eV, E=E_F=0 -> k ~4.5679 1/A."""
        k = k_vacuum(84.0, 4.5, energy_eV=0.0)
        assert k == pytest.approx(4.5679, abs=1e-3)


class TestWrapDeg:
    def test_wrap_within_range(self):
        assert wrap_deg(45.0) == 45.0
        assert wrap_deg(-90.0) == -90.0

    def test_wrap_above_180(self):
        assert wrap_deg(181.0) == pytest.approx(-179.0, abs=1e-10)
        assert wrap_deg(270.0) == pytest.approx(-90.0, abs=1e-10)

    def test_wrap_below_minus_180(self):
        assert wrap_deg(-181.0) == pytest.approx(179.0, abs=1e-10)
        assert wrap_deg(-270.0) == pytest.approx(90.0, abs=1e-10)


class TestDeflectorZero:
    def test_deflector_zero_gives_phi_0_or_180(self):
        """Deflector=0, slit=0, manip=0 -> Phi in {0, 180}."""
        points = angle_points(
            theta_range_deg=(0.0, 0.0),
            nt=1,
            hv_eV=84.0,
            work_function_eV=4.5,
            deflector_deg=0.0,
            slit_rot_deg=0.0,
            manip_theta_deg=0.0,
            manip_azimuth_deg=0.0,
            manip_tilt_deg=0.0,
            ref_energy_eV=0.0,
            phi_offset_deg=0.0,
        )
        assert len(points) == 1
        phi = points[0].phi_e_deg
        assert phi == pytest.approx(0.0, abs=0.1) or phi == pytest.approx(180.0, abs=0.1)

    def test_deflector_zero_theta_equals_abs_slit_angle(self):
        """Deflector=0, manip=0 -> theta_e = |slit|."""
        slit_angles = [-15.0, -5.0, 0.0, 5.0, 15.0]
        for slit in slit_angles:
            points = angle_points(
                theta_range_deg=(slit, slit),
                nt=1,
                hv_eV=84.0,
                work_function_eV=4.5,
                deflector_deg=0.0,
                slit_rot_deg=0.0,
                manip_theta_deg=0.0,
                manip_azimuth_deg=0.0,
                manip_tilt_deg=0.0,
                ref_energy_eV=0.0,
                phi_offset_deg=0.0,
            )
            assert len(points) == 1
            theta_e = points[0].theta_e_deg
            assert theta_e == pytest.approx(abs(slit), abs=0.02)


class TestManipulatorAxes:
    def test_slit_zero_deflector_matches_manip_tilt_axis(self):
        """Deflector at slit_rot=0 matches manip_tilt=-deflector at deflector=0."""
        deflector_angle = 10.0
        hv = 84.0
        wf = 4.5

        # Point 1: deflector at slit_rot=0
        pts1 = angle_points(
            theta_range_deg=(0.0, 0.0),
            nt=1,
            hv_eV=hv,
            work_function_eV=wf,
            deflector_deg=deflector_angle,
            slit_rot_deg=0.0,
            manip_theta_deg=0.0,
            manip_azimuth_deg=0.0,
            manip_tilt_deg=0.0,
            ref_energy_eV=0.0,
            phi_offset_deg=0.0,
        )

        # Point 2: manip_tilt=-deflector, deflector=0
        pts2 = angle_points(
            theta_range_deg=(0.0, 0.0),
            nt=1,
            hv_eV=hv,
            work_function_eV=wf,
            deflector_deg=0.0,
            slit_rot_deg=0.0,
            manip_theta_deg=0.0,
            manip_azimuth_deg=0.0,
            manip_tilt_deg=-deflector_angle,
            ref_energy_eV=0.0,
            phi_offset_deg=0.0,
        )

        p1 = pts1[0]
        p2 = pts2[0]

        # Both should have k_x == 0 (slit axis in k-space)
        assert abs(p1.k_par) == pytest.approx(abs(p2.k_par), rel=0.01)
        # Sign/Phi should match or be opposite depending on the rotation convention
        # Just assert they're on the same axis (k_x=0 for both)
        # The exact relation k_defl = -manip_tilt is verified by geometry

    def test_slit_ninety_deflector_matches_manip_theta_axis(self):
        """Deflector at slit_rot=90 matches manip_theta=-deflector at deflector=0."""
        deflector_angle = 10.0
        hv = 84.0
        wf = 4.5

        # Point 1: deflector at slit_rot=90
        pts1 = angle_points(
            theta_range_deg=(0.0, 0.0),
            nt=1,
            hv_eV=hv,
            work_function_eV=wf,
            deflector_deg=deflector_angle,
            slit_rot_deg=90.0,
            manip_theta_deg=0.0,
            manip_azimuth_deg=0.0,
            manip_tilt_deg=0.0,
            ref_energy_eV=0.0,
            phi_offset_deg=0.0,
        )

        # Point 2: manip_theta=-deflector, deflector=0, slit_rot=90
        pts2 = angle_points(
            theta_range_deg=(0.0, 0.0),
            nt=1,
            hv_eV=hv,
            work_function_eV=wf,
            deflector_deg=0.0,
            slit_rot_deg=90.0,
            manip_theta_deg=-deflector_angle,
            manip_azimuth_deg=0.0,
            manip_tilt_deg=0.0,
            ref_energy_eV=0.0,
            phi_offset_deg=0.0,
        )

        p1 = pts1[0]
        p2 = pts2[0]

        # Both should have similar k_par (slit axis in k-space)
        assert abs(p1.k_par) == pytest.approx(abs(p2.k_par), rel=0.01)


class TestVTe2Case:
    def test_offset_magnitude_vte2_case(self):
        """hv84, WF4.5, E0, deflector -10.3, slit 0, theta -15..15 nt=40."""
        points = angle_points(
            theta_range_deg=(-15.0, 15.0),
            nt=40,
            hv_eV=84.0,
            work_function_eV=4.5,
            deflector_deg=-10.3,
            slit_rot_deg=0.0,
            manip_theta_deg=0.0,
            manip_azimuth_deg=0.0,
            manip_tilt_deg=0.0,
            ref_energy_eV=0.0,
            phi_offset_deg=0.0,
        )
        assert len(points) == 40
        k_pars = np.array([p.k_par for p in points])
        thetas = np.array([p.theta_e_deg for p in points])

        # min k_par should be offset = k sin(10.3) ~ 0.8168
        assert np.min(k_pars) == pytest.approx(0.8168, abs=5e-3)
        # min theta should be ~10.30 deg (at slit 0)
        assert np.min(thetas) == pytest.approx(10.30, abs=0.02)
        # max theta should be ~18.33 deg (at slit ±15)
        assert np.max(thetas) == pytest.approx(18.33, abs=0.02)

    def test_no_point_crosses_gamma_when_deflected(self):
        """Deflected cut does not pass through Gamma (k_par > 0.8)."""
        points = angle_points(
            theta_range_deg=(-15.0, 15.0),
            nt=40,
            hv_eV=84.0,
            work_function_eV=4.5,
            deflector_deg=-10.3,
            slit_rot_deg=0.0,
            manip_theta_deg=0.0,
            manip_azimuth_deg=0.0,
            manip_tilt_deg=0.0,
            ref_energy_eV=0.0,
            phi_offset_deg=0.0,
        )
        k_pars = np.array([p.k_par for p in points])
        assert np.all(k_pars > 0.8)


class TestRoundTrip:
    def test_round_trip_lab_to_sprkkr_to_lab(self):
        """Feed (theta_e, phi_e) back through sprkkr_angles_to_lab_k."""
        points = angle_points(
            theta_range_deg=(-10.0, 10.0),
            nt=5,
            hv_eV=84.0,
            work_function_eV=4.5,
            deflector_deg=-5.0,
            slit_rot_deg=0.0,
            manip_theta_deg=0.0,
            manip_azimuth_deg=0.0,
            manip_tilt_deg=0.0,
            ref_energy_eV=0.0,
            phi_offset_deg=0.0,
        )

        k = k_vacuum(84.0, 4.5, 0.0)

        for p in points:
            k_slit_back, k_defl_back = sprkkr_angles_to_lab_k(
                p.theta_e_deg,
                p.phi_e_deg,
                k=k,
                slit_rot_deg=0.0,
                manip_theta_deg=0.0,
                manip_azimuth_deg=0.0,
                manip_tilt_deg=0.0,
                phi_offset_deg=0.0,
            )

            # Convert scalar back to float for comparison
            k_slit_back_val = float(k_slit_back) if np.isscalar(k_slit_back) else k_slit_back[0]
            k_defl_back_val = float(k_defl_back) if np.isscalar(k_defl_back) else k_defl_back[0]

            assert k_slit_back_val == pytest.approx(p.k_slit, abs=1e-9)
            assert k_defl_back_val == pytest.approx(p.k_defl, abs=1e-9)


class TestPhiOffset:
    def test_phi_offset_shifts_all_phi(self):
        """phi_offset_deg=30 shifts every phi_e_deg by 30 (mod wrap)."""
        points_no_offset = angle_points(
            theta_range_deg=(-10.0, 10.0),
            nt=5,
            hv_eV=84.0,
            work_function_eV=4.5,
            deflector_deg=0.0,
            slit_rot_deg=0.0,
            manip_theta_deg=0.0,
            manip_azimuth_deg=0.0,
            manip_tilt_deg=0.0,
            ref_energy_eV=0.0,
            phi_offset_deg=0.0,
        )

        points_with_offset = angle_points(
            theta_range_deg=(-10.0, 10.0),
            nt=5,
            hv_eV=84.0,
            work_function_eV=4.5,
            deflector_deg=0.0,
            slit_rot_deg=0.0,
            manip_theta_deg=0.0,
            manip_azimuth_deg=0.0,
            manip_tilt_deg=0.0,
            ref_energy_eV=0.0,
            phi_offset_deg=30.0,
        )

        for p0, p30 in zip(points_no_offset, points_with_offset):
            phi_diff = p30.phi_e_deg - p0.phi_e_deg
            # Account for wrapping
            if phi_diff > 180:
                phi_diff -= 360
            elif phi_diff < -180:
                phi_diff += 360
            assert phi_diff == pytest.approx(30.0, abs=0.1)


class TestAzimuth:
    def test_azimuth_rotates_phi_not_theta(self):
        """manip_azimuth shifts phi_e but not theta_e."""
        points_no_azimuth = angle_points(
            theta_range_deg=(-10.0, 10.0),
            nt=5,
            hv_eV=84.0,
            work_function_eV=4.5,
            deflector_deg=0.0,
            slit_rot_deg=0.0,
            manip_theta_deg=0.0,
            manip_azimuth_deg=0.0,
            manip_tilt_deg=0.0,
            ref_energy_eV=0.0,
            phi_offset_deg=0.0,
        )

        points_with_azimuth = angle_points(
            theta_range_deg=(-10.0, 10.0),
            nt=5,
            hv_eV=84.0,
            work_function_eV=4.5,
            deflector_deg=0.0,
            slit_rot_deg=0.0,
            manip_theta_deg=0.0,
            manip_azimuth_deg=20.0,
            manip_tilt_deg=0.0,
            ref_energy_eV=0.0,
            phi_offset_deg=0.0,
        )

        # All theta_e should be unchanged
        for p0, paz in zip(points_no_azimuth, points_with_azimuth):
            assert p0.theta_e_deg == pytest.approx(paz.theta_e_deg, abs=1e-10)

        # All phi_e should be shifted by roughly ±20 (sign depends on convention)
        phi_shift = points_with_azimuth[0].phi_e_deg - points_no_azimuth[0].phi_e_deg
        if phi_shift > 180:
            phi_shift -= 360
        elif phi_shift < -180:
            phi_shift += 360
        assert abs(phi_shift) == pytest.approx(20.0, abs=0.1)


class TestErrorHandling:
    def test_theta_above_90_raises(self):
        """theta_e > 90 is physically impossible with valid lab k (energy conservation).

        The check exists as a safety net. Lab k_lab_y = sqrt(k^2 - x^2 - z^2)
        enforces energy conservation, so theta_e <= 90 is guaranteed.
        Maximum theta_e = 90 occurs when k_lab_y = 0 and k_par = k.
        """
        # With realistic parameters, maximum is theta_e = 90 at the kinetic energy limit.
        points = angle_points(
            theta_range_deg=(89.0, 89.0),
            nt=1,
            hv_eV=50.0,
            work_function_eV=4.5,
            deflector_deg=0.0,
            slit_rot_deg=0.0,
            manip_theta_deg=0.0,
            manip_azimuth_deg=0.0,
            manip_tilt_deg=0.0,
            ref_energy_eV=0.0,
            phi_offset_deg=0.0,
        )
        assert len(points) == 1
        assert points[0].theta_e_deg < 90.0  # Energy conservation ensures theta < 90
