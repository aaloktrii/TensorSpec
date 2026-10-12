"""Tests for spc_results.spc_paths_to_results core function (no Qt).

Pure Python, no PySide6 imports. Tests the point-wise and fallback .spc loading paths.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tensorspec.core.dft.sprkkr import (
    angle_points,
    write_points_json,
    spc_paths_to_results,
)
from tests.sprkkr.test_pointwise_stitch import make_fake_spc_single_point


class TestSpcPathsToResultsPointwise:
    """Point-wise mode: pointwise_points.json + 3 single-point .spc files."""

    def test_pointwise_with_json_and_three_spc_files(self, tmp_path):
        """Create 3 fake single-point .spc files + pointwise_points.json, load via spc_paths_to_results."""
        # Create angle_points: 3 points at slit angles -15, 0, +15 deg
        points = angle_points(
            theta_range_deg=(-15, 15),
            nt=3,
            hv_eV=84.0,
            work_function_eV=4.5,
            deflector_deg=-10.3,
        )
        assert len(points) == 3

        # Create pointwise_points.json
        json_path = tmp_path / "pointwise_points.json"
        write_points_json(str(json_path), points, meta={"deflector_deg": -10.3})

        # Create 3 fake .spc files in subdirectories matching the pattern
        dataset = "test_arpes"
        spc_files = []
        ne = 5
        for p in points:
            # Create dataset_pNNNN/ subdirectory
            subdir = tmp_path / f"{dataset}_p{p.index:04d}"
            subdir.mkdir(parents=True, exist_ok=True)

            # Create .spc file
            spc_path = subdir / f"{dataset}_p{p.index:04d}_ARPES_data.spc"
            spc_text = make_fake_spc_single_point(
                theta_deg=p.theta_e_deg,
                phi_deg=p.phi_e_deg,
                ne=ne,
                nt=1,
                np_=1,
                k_par_val=0.8,
            )
            spc_path.write_text(spc_text)
            spc_files.append(str(spc_path))

        # Call spc_paths_to_results with points_json
        result = spc_paths_to_results(spc_files, points_json=str(json_path))

        # Verify return dict shape
        assert "intensity_broadened" in result
        assert "energy" in result
        assert "theta" in result
        assert "phi" in result

        # Check theta axis: should be the 3 lab slit angles
        np.testing.assert_array_almost_equal(result["theta"], [-15.0, 0.0, 15.0], decimal=1)

        # Check phi axis: should be the deflector angle
        np.testing.assert_array_almost_equal(result["phi"], [-10.3], decimal=1)

        # Check intensity shape: (3 theta points, 1 phi point, NE)
        assert result["intensity_broadened"].shape == (3, 1, ne)


class TestSpcPathsToResultsFallback:
    """Fallback mode: no JSON, sibling .inp with SPEC_EL PHI extraction."""

    def test_fallback_reads_phi_from_sibling_inp(self, tmp_path):
        """Create 1 fake .spc + sibling .inp with SPEC_EL PHI, load via spc_paths_to_results."""
        # Create a fake .spc file
        spc_path = tmp_path / "test_ARPES_data.spc"
        ne = 4
        spc_text = make_fake_spc_single_point(
            theta_deg=5.0,
            phi_deg=-10.3,
            ne=ne,
            nt=1,
            np_=1,
            k_par_val=0.8,
        )
        spc_path.write_text(spc_text)

        # Create sibling .inp file with SPEC_EL PHI
        inp_path = tmp_path / "test.inp"
        inp_text = """\
SPEC_EL
  THETA = {5.0, 5.0}
  PHI = -10.3
  NT = 1
  NP = 1
  NE = 40
  EMIN = -1.0
  EMAX = 0.1
"""
        inp_path.write_text(inp_text)

        # Call spc_paths_to_results without points_json (fallback mode)
        result = spc_paths_to_results([str(spc_path)])

        # Verify return dict shape
        assert "intensity_broadened" in result
        assert "energy" in result
        assert "theta" in result
        assert "phi" in result

        # Check phi: should be extracted from the .inp SPEC_EL PHI
        np.testing.assert_array_almost_equal(result["phi"], [-10.3], decimal=1)

        # Check intensity shape: (1 theta, 1 phi, NE)
        assert result["intensity_broadened"].shape == (1, 1, ne)
