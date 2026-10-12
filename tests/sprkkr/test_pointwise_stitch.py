"""Tests for stitch_points, write_points_json, read_points_json."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from tensorspec.core.dft.sprkkr.outputs import (
    read_points_json,
    stitch_points,
    write_points_json,
)
from tensorspec.core.dft.sprkkr.pointwise import AnglePoint


def make_fake_spc_single_point(
    theta_deg: float, phi_deg: float, ne: int = 3, nt: int = 1, np_: int = 1,
    k_par_val: float = 0.5
) -> str:
    """Synthesize a minimal-but-valid single-point .spc text.

    Args:
        theta_deg: Single theta angle value (in degrees)
        phi_deg: Single phi angle value (in degrees) — not written to file, but used for header
        ne: Number of energy points
        nt: Number of theta points (should be 1 for pointwise)
        np_: Number of phi points (should be 1 for pointwise)
        k_par_val: Value for column 7 (k_par), must be non-zero for ratio test

    Returns:
        Text content of a valid .spc file
    """
    energy = np.linspace(-1.0, 0.1, ne)[::-1]  # descending
    ef_ry = 0.5

    lines = [
        "KEYWORD   ARPES",
        f"NE          {ne}",
        f"EFERMI      {ef_ry:.5f}",
        f"NT          {nt}",
        f"NP          {np_}",
        "#######################",
    ]

    # Energy descending, phi loop, theta ascending
    for e in energy:
        for _p in range(np_):
            for i in range(nt):
                t = theta_deg
                # 8 columns: theta, E, I_tot, I_up, I_dn, pol, k_par, det
                row = " ".join(
                    f"{v:.5E}"
                    for v in (t, e, 1.0, 0.5, 0.5, 0.0, k_par_val, 0.0)
                )
                lines.append("    " + row)

    return "\n".join(lines) + "\n"


def parse_spc_from_text(text: str) -> xr.Dataset:
    """Parse .spc text content and return xr.Dataset."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".spc", delete=False) as f:
        f.write(text)
        temp_path = f.name

    try:
        from tensorspec.core.dft.sprkkr.outputs import parse_spc
        ds = parse_spc(temp_path)
        return ds
    finally:
        Path(temp_path).unlink()


class TestStitchPoints:
    """Tests for stitch_points function."""

    def test_stitch_points_theta_axis_is_lab_slit_angle(self):
        """Verify theta axis is replaced with lab slit angles."""
        points = [
            AnglePoint(index=0, slit_deg=-15.0, theta_e_deg=15.0, phi_e_deg=0.0,
                      k_par=1.0, k_slit=0.2588, k_defl=0.0),
            AnglePoint(index=1, slit_deg=0.0, theta_e_deg=0.0, phi_e_deg=0.0,
                      k_par=1.0, k_slit=0.0, k_defl=0.0),
            AnglePoint(index=2, slit_deg=15.0, theta_e_deg=15.0, phi_e_deg=0.0,
                      k_par=1.0, k_slit=0.2588, k_defl=0.0),
        ]

        spc_texts = [
            make_fake_spc_single_point(-15.0, 0.0, ne=2, k_par_val=1.0),
            make_fake_spc_single_point(0.0, 0.0, ne=2, k_par_val=1.0),
            make_fake_spc_single_point(15.0, 0.0, ne=2, k_par_val=1.0),
        ]
        datasets = [parse_spc_from_text(text) for text in spc_texts]

        merged = stitch_points(datasets, points, deflector_deg=0.0)

        np.testing.assert_array_almost_equal(
            merged["theta"].values,
            [-15.0, 0.0, 15.0]
        )

    def test_stitch_points_phi_axis_is_deflector(self):
        """Verify phi axis is set to deflector angle."""
        points = [
            AnglePoint(index=0, slit_deg=0.0, theta_e_deg=0.0, phi_e_deg=-10.3,
                      k_par=1.0, k_slit=0.0, k_defl=-0.1787),
        ]

        spc_text = make_fake_spc_single_point(0.0, -10.3, ne=2, k_par_val=1.0)
        datasets = [parse_spc_from_text(spc_text)]

        merged = stitch_points(datasets, points, deflector_deg=-10.3)

        np.testing.assert_array_almost_equal(merged["phi"].values, [-10.3])

    def test_stitch_points_k_par_is_signed_lab_slit_k(self):
        """Verify k_par is recomputed as signed lab slit k."""
        points = [
            AnglePoint(index=0, slit_deg=-15.0, theta_e_deg=15.0, phi_e_deg=0.0,
                      k_par=1.0, k_slit=-0.2588, k_defl=0.0),
        ]

        # SPR-KKR reports k_par_sprkkr = 1.0
        # Lab ratio = k_slit / k_par = -0.2588 / 1.0 = -0.2588
        # After stitch: k_par should be approximately -0.2588
        spc_text = make_fake_spc_single_point(-15.0, 0.0, ne=2, k_par_val=1.0)
        datasets = [parse_spc_from_text(spc_text)]

        merged = stitch_points(datasets, points, deflector_deg=0.0)

        # Check that k_par is now the lab slit k (scaled by ratio)
        assert "k_par" in merged.data_vars
        k_par_val = merged["k_par"].values[0, 0, 0]
        np.testing.assert_almost_equal(k_par_val, -0.2588, decimal=4)

    def test_stitch_points_keeps_sprkkr_k_par(self):
        """Verify original SPR-KKR k_par is preserved as k_par_sprkkr."""
        points = [
            AnglePoint(index=0, slit_deg=0.0, theta_e_deg=0.0, phi_e_deg=0.0,
                      k_par=1.0, k_slit=0.0, k_defl=0.0),
        ]

        spc_text = make_fake_spc_single_point(0.0, 0.0, ne=2, k_par_val=1.0)
        datasets = [parse_spc_from_text(spc_text)]

        merged = stitch_points(datasets, points, deflector_deg=0.0)

        assert "k_par_sprkkr" in merged.data_vars
        k_par_sprkkr_val = merged["k_par_sprkkr"].values[0, 0, 0]
        np.testing.assert_almost_equal(k_par_sprkkr_val, 1.0)

    def test_stitch_points_length_mismatch_raises(self):
        """Verify ValueError is raised when dataset and point lists have different lengths."""
        points = [
            AnglePoint(index=0, slit_deg=0.0, theta_e_deg=0.0, phi_e_deg=0.0,
                      k_par=1.0, k_slit=0.0, k_defl=0.0),
        ]

        spc_texts = [
            make_fake_spc_single_point(0.0, 0.0, ne=2, k_par_val=1.0),
            make_fake_spc_single_point(15.0, 0.0, ne=2, k_par_val=1.0),
        ]
        datasets = [parse_spc_from_text(text) for text in spc_texts]

        with pytest.raises(ValueError, match="len\\(datasets\\).*!=.*len\\(points\\)"):
            stitch_points(datasets, points, deflector_deg=0.0)

    def test_points_json_round_trip(self):
        """Verify write_points_json and read_points_json are inverses."""
        points = [
            AnglePoint(index=0, slit_deg=-15.0, theta_e_deg=15.0, phi_e_deg=10.5,
                      k_par=1.0, k_slit=0.2588, k_defl=0.1736),
            AnglePoint(index=1, slit_deg=0.0, theta_e_deg=0.0, phi_e_deg=0.0,
                      k_par=1.0, k_slit=0.0, k_defl=0.0),
        ]
        meta = {"hv_eV": 84.0, "deflector_deg": 0.0, "dataset": "test"}

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "points.json"

            # Write
            write_points_json(str(json_path), points, meta)
            assert json_path.exists()

            # Read
            read_points, read_meta = read_points_json(str(json_path))

            # Verify
            assert len(read_points) == len(points)
            for orig, read in zip(points, read_points):
                assert read.index == orig.index
                assert read.slit_deg == orig.slit_deg
                assert read.theta_e_deg == orig.theta_e_deg
                assert read.phi_e_deg == orig.phi_e_deg
                assert read.k_par == orig.k_par
                assert read.k_slit == orig.k_slit
                assert read.k_defl == orig.k_defl

            assert read_meta == meta


def test_stitch_points_unsorted_points_align_by_slit_angle():
    """Points given out of slit order: k_par ratio must follow the coordinate, not position."""
    import numpy as np
    from tensorspec.core.dft.sprkkr.pointwise import angle_points
    from tensorspec.core.dft.sprkkr.outputs import stitch_points
    pts = angle_points((-15.0, 15.0), 3, hv_eV=84.0, work_function_eV=4.5, deflector_deg=-10.3)
    order = [2, 0, 1]
    shuffled = [pts[i] for i in order]
    dss = []
    for p in shuffled:
        ds = parse_spc_from_text(make_fake_spc_single_point(theta_deg=p.theta_e_deg, phi_deg=p.phi_e_deg, k_par_val=p.k_par))
        dss.append(ds)
    merged = stitch_points(dss, shuffled, deflector_deg=-10.3)
    np.testing.assert_allclose(merged["theta"].values, [-15.0, 0.0, 15.0])
    kp = merged["k_par"].isel(energy=0, phi=0).values
    np.testing.assert_allclose(kp, [pts[0].k_slit, pts[1].k_slit, pts[2].k_slit], atol=1e-4)  # fake .spc writes k_par at ~6 sig figs
