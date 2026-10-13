"""Real-beamtime integration tests; skipped unless the data folders are given.

UE49_DATA_DIR         folder containing the 'June 2025' beamtime days and
                      'Procedure files' (dead-pixel / NormImage TIFFs)
TENSORSPEC_BNL_PEEM  the BNL (NSLS-II ESM) beamtime folder, e.g. <data>/BNL/<proposal>
"""
import os
from pathlib import Path

import numpy as np
import pytest

from tensorspec.core.workspace import global_workspace
from tensorspec.gui.services.peem_service import PeemService

UE49 = os.environ.get("UE49_DATA_DIR")
BNL = os.environ.get("TENSORSPEC_BNL_PEEM")


def _ue49_service() -> tuple[PeemService, Path]:
    root = Path(UE49)
    svc = PeemService()
    procs = root / "Procedure files"
    dead = next(procs.glob("deadpixel_*.tif"), None)
    norm = next(procs.glob("NormImage_*.tif"), None)
    svc.set_detector_maps(str(dead) if dead else None, str(norm) if norm else None)
    return svc, root


@pytest.mark.skipif(not UE49, reason="set UE49_DATA_DIR to run")
def test_ue49_cr_xmcd_domains():
    svc, root = _ue49_service()
    day = root / "June 2025" / "20250603"
    res = svc.load_ue49([day / f"X_000{i}.txt" for i in (2, 3, 4)], name="rt_cr_xmcd")
    try:
        assert res["pol_counts"] == {"CM": 40, "CP": 40}  # filenames beat the lagging PolTable
        roi = svc.auto_fov_roi(svc.get_view_tensor("rt_cr_xmcd", "raw", 0))
        pcd = svc.compute_pcd("rt_cr_xmcd", crop_roi=roi)
        assert abs(pcd["mean"]) < 0.01 and 0.005 < pcd["std"] < 0.1
        assert abs(pcd["cm_shift_dy"]) < 5 and abs(pcd["cm_shift_dx"]) < 5
        dm = svc.compute_domain_map("rt_cr_xmcd", roi=roi, threshold=0.1, smooth_sigma=1.5)
        assert dm["f_up"] > 0.2 and dm["f_down"] > 0.2
        assert 0.05 < dm["width_um_up"] < 2.0
    finally:
        global_workspace.remove("rt_cr_xmcd")


@pytest.mark.skipif(not UE49, reason="set UE49_DATA_DIR to run")
def test_ue49_cr_xmld_matched_fov():
    svc, root = _ue49_service()
    day = root / "June 2025"
    with pytest.raises(ValueError, match="fields of view"):
        svc.load_ue49([day / "20250606" / "Mono_S_0199.txt", day / "20250606" / "Mono_S_0201.txt"], name="rt_bad")
    res = svc.load_ue49([day / "20250607" / "Mono_S_0226.txt", day / "20250607" / "Mono_S_0227.txt"],
                        name="rt_cr_xmld")
    try:
        assert set(res["pol_counts"]) == {"LH", "LV"} and res["fov_um"] == 10.0
        out = svc.compute_pmld("rt_cr_xmld", 584.4, 585.4, half_width=0.15)
        for kind in ("pld", "pmld", "pmnld"):
            assert out["stats"][kind]["n_finite"] > 1000
    finally:
        global_workspace.remove("rt_cr_xmld")


@pytest.mark.skipif(not BNL, reason="set TENSORSPEC_BNL_PEEM to run")
def test_bnl_esm_uview_catalog_and_metadata():
    """BNL NSLS-II ESM beamtime: Elmitec UView .dat files."""
    from tensorspec.core.io import uview_loaders as uview
    from tensorspec.core.peem.esm import batch as esm

    root = Path(BNL)
    cat = esm.catalog(root)
    groups = esm.plan(cat)
    assert len(cat["acqs"]) >= 40 and len(groups) >= 20
    assert all(g.units for g in groups)  # every CL has a CR and every LH an LV
    s1 = next(g for g in groups if g.gid == "Sample_1_110K_30um_579.2eV_0deg_XMCD")
    assert len(s1.units) == 2  # "0deg" and "0deg_2" repeats
    img, meta = uview.read_uview(next(Path(s1.units[0][0].path).glob("*.dat")))
    assert img.shape == (1024, 1024)
    assert 95 < meta["sample_temp_K"] < 120 and meta["fov_um"] == 30.0
    assert 1e-11 < meta["pressure_COL_torr"] < 1e-8


def _d0613(root: Path) -> Path:
    return root / "June 2025" / "20250613"


def _robust_corr(a, b):
    """Correlation after clipping to the 2-98 % range and 2 px smoothing (hot pixels dominate otherwise)."""
    from scipy import ndimage

    def prep(x):
        lo, hi = np.nanpercentile(x, [2, 98])
        return ndimage.gaussian_filter(np.clip(np.nan_to_num(x, nan=lo), lo, hi), 2)

    return _corr(prep(a), prep(b))


def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))


@pytest.mark.skipif(not UE49, reason="set UE49_DATA_DIR to run")
def test_0613_coadd_and_drift_match_igor():
    import tifffile

    from tensorspec.core.io import ue49_loaders as loaders
    from tensorspec.core.peem import dichroism, registration

    svc, root = _ue49_service()
    day = _d0613(root)
    td = loaders.load_scans([day / f"X_{i:04d}.txt" for i in (215, 216, 217)],
                            dead_map_path=svc.dead_map_path, norm_image_path=svc.norm_image_path)
    ny, nx = td.value.shape[1:]
    aligned, shifts = registration.register_stack(td.value, 0, box=(ny // 4, 3 * ny // 4, nx // 4, 3 * nx // 4))
    igor = np.loadtxt(day / "displacement_X_0217.txt", skiprows=1)  # Igor out[p,q] = in[p+pp, q+qq], p = x
    diff = np.c_[-shifts[:, 1], -shifts[:, 0]] - igor
    assert np.median(np.abs(diff)) < 0.5
    groups = dichroism.group_by_key(td.metadata["frame_names"])
    ref = tifffile.imread(day / "CrL2_A_0217.tif").astype(float)
    assert _robust_corr(aligned[groups["CrL2_A"]].mean(0), ref) > 0.95


@pytest.mark.skipif(not UE49, reason="set UE49_DATA_DIR to run")
def test_0613_xmcd_more_reproducible_than_igor():
    import tifffile

    from tensorspec.core.peem import dichroism

    svc, root = _ue49_service()
    day = _d0613(root)
    maps, igor = [], []
    for ids, tag in (((193, 194, 195), "0195"), ((196, 197, 198), "0198")):
        name = f"rt_0613_{tag}"
        svc.load_ue49([day / f"X_{i:04d}.txt" for i in ids], name=name)
        try:
            roi = svc.auto_fov_roi(svc.get_view_tensor(name, "raw", 0))
            svc.compute_pcd(name, crop_roi=roi)
            a = svc.get_map(name, "pcd")
            maps.append(a)
            # Igor's XMCD (neg - pos)/(neg + pos): the suite uses the same UE49 sign
            ig = dichroism.asymmetry(tifffile.imread(day / f"Cr_neg_{tag}.tif").astype(float),
                                     tifffile.imread(day / f"Cr_pos_{tag}.tif").astype(float))
            igor.append(np.where(np.isfinite(a), ig, np.nan))
            assert _corr(a, ig) > 0.5  # same magnetic contrast as the Igor XMCD
        finally:
            global_workspace.remove(name)
    assert _corr(*maps) > _corr(*igor) + 0.05


@pytest.mark.skipif(not UE49, reason="set UE49_DATA_DIR to run")
def test_0613_xmld_l2_and_reference_split_flags_noise():
    svc, root = _ue49_service()
    day = _d0613(root)
    svc.load_ue49([day / f"X_{i:04d}.txt" for i in range(205, 214)], name="rt_xmld45")
    svc.load_ue49([day / "Mono_S_0224.txt"], name="rt_ref200")
    try:
        out = svc.compute_l2_ld("rt_xmld45", 583.9, 585.1, half_width=0.05)
        assert out["pol"] == "LV" and out["n_a"] == 120 and out["n_b"] == 120
        svc.compute_l2_ld("rt_ref200", 583.9, 585.1, half_width=0.3)
        split = svc.split_ld_by_reference("rt_xmld45", "rt_ref200")
        assert abs(split["reference_shift_dy"]) < 5 and 10 < abs(split["reference_shift_dx"]) < 25
        assert split["reliable"] is False  # 6-frame 200 K reference vs 120-frame map
        smoothed = svc.split_ld_by_reference("rt_xmld45", "rt_ref200", reference_smooth_sigma=4)
        assert smoothed["noise_ratio"] < split["noise_ratio"]
    finally:
        global_workspace.remove("rt_xmld45")
        global_workspace.remove("rt_ref200")
