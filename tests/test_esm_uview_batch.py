"""UView reader and ESM (BNL) folder planning on synthetic UKSOFT2001 files."""
import struct

import numpy as np
import pytest

from tensorspec.core.io import uview_loaders as uview
from tensorspec.core.peem.esm import batch


def write_uview(path, img, temp_c=-166.8, fov="30", pressures=None, filetime=134248151261920000):
    h, w = img.shape
    fh = bytearray(104)
    fh[:10] = b"UKSOFT2001"
    struct.pack_into("<3h", fh, 20, 104, 9, 16)
    struct.pack_into("<h", fh, 28, 4096)
    fh[30], fh[31] = 4, 4
    struct.pack_into("<4h", fh, 40, w, h, 1, 0)
    ih = bytearray(288 + 128)
    struct.pack_into("<2h", ih, 0, 288, 7)
    struct.pack_into("<q", ih, 8, filetime)
    tags = bytearray()
    tags += bytes([0xD2]) + b"Acc. Lens1\x00" + struct.pack("<f", 472.5)
    for name, val in (pressures or {"COL": 1.9e-10, "MCH": 5.5e-10, "PCH": 1.6e-9}).items():
        tags += bytes([0xEB]) + name.encode() + b"\x00Torr\x00" + struct.pack("<f", val)
    tags += bytes([0x6E]) + fov.encode() + b"\xb5m\x00" + struct.pack("<f", 4096.0)
    tags += bytes([0x27]) + b"Sample Temp.4\x00" + struct.pack("<f", temp_c)
    tags += bytes([0x26]) + b"Start Voltage1\x00" + struct.pack("<f", 0.0)
    tags += b"\xff"
    pad = bytearray(16384 - 104 - len(ih) - len(tags))
    path.write_bytes(bytes(fh) + bytes(ih) + bytes(tags) + bytes(pad) + img.astype("<u2").tobytes())


def test_read_uview_image_and_metadata(tmp_path):
    img = np.arange(64 * 48, dtype=np.uint16).reshape(48, 64)
    p = tmp_path / "a.dat"
    write_uview(p, img)
    out, meta = uview.read_uview(p)
    assert out.shape == (48, 64) and np.array_equal(out, img.astype(np.float32))
    assert meta["sample_temp_K"] == pytest.approx(106.35, abs=0.01)
    assert meta["pressure_MCH_torr"] == pytest.approx(5.5e-10)
    assert meta["fov_um"] == 30.0
    assert meta["timestamp"].startswith("2026-06-01")
    assert uview.um_per_px(meta) == pytest.approx(30 / 1024)


def test_not_uview(tmp_path):
    p = tmp_path / "x.dat"
    p.write_bytes(b"hello" * 100)
    with pytest.raises(ValueError, match="UKSOFT2001"):
        uview.read_uview(p)


@pytest.fixture
def beamtime(tmp_path):
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:96, 0:96]
    mz = np.where((xx // 16) % 2 == 0, 1, -1)
    # textured topography well above the 5 % magnetic contrast, as on the real BNL images;
    # with weak topography CL/CR registration can lock half a domain period off
    # (see dichroism.joint_means, test_half_period_lock_and_opt_in_bound)
    from scipy import ndimage
    tex = ndimage.gaussian_filter(rng.normal(size=(96, 96)), 3)
    topo = 3000 * (1 + 0.25 * tex / tex.std())

    def acq(folder, pol, n=4):
        folder.mkdir(parents=True)
        a = {"CL": 1, "CR": -1}.get(pol, 0) * 0.05
        for i in range(n):
            frame = topo * (1 + a * mz) + rng.normal(0, 5, topo.shape)
            write_uview(folder / f"img_{i:06d}.dat", np.clip(frame, 0, 65535))

    base = tmp_path / "320911_X"
    for t in ("110K", "RT"):
        for pol in ("CL", "CR", "LH", "LV"):
            acq(base / "20260602_CrSBr" / "Sample_2" / t / f"XPEEM_20um_2CA_1AN_579eV_Cr2p_{pol}_0deg", pol)
    acq(base / "20260602_CrSBr" / "Sample_2" / "110K" / "XPEEM_20um_2CA_1AN_579eV_Cr2p_CL_0deg_2", "CL")
    acq(base / "20260602_CrSBr" / "Sample_2" / "110K" / "XPEEM_20um_2CA_1AN_579eV_Cr2p_CR_0deg_2", "CR")
    acq(base / "20260602_CrSBr" / "Sample_2" / "RT" / "XAS_50um_2CA_1AN_570eV_571eV_0.2eV-step_Cr2p_LH", "LH", n=7)
    write_uview(base / "20260602_CrSBr" / "Sample_2" / "RT" / "LEEM_mirror_50um_1.dat", topo.astype(np.uint16))
    return base, mz


def test_plan_and_run(beamtime):
    root, mz = beamtime
    cat = batch.catalog(root)
    groups = {g.gid: g for g in batch.plan(cat)}
    assert len(cat["leem"]) == 1
    assert len(groups["Sample_2_110K_20um_579eV_0deg_XMCD"].units) == 2  # 0deg and 0deg_2 repeats
    res = batch.run(root, preview=None, log=lambda *_: None)
    rows = {r["id"]: r for r in res["rows"]}
    assert rows["Sample_2_110K_20um_579eV_0deg_XMCD"]["status"] == "ok"
    assert rows["Sample_2_110K_20um_579eV_0deg_XMCD"]["T_K_measured"] == pytest.approx(106.35, abs=0.1)
    split = rows["Sample_2_110K_20um_579eV_0deg_XLD_vs_RT"]
    assert split["status"] == "ok"
    assert rows["Sample_2_110K_20um_579eV_0deg_XMCD_vs_RT"]["status"] == "ok"
    xas = next(r for r in res["rows"] if r["type"] == "spectra:XAS")
    assert "7 frames but" in xas["energy_note"]  # (571-570)/0.2 + 1 = 6 points named
    # pcd = (CL - CR)/(CL + CR): CL is brighter where mz = +1
    import xarray as xr
    nc = root / rows["Sample_2_110K_20um_579eV_0deg_XMCD"]["path"] / "merged" / "maps.nc"
    with xr.open_dataset(nc) as ds:
        pcd = ds["pcd"].values
    inner = (slice(20, -20), slice(20, -20))
    ok = np.isfinite(pcd[inner])
    assert np.mean(np.sign(pcd[inner][ok]) == mz[inner][ok]) > 0.9
