"""UView folder loader + PeemService.load_uview / compute_pcd on synthetic UView files."""
import struct

import numpy as np
import pytest

from tensorspec.core.io import uview_loaders as esm_loaders
from tensorspec.core.io import uview_loaders as uview


def _write_uview(path, img, fov_um=20.0):
    h, w = img.shape
    hdr = bytearray(104)
    hdr[:10] = uview.MAGIC
    struct.pack_into("<3h", hdr, 20, 104, 9, 16)
    struct.pack_into("<h", hdr, 28, 4096)
    hdr[30], hdr[31] = 4, 4
    struct.pack_into("<4h", hdr, 40, w, h, 1, 0)
    ihdr = bytearray(288)
    meta = b"Sample Temp.5\x00" + struct.pack("<f", 110.0) + f"{fov_um:g} \xb5m".encode("latin1") + b"\x00" + struct.pack("<f", 0)
    path.write_bytes(bytes(hdr) + bytes(ihdr) + meta + img.astype("<u2").tobytes())


def _acq(root, pol, value, n=3, shape=(32, 32)):
    d = root / f"XPEEM_20um_2CA_1AN_579eV_Cr2p_{pol}_0deg"
    d.mkdir()
    for i in range(n):
        _write_uview(d / f"XPEEM_{pol}_{i:06d}.dat", np.full(shape, value, dtype=np.uint16))
    return d


def test_pol_from_name():
    assert esm_loaders.pol_from_name("XPEEM_20um_2CA_1AN_579eV_Cr2p_CL_+15deg") == "CL"
    assert esm_loaders.pol_from_name("XPEEM_30um_x_LV_0deg_2") == "LV"
    assert esm_loaders.pol_from_name("LEEM_mirror_20um") is None


def test_load_uview_folders(tmp_path):
    cl, cr = _acq(tmp_path, "CL", 1100), _acq(tmp_path, "CR", 900)
    t = esm_loaders.load_uview_folders([cl, cr])
    assert t.value.shape == (6, 32, 32)
    assert t.metadata["pol"] == ["CP"] * 3 + ["CM"] * 3
    assert t.metadata["fov_um"] == 20.0
    assert t.metadata["um_per_px"] == pytest.approx(20.0 / 1024)
    assert t.metadata["temperature_K"] == pytest.approx(110.0)


def test_mixed_fov_refused(tmp_path):
    a = _acq(tmp_path, "CL", 1000)
    d = tmp_path / "XPEEM_30um_2CA_1AN_579eV_Cr2p_CR_0deg"
    d.mkdir()
    _write_uview(d / "x.dat", np.full((32, 32), 1000, np.uint16), fov_um=30.0)
    with pytest.raises(ValueError, match="different fields of view"):
        esm_loaders.load_uview_folders([a, d])


def test_service_pcd_bnl_sign(tmp_path):
    from tensorspec.gui.services.peem_service import PeemService
    svc = PeemService()
    cl, cr = _acq(tmp_path, "CL", 1100), _acq(tmp_path, "CR", 900)
    res = svc.load_uview([cl, cr], name="bnl_test")
    assert res["pol_counts"] == {"CM": 3, "CP": 3}
    svc.compute_pcd("bnl_test", convention="cp_minus_cm", register=False)
    pcd = svc.get_map("bnl_test", "pcd")
    assert np.nanmean(pcd) == pytest.approx(0.1, abs=1e-6)  # (1100-900)/(1100+900)
