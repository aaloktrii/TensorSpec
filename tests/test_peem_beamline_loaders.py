"""Loader-only tests for UE49 / UView / P3B (core/io/peem_beamline_loaders.py).

Imaging / PeemService / XPEEM-map GUI tests stay with PR #6 until that tab is merged.
"""
import numpy as np
import pytest
from test_als_p3b import write_p3b
from test_esm_uview_loader import _acq as uview_acq

from tensorspec.core.io import peem_beamline_loaders
from tensorspec.core.peem import synthetic


def _ue49(tmp_path):
    frames = np.stack([np.full((16, 16), v, np.float32) for v in (1100, 1100, 900, 900)])
    names = ["Cr_pos_0001.tif", "Cr_pos_0002.tif", "Cr_neg_0001.tif", "Cr_neg_0002.tif"]
    return synthetic.write_ue49_scan(tmp_path, "X_0002", frames, names)


def _p3b(tmp_path, n=2, shape=(16, 16)):
    d = tmp_path / "S2_IF_AL_011"
    d.mkdir()
    for i in range(2 * n):
        pol = 0.9 if i < n else -0.9
        write_p3b(
            d / f"S2_IF_AL_011#{i:04d}.P3B",
            np.full(shape, 1100 if pol > 0 else 900, np.uint16),
            pol=pol,
        )
    return d


def test_detect_each_format(tmp_path):
    txt = _ue49(tmp_path)
    assert peem_beamline_loaders.detect_format(txt) == ("ue49", txt)
    assert peem_beamline_loaders.detect_format(txt.with_suffix("")) == ("ue49", txt)
    cl = uview_acq(tmp_path, "CL", 1100)
    assert peem_beamline_loaders.detect_format(cl)[0] == "uview"
    assert peem_beamline_loaders.detect_format(next(cl.glob("*.dat")))[0] == "uview"
    als = _p3b(tmp_path)
    assert peem_beamline_loaders.detect_format(als)[0] == "p3b"
    (tmp_path / "junk").mkdir()
    with pytest.raises(ValueError, match="no UE49"):
        peem_beamline_loaders.detect_format(tmp_path / "junk")


def test_mixed_formats_refused(tmp_path):
    with pytest.raises(ValueError, match="mixed formats"):
        peem_beamline_loaders.load_beamline([_ue49(tmp_path), _p3b(tmp_path)])


def test_p3b_stack_pol_and_i0_default_off(tmp_path):
    t = peem_beamline_loaders.load_beamline([_p3b(tmp_path)])
    md = t.metadata
    assert t.value.shape == (4, 16, 16)
    assert md["pol"] == ["CP", "CP", "CM", "CM"]
    assert md["format"] == "p3b" and md["default_convention"] == "ue49"
    assert md["i0_normalized"] is False and md["um_per_px"] == pytest.approx(20.76 / 16)


def test_beamline_tokens_go_through_the_original_parser():
    from tensorspec.core.io.peem_loaders import infer_pol_from_name
    from tensorspec.core.io import ue49_loaders, uview_loaders

    assert infer_pol_from_name("Cr_pos_0000") == "unknown"  # original behaviour unchanged
    assert ue49_loaders.pol_from_name("Cr_pos_0000") == "CP"
    assert ue49_loaders.pol_from_name("position_scan_LV") == "LV"
    assert uview_loaders.pol_from_name("XPEEM_20um_2CA_1AN_579eV_Cr2p_CR_0deg") == "CR"


def test_loading_does_not_need_names(tmp_path):
    """Format from file contents; UView pol from user when folders lack CL/CR tokens."""
    import shutil

    a, b = tmp_path / "first", tmp_path / "second"
    shutil.move(uview_acq(tmp_path, "CL", 1100), a)
    shutil.move(uview_acq(tmp_path, "CR", 900), b)
    assert peem_beamline_loaders.detect_format(a)[0] == "uview"
    assert peem_beamline_loaders.detect_pol(a)[0] is None
    with pytest.raises(ValueError, match="set its polarization"):
        peem_beamline_loaders.load_beamline([a, b])
    t = peem_beamline_loaders.load_beamline([a, b], pol={str(a): "CP", str(b): "CM"})
    assert set(t.metadata["pol"]) == {"CP", "CM"} and t.metadata["pol"][0] == "CP"

    d = _p3b(tmp_path)
    for i, f in enumerate(sorted(d.glob("*.P3B"))):
        f.rename(d / f"frame_{i}.P3B")
    (d / "processed.P3B").write_bytes(b"not a raw frame")
    t = peem_beamline_loaders.load_beamline([d])
    assert t.metadata["pol"] == ["CP", "CP", "CM", "CM"]
    assert peem_beamline_loaders.detect_pol(d)[0] == "per frame"
    with pytest.raises(ValueError, match="cannot be set by hand"):
        peem_beamline_loaders.load_beamline([d], pol={str(d): "CM"})

    (tmp_path / "u").mkdir()
    txt = _ue49(tmp_path / "u")
    new = txt.with_name("my_scan.txt")
    txt.rename(new)
    txt.with_suffix("").rename(new.with_suffix(""))
    assert peem_beamline_loaders.detect_pol(new) == (None, "mixed / no polarization in the scan")
    t = peem_beamline_loaders.load_beamline([new], tpx3=False)
    assert t.metadata["pol"] == ["CP", "CP", "CM", "CM"]


def test_find_acquisitions_in_a_folder_of_acquisitions(tmp_path):
    root = tmp_path / "sample"
    (root / "RT").mkdir(parents=True)
    cl, cr = uview_acq(root / "RT", "CL", 1100), uview_acq(root / "RT", "CR", 900)
    loose = next(cl.glob("*.dat"))
    (root / "RT" / "single.dat").write_bytes(loose.read_bytes())
    day = root / "day"
    day.mkdir()
    txt = _ue49(day)
    als = _p3b(root)
    (root / "tensorspec_peem").mkdir()
    found = {str(p.relative_to(root)): fmt for fmt, p in peem_beamline_loaders.find_acquisitions(root)}
    assert found == {
        "RT/" + cl.name: "uview",
        "RT/" + cr.name: "uview",
        "RT/single.dat": "uview",
        "day/" + txt.name: "ue49",
        als.name: "p3b",
    }
    assert peem_beamline_loaders.is_container(root / "RT") and not peem_beamline_loaders.is_container(cl)
