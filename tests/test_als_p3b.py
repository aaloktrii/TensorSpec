"""ALS PEEM-3 .P3B reader on a synthetic file with the beamline record layout."""
import struct

import numpy as np
import pytest

from tensorspec.core.io import p3b_loaders as p3b


def rec(name, typ, values, fmt):
    body = b"".join(struct.pack("<" + fmt, v) for v in values)
    return struct.pack("<i", len(name)) + name.encode() + struct.pack("<ii", typ, len(values)) + body


def write_p3b(path, img, energy=709.98, pol=0.9, i0=0.01):
    h, w = img.shape
    b = struct.pack("<i", 113)
    b += rec("DIMX", 3, [w], "i") + rec("DIMY", 3, [h], "i") + rec("PIXTYPE", 2, [12], "h")
    b += struct.pack("<i", 4) + b"DATA" + struct.pack("<ii", 12, img.size) + img.astype("<u2").tobytes()
    t = b"Tue Jun 01 17:06:26 2021"
    b += struct.pack("<i", 4) + b"TIME" + struct.pack("<ii", 7, len(t)) + t
    b += rec("EXPOSURETIME", 13, [250], "I") + rec("FLATFIELD", 1, [True], "?")
    b += rec("BEAMLINE_ENERGY", 5, [energy], "d") + rec("EPU_POLARIZATION", 5, [pol], "d")
    b += rec("I0", 5, [i0], "d") + rec("FOV", 5, [20.76], "d")
    path.write_bytes(b)


def test_read_p3b(tmp_path):
    img = np.arange(30 * 20, dtype=np.uint16).reshape(30, 20)
    f = tmp_path / "x#0000.P3B"
    write_p3b(f, img, pol=-0.9)
    out, m = p3b.read_p3b(f)
    assert out.shape == (30, 20) and np.array_equal(out, img.astype(np.float32))
    assert m["EPU_POLARIZATION"] == -0.9 and m["BEAMLINE_ENERGY"] == pytest.approx(709.98)
    assert m["EXPOSURETIME"] == 250 and m["FLATFIELD"] is True
    assert m["timestamp"] == "2021-06-01T17:06:26"
    assert p3b.frame_files(f) == [f]


def test_read_dat_flexible_columns(tmp_path):
    f = tmp_path / "a.dat"
    f.write_text("EPU Polarization\tCryostat Temperature\tPressure IG111\nData#\t2\n\t\n0.9\t163.9\t4.8e-9\n-0.9\t164.1\t4.7e-9\n")
    cols = p3b.read_dat(f)
    assert cols["Cryostat Temperature"].tolist() == [163.9, 164.1]
