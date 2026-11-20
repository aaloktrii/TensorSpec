import numpy as np
from pathlib import Path
from tensorspec.core.ml.ssl.reference import (
    FloorReference,
    build_roi_dict,
    integrate_xy_map,
    load_floor_reference,
    save_floor_reference,
)


def test_integrate_xy_map_sums_energy_angle_window():
    # value shape (Y, X, Energy, Angle) — labels match
    rng = np.random.default_rng(0)
    value = rng.random((4, 5, 8, 6), dtype=np.float32)
    labels = ["Y", "X", "Energy", "Angle"]
    coords = {0: 0, 1: 0, 2: 3, 3: 2}
    half = {0: 0, 1: 0, 2: 1, 3: 1}  # Energy 2:5, Angle 1:4
    out = integrate_xy_map(
        value,
        labels=labels,
        coords=coords,
        halfwidths=half,
        y_label="Y",
        x_label="X",
        reduce_mode="sum",
    )
    expect = value[:, :, 2:5, 1:4].sum(axis=(2, 3))
    assert out.shape == (4, 5)
    np.testing.assert_allclose(out, expect, rtol=1e-5)


def test_save_load_roundtrip(tmp_path: Path):
    m = np.arange(12, dtype=np.float32).reshape(3, 4)
    roi = build_roi_dict(
        labels=["Y", "X", "Energy", "Angle"],
        axes=[np.arange(3), np.arange(4), np.linspace(0, 1, 8), np.linspace(-1, 1, 6)],
        coords={0: 0, 1: 0, 2: 3, 3: 2},
        halfwidths={0: 0, 1: 0, 2: 1, 3: 1},
        reduce_mode="sum",
        source_id="synthetic_4d.h5",
        display_y_label="Y",
        display_x_label="X",
        saved_utc="2026-09-07T00:00:00Z",
    )
    path = tmp_path / "floor.npz"
    save_floor_reference(
        path,
        FloorReference(map=m, y_axis=np.arange(3.0), x_axis=np.arange(4.0), roi=roi),
    )
    loaded = load_floor_reference(path)
    np.testing.assert_array_equal(loaded.map, m)
    assert loaded.roi["source_id"] == "synthetic_4d.h5"
    assert loaded.roi["dims"][0]["label"] == "Energy" or any(
        d["label"] == "Energy" for d in loaded.roi["dims"]
    )


def test_integrate_rejects_missing_labels():
    import pytest

    with pytest.raises(ValueError, match="Energy"):
        integrate_xy_map(
            np.zeros((2, 2), dtype=np.float32),
            labels=["Y", "X"],
            coords={0: 0, 1: 0},
            halfwidths={0: 0, 1: 0},
            y_label="Y",
            x_label="X",
            reduce_mode="sum",
        )
