"""Unit tests for DataViewerPanel.build_floor_reference (no live GUI clicks)."""

import numpy as np
import pytest

from tensorspec.core.data_models import TensorData
from tensorspec.gui.components.data_viewer_panel import DataViewerPanel


def _make_tensor(*, labels=("Y", "X", "Energy", "Angle")):
    rng = np.random.default_rng(0)
    shape = (4, 5, 8, 6)
    value = rng.random(shape, dtype=np.float32)
    axes = [np.arange(n, dtype=np.float64) for n in shape]
    units = ["mm", "mm", "eV", "deg"]
    return TensorData(
        value=value,
        axes=axes,
        labels=list(labels),
        units=units,
        data_type="ARPES",
        metadata={},
    )


def test_build_floor_reference_integrates_yx(qapp):
    td = _make_tensor()
    panel = DataViewerPanel()
    panel.tensor_data = td
    panel.global_coords = {0: 0, 1: 0, 2: 3, 3: 2}
    panel.global_halfwidths = {0: 0, 1: 0, 2: 1, 3: 1}

    ref = panel.build_floor_reference(
        y_label="Y",
        x_label="X",
        source_id="synthetic_00737.h5",
        reduce_mode="sum",
    )

    expect = td.value[:, :, 2:5, 1:4].sum(axis=(2, 3))
    assert ref.map.shape == (4, 5)
    np.testing.assert_allclose(ref.map, expect, rtol=1e-5)
    assert ref.roi["source_id"] == "synthetic_00737.h5"
    assert ref.roi["reduce_mode"] == "sum"
    assert ref.roi["display_labels"] == ["Y", "X"]
    assert ref.y_axis is not None and ref.x_axis is not None
    np.testing.assert_array_equal(ref.y_axis, td.axes[0])
    np.testing.assert_array_equal(ref.x_axis, td.axes[1])


def test_build_floor_reference_keeps_slit_defl_axis(qapp):
    """5D cubes have Slit Defl. plus detector Angle; both must stay distinct."""
    rng = np.random.default_rng(1)
    shape = (3, 4, 5, 6, 7)
    td = TensorData(
        value=rng.random(shape, dtype=np.float32),
        axes=[np.arange(n, dtype=np.float64) for n in shape],
        labels=["Y", "X", "Slit Defl.", "Energy", "Angle"],
        units=["mm", "mm", "deg", "eV", "deg"],
        data_type="ARPES",
        metadata={"source_path": "/tmp/20260630_00742.h5"},
    )
    panel = DataViewerPanel()
    panel.tensor_data = td
    panel.global_coords = {0: 0, 1: 0, 2: 2, 3: 3, 4: 1}
    panel.global_halfwidths = {0: 0, 1: 0, 2: 1, 3: 0, 4: 0}

    ref = panel.build_floor_reference(
        y_label="Y",
        x_label="X",
        source_id="20260630_00742.h5",
        reduce_mode="sum",
    )
    expect = td.value[:, :, 1:4, 3:4, 1:2].sum(axis=(2, 3, 4))
    assert ref.map.shape == (3, 4)
    np.testing.assert_allclose(ref.map, expect, rtol=1e-5)
    labels = [d["label"] for d in ref.roi["dims"]]
    assert labels == ["Slit Defl.", "Energy", "Angle"]
    assert DataViewerPanel._canonicalize_ssl_labels(td.labels) == [
        "Y",
        "X",
        "Slit Defl.",
        "Energy",
        "Angle",
    ]


def test_build_floor_reference_accepts_slit_alias(qapp):
    td = _make_tensor(labels=("Y", "X", "Energy", "Slit Angle"))
    panel = DataViewerPanel()
    panel.tensor_data = td
    panel.global_coords = {0: 0, 1: 0, 2: 3, 3: 2}
    panel.global_halfwidths = {0: 0, 1: 0, 2: 0, 3: 0}

    ref = panel.build_floor_reference(
        source_id="synthetic_slit.h5",
        reduce_mode="mean",
    )
    assert ref.map.shape == (4, 5)
    assert ref.roi["reduce_mode"] == "mean"


def test_build_floor_reference_rejects_no_data(qapp):
    panel = DataViewerPanel()
    with pytest.raises(ValueError, match="no data loaded"):
        panel.build_floor_reference(source_id="x", reduce_mode="sum")


def test_build_floor_reference_rejects_missing_yx(qapp):
    td = _make_tensor(labels=("Energy", "Angle", "ky", "kx"))
    # reshape to match labels length — value still 4D
    panel = DataViewerPanel()
    panel.tensor_data = td
    panel.global_coords = {0: 0, 1: 0, 2: 0, 3: 0}
    panel.global_halfwidths = {0: 0, 1: 0, 2: 0, 3: 0}
    with pytest.raises(ValueError, match="Y"):
        panel.build_floor_reference(
            y_label="Y",
            x_label="X",
            source_id="x",
            reduce_mode="sum",
        )
