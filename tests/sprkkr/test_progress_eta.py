"""Gate: progress.EtaModel material-aware calibration. No binary, no Qt."""
import json

import pytest

from tensorspec.core.dft.sprkkr.params import ArpesParams
from tensorspec.core.dft.sprkkr.progress import EtaModel


def _params(n_points=6):
    # ne * nt * np_ == n_points
    return ArpesParams(ne=n_points, nt=1, np_=1)


def test_old_two_arg_call_signatures_still_work():
    """estimate_seconds(params, nproc) / calibrate(params, nproc, wall, host)
    -- the pre-existing call shapes -- keep working unchanged."""
    model = EtaModel(t_point_s=2.0)
    assert model.estimate_seconds(_params(6), 2) == pytest.approx(6.0)

    model.calibrate(_params(6), nproc=1, wall_s=12.0, host="local")
    assert model.t_point_s == pytest.approx(2.0)


def test_calibrate_and_estimate_prefer_material_over_host():
    model = EtaModel()
    model.calibrate(_params(6), nproc=1, wall_s=12.0, host="local", material="Cu")  # t_point=2.0
    model.calibrate(_params(6), nproc=1, wall_s=60.0, host="local")  # generic host t_point=10.0

    # material-specific point wins for Cu
    assert model.estimate_seconds(_params(6), 1, host="local", material="Cu") == pytest.approx(12.0)
    # unknown material on the same host falls back to the host-only point
    assert model.estimate_seconds(_params(6), 1, host="local", material="VTe2") == pytest.approx(60.0)
    # host with nothing calibrated falls back to the current default t_point_s
    assert model.estimate_seconds(_params(6), 1, host="other") == pytest.approx(model.t_point_s * 6)


def test_calibrate_persists_material_key_to_store(tmp_path):
    store = tmp_path / "eta.json"
    model = EtaModel(store_path=str(store))
    model.calibrate(_params(6), nproc=1, wall_s=12.0, host="local", material="Cu")

    data = json.loads(store.read_text())
    assert data["local"]["t_point_s"] == pytest.approx(2.0)
    assert data["local|Cu"]["t_point_s"] == pytest.approx(2.0)

    reloaded = EtaModel(store_path=str(store))
    reloaded.load(host="local", material="Cu")
    assert reloaded.t_point_s == pytest.approx(2.0)

    # a different material on the same host, not yet calibrated, loads the
    # host-only fallback rather than raising
    reloaded2 = EtaModel(store_path=str(store))
    reloaded2.load(host="local", material="Fe")
    assert reloaded2.t_point_s == pytest.approx(2.0)


def test_load_with_no_store_is_noop():
    model = EtaModel(t_point_s=3.5)
    assert model.load(host="local") == pytest.approx(3.5)
