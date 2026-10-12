"""Gate B tests: fanout.py + vault.py. No binary, no Qt."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

PYMATGEN = importlib.util.find_spec("pymatgen") is not None

from tensorspec.core.dft.sprkkr.fanout import (
    FanoutPlan,
    full_energy_axis,
    plan_jobs,
    split_energy,
    split_hv,
)
from tensorspec.core.dft.sprkkr.params import ArpesParams, ScfParams
from tensorspec.core.dft.sprkkr.vault import Vault, VaultEntry, pot_key

# ---------------------------------------------------------------------------
# fanout: split_energy
# ---------------------------------------------------------------------------

ATOL = 1e-9  # ULP-level linspace re-derivation slop, see fanout.py docstring


@pytest.mark.parametrize("ne", [71, 8])
@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_split_energy_union_equals_full(ne, k):
    params = ArpesParams(e_min_eV=-6.0, e_max_eV=1.0, ne=ne, dataset="arpes")
    full = full_energy_axis(params)

    chunks = split_energy(params, k)

    # every chunk has >= 2 points
    for c in chunks:
        assert c.ne >= 2

    union = np.concatenate([full_energy_axis(c) for c in chunks])
    assert union.shape == full.shape
    assert np.allclose(union, full, atol=ATOL, rtol=0)

    # no chunk overlaps another and coverage is exact (count match + allclose
    # above together prove no gap/no duplicate boundary)
    assert sum(c.ne for c in chunks) == ne


def test_split_energy_k1_returns_same_params():
    params = ArpesParams(ne=71, dataset="arpes")
    chunks = split_energy(params, 1)
    assert chunks == [params]


def test_split_energy_datasets_unique():
    params = ArpesParams(ne=71, dataset="arpes")
    chunks = split_energy(params, 5)
    names = [c.dataset for c in chunks]
    assert len(names) == len(set(names))
    assert all(n.startswith("arpes_e") for n in names)


def test_split_energy_clips_k_for_small_ne():
    # ne=8 with k=5 would need chunks < 2 pts -> clipped to k=4 (2 pts each)
    params = ArpesParams(ne=8, dataset="arpes")
    chunks = split_energy(params, 5)
    assert len(chunks) == 4
    assert all(c.ne == 2 for c in chunks)


# ---------------------------------------------------------------------------
# fanout: split_hv
# ---------------------------------------------------------------------------


def test_split_hv():
    params = ArpesParams(hv_eV=21.2, dataset="arpes")
    hv_list = [21.2, 40.0, 100.5]
    jobs = split_hv(params, hv_list)
    assert len(jobs) == 3
    assert [j.hv_eV for j in jobs] == hv_list
    names = [j.dataset for j in jobs]
    assert len(names) == len(set(names))
    assert names[0] == "arpes_hv21p2"
    assert names[1] == "arpes_hv40p0"
    # unaffected fields stay put
    assert jobs[0].ne == params.ne


# ---------------------------------------------------------------------------
# fanout: plan_jobs
# ---------------------------------------------------------------------------


def test_plan_jobs_mpi_mode():
    params = ArpesParams(ne=71, dataset="arpes")
    plan = plan_jobs(params, nproc_total=8, mode="mpi")
    assert isinstance(plan, FanoutPlan)
    assert plan.mode == "mpi"
    assert len(plan.jobs) == 1
    job_params, subdir, nproc = plan.jobs[0]
    assert job_params is params
    assert subdir == "arpes"
    assert nproc == 8


def test_plan_jobs_chunks_mode():
    params = ArpesParams(ne=71, dataset="arpes")
    plan = plan_jobs(params, nproc_total=4, mode="chunks")
    assert plan.mode == "chunks"
    assert len(plan.jobs) == 4
    for job_params, subdir, nproc in plan.jobs:
        assert nproc == 1
        assert subdir == job_params.dataset
    total_ne = sum(job_params.ne for job_params, _, _ in plan.jobs)
    assert total_ne == 71


def test_plan_jobs_auto_mode():
    params = ArpesParams(ne=71, dataset="arpes")
    plan_yes = plan_jobs(params, nproc_total=4, mode="auto", mpi_available=True)
    assert plan_yes.mode == "mpi"
    assert len(plan_yes.jobs) == 1

    plan_no = plan_jobs(params, nproc_total=4, mode="auto", mpi_available=False)
    assert plan_no.mode == "chunks"
    assert len(plan_no.jobs) == 4


# ---------------------------------------------------------------------------
# vault
# ---------------------------------------------------------------------------


def _make_cu_structure():
    from pymatgen.core import Lattice, Structure

    lat = Lattice.cubic(3.615)
    coords = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    s = Structure(lat, ["Cu"] * 4, coords)
    return s.get_primitive_structure()


@pytest.mark.skipif(not PYMATGEN, reason="pymatgen not installed")
def test_pot_key_stable_across_equivalent_structures():
    s1 = _make_cu_structure()
    s2 = _make_cu_structure()
    scf = ScfParams()
    assert pot_key(s1, scf) == pot_key(s2, scf)


@pytest.mark.skipif(not PYMATGEN, reason="pymatgen not installed")
def test_pot_key_changes_with_scf_params():
    s = _make_cu_structure()
    k1 = pot_key(s, ScfParams(nl=3))
    k2 = pot_key(s, ScfParams(nl=4))
    assert k1 != k2


@pytest.mark.skipif(not PYMATGEN, reason="pymatgen not installed")
def test_vault_register_lookup_roundtrip(tmp_path):
    s = _make_cu_structure()
    scf = ScfParams()
    key = pot_key(s, scf)

    pot_file = tmp_path / "source" / "Cu_SCF.pot"
    pot_file.parent.mkdir()
    pot_file.write_text("fake potential contents\n")

    vault = Vault(tmp_path / "vault")
    entry = vault.register(key, "Cu_fcc", pot_path=str(pot_file), meta={"a": 3.615})

    assert isinstance(entry, VaultEntry)
    assert entry.key == key
    assert entry.pot_path is not None
    # pot was copied into the vault, not referenced in place
    assert vault.root in Path(entry.pot_path).parents
    assert Path(entry.pot_path).read_text() == "fake potential contents\n"

    by_key = vault.lookup(key)
    assert by_key is not None
    assert by_key.name == "Cu_fcc"

    by_name = vault.get("Cu_fcc")
    assert by_name is not None
    assert by_name.key == key
    assert by_name.meta == {"a": 3.615}

    listed = vault.list()
    assert len(listed) == 1
    assert listed[0].name == "Cu_fcc"

    assert vault.lookup("does-not-exist") is None
    assert vault.get("does-not-exist") is None

    assert vault.remove("Cu_fcc") is True
    assert vault.list() == []
    assert vault.remove("Cu_fcc") is False


def test_vault_register_remote_entry(tmp_path):
    vault = Vault(tmp_path / "vault")
    entry = vault.register(
        "deadbeef00000001",
        "VTe2_remote",
        cluster="remote-gpu",
        remote_path="/scratch/user/VTe2/VTe2_SCF.pot",
    )
    assert entry.pot_path is None
    assert entry.cluster == "remote-gpu"
    got = vault.get("VTe2_remote")
    assert got.remote_path == "/scratch/user/VTe2/VTe2_SCF.pot"
