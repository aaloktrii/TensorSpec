"""Gate C / P4 tests: workflow.py orchestration + KKRWrapper.run_simulation.

No real kkrscf/kkrspec run (FakeLauncher fakes those); ase2sprkkr + pymatgen
ARE used for real to generate .inp/.pot text (fast, no binary).
"""
import importlib.util
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

ASE2SPRKKR = importlib.util.find_spec("ase2sprkkr") is not None
PYMATGEN = importlib.util.find_spec("pymatgen") is not None

pytestmark = pytest.mark.skipif(
    not (ASE2SPRKKR and PYMATGEN), reason="ase2sprkkr and/or pymatgen not installed"
)

from tensorspec.core.dft.sprkkr.fanout import split_energy
from tensorspec.core.dft.sprkkr.geometry import parse_pot_geometry
from tensorspec.core.dft.sprkkr.inputs import build_scf_inputs, read_inp_keywords
from tensorspec.core.dft.sprkkr.params import ArpesParams, ScfParams
from tensorspec.core.dft.sprkkr.pointwise import angle_points
from tensorspec.core.dft.sprkkr.workflow import (
    ArpesRunHandle,
    resolve_surface_geometry,
    run_arpes,
    run_scf,
)
from tensorspec.core.dft.sprkkr import read_points_json
from tensorspec.core.arpes.one_step.kkr_wrapper import KKRWrapper, _build_arpes_params

FIXTURES = Path(__file__).parent / "fixtures"
SCF_LOG_TAIL = FIXTURES / "Cu_SCF.out.tail"
ARPES_SPC_FULL = FIXTURES / "_Cu_ARPES_ARPES_data.spc"  # NE=8 NT=9 NP=1
VTE2_POT = FIXTURES / "VTe2_prim.pot"


def _vte2_cif_lattice():
    """VTe2_CDW.cif's own cell (monoclinic, a/b/c/beta below) -- the frame the
    user's h k l means, independent of the primitive cell the .pot was built
    from. Hardcoded here (pymatgen Lattice.from_parameters) so the test does
    not depend on any CIF file existing on disk."""
    from pymatgen.core import Lattice

    return Lattice.from_parameters(
        14.175314511441162, 3.5408678879728384, 9.058026999706719,
        90.0, 110.5531492904914, 90.0,
    ).matrix


def _cu_structure():
    from pymatgen.core import Lattice, Structure

    lat = Lattice.cubic(3.615)
    conv = Structure.from_spacegroup("Fm-3m", lat, ["Cu"], [[0, 0, 0]])
    return conv.get_primitive_structure()


def _touch_bin(bin_dir: Path, *names: str) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    for n in names:
        p = bin_dir / n
        p.write_text("#!/bin/bash\n")
        p.chmod(0o755)


def make_fake_spc(params: ArpesParams, ef_ry: float = 0.5) -> str:
    """Synthesize a minimal-but-valid .spc text matching params' grid."""
    energy = np.linspace(params.e_min_eV, params.e_max_eV, params.ne)[::-1]
    theta = np.linspace(params.theta_e[0], params.theta_e[1], params.nt)
    lines = [
        "KEYWORD   ARPES",
        f"NE          {params.ne}",
        f"EFERMI      {ef_ry:.5f}",
        f"NT          {params.nt}",
        f"NP          {params.np_}",
        "#######################",
    ]
    for e in energy:
        for _p in range(params.np_):
            for t in theta:
                row = " ".join(f"{v:.5E}" for v in (t, e, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0))
                lines.append("    " + row)
    return "\n".join(lines) + "\n"


class FakeHandle:
    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


class FakeLauncher:
    """Mirrors LocalLauncher's shape (.bin_dir, .launch) but fakes output."""

    def __init__(self, bin_dir, scf_log_fixture=None, arpes_spc_fixture=None, spc_by_dataset=None):
        self.bin_dir = Path(bin_dir)
        self.scf_log_fixture = scf_log_fixture
        self.arpes_spc_fixture = arpes_spc_fixture
        self.spc_by_dataset = spc_by_dataset or {}
        self.launched = []

    def launch(self, job):
        self.launched.append(job)
        workdir = Path(job.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        dataset = Path(job.inp_name).stem

        if job.kind == "scf":
            shutil.copy(self.scf_log_fixture, workdir / job.log_name)
            (workdir / f"{dataset}.pot_new").write_text("fake converged pot\n")
        else:
            (workdir / job.log_name).write_text("kkrspec9.7 fake run finished\n")
            dest = workdir / f"{dataset}_ARPES_data.spc"
            if dataset in self.spc_by_dataset:
                dest.write_text(make_fake_spc(self.spc_by_dataset[dataset]))
            else:
                shutil.copy(self.arpes_spc_fixture, dest)
        return FakeHandle()


class FakeRemoteHandle:
    """Mirrors RemoteHandle's shape (.is_running, .fetch) but fakes sftp."""

    def __init__(self, launcher, job, dataset):
        self._launcher = launcher
        self.job = job
        self.dataset = dataset

    def is_running(self):
        return False

    def poll(self):
        return 0

    def fetch(self, remote_names, local_dir):
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        fetched = []
        for name in remote_names:
            dest = local_dir / Path(name).name
            if self.job.kind == "scf":
                if name.endswith(".pot_new"):
                    dest.write_text("fake converged pot\n")
                else:
                    shutil.copy(self._launcher.scf_log_fixture, dest)
            else:
                if name.endswith(".spc"):
                    if self.dataset in self._launcher.spc_by_dataset:
                        dest.write_text(make_fake_spc(self._launcher.spc_by_dataset[self.dataset]))
                    else:
                        shutil.copy(self._launcher.arpes_spc_fixture, dest)
                else:
                    dest.write_text("kkrspec9.7 fake remote run finished\n")
            fetched.append(dest)
        return fetched


class FakeRemoteLauncher:
    """Mirrors RemoteLauncher's shape (.cluster, no .bin_dir) but fakes ssh/sftp.

    ``pot_src`` is a real, valid local .pot file (reuse the ``cu_pot`` fixture)
    whose bytes stand in for whatever lives at the cluster-only ``pot_path``
    -- the truncated ``Cu.pot.head`` fixture is header-only and won't parse
    via ase2sprkkr's real potential loader, so it isn't usable here.
    """

    def __init__(self, scf_log_fixture=None, arpes_spc_fixture=None, spc_by_dataset=None, pot_src=None):
        self.cluster = {"host": "fake", "user": "u"}
        self.scf_log_fixture = scf_log_fixture
        self.arpes_spc_fixture = arpes_spc_fixture
        self.spc_by_dataset = spc_by_dataset or {}
        self.pot_src = pot_src
        self.uploads = []  # [(remote_dir, [names])]
        self.launched = []

    def upload(self, local_paths, remote_dir):
        self.uploads.append((remote_dir, [Path(p).name for p in local_paths]))

    def download(self, remote_paths, local_dir):
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        fetched = []
        for rp in remote_paths:
            dest = local_dir / Path(rp).name
            shutil.copy(self.pot_src, dest)
            fetched.append(dest)
        return fetched

    def launch(self, job):
        self.launched.append(job)
        dataset = Path(job.inp_name).stem
        return FakeRemoteHandle(self, job, dataset)


@pytest.fixture(scope="module")
def cu_pot(tmp_path_factory):
    """A real (starting) .pot, generated once, reused as ARPES input pot."""
    structure = _cu_structure()
    out_dir = tmp_path_factory.mktemp("scf_seed")
    scf_inputs = build_scf_inputs(structure, ScfParams(dataset="Cu"), out_dir)
    return scf_inputs.pot_path


# ---------------------------------------------------------------------------
# run_scf
# ---------------------------------------------------------------------------


class TestRunScf:
    def test_converged_and_pot_exists(self, tmp_path):
        bin_dir = tmp_path / "bin"
        _touch_bin(bin_dir, "kkrscf9.7")
        launcher = FakeLauncher(bin_dir=bin_dir, scf_log_fixture=SCF_LOG_TAIL)

        structure = _cu_structure()
        params = ScfParams(dataset="scf")
        result = run_scf(structure, params, tmp_path / "scf_run", launcher, nproc=1)

        assert result.status.converged is True
        assert Path(result.pot_path).exists()
        assert Path(result.pot_path).name == "scf.pot_new"
        assert result.wall_s >= 0.0


# ---------------------------------------------------------------------------
# run_arpes
# ---------------------------------------------------------------------------


class TestRunArpes:
    def test_mpi_mode_wait(self, tmp_path, cu_pot):
        bin_dir = tmp_path / "bin"
        _touch_bin(bin_dir, "kkrspec9.7", "kkrspec9.7MPI")
        launcher = FakeLauncher(bin_dir=bin_dir, arpes_spc_fixture=ARPES_SPC_FULL)

        params = ArpesParams(ne=8, nt=9, np_=1, dataset="_Cu_ARPES")
        result = run_arpes(
            str(cu_pot), params, tmp_path / "arpes_run", launcher, nproc=2, mode="mpi"
        )

        assert len(launcher.launched) == 1
        assert result.tensor.value.shape == (8, 9)  # phi squeezed, NP=1
        assert "/raw" in result.tree.groups
        assert "/processed" in result.tree.groups
        raw_ds = result.tree["raw"].to_dataset() if hasattr(result.tree["raw"], "to_dataset") else result.tree["raw"].ds
        assert raw_ds["data"].shape == (8, 9)

    def test_chunks_mode_stitches(self, tmp_path, cu_pot):
        bin_dir = tmp_path / "bin"
        _touch_bin(bin_dir, "kkrspec9.7")

        params = ArpesParams(ne=8, nt=9, np_=1, theta_e=(-20.0, 20.0), dataset="_ChunkTest")
        expected_chunks = split_energy(params, 2)
        assert len(expected_chunks) == 2
        spc_by_dataset = {c.dataset: c for c in expected_chunks}

        launcher = FakeLauncher(bin_dir=bin_dir, spc_by_dataset=spc_by_dataset)

        result = run_arpes(
            str(cu_pot), params, tmp_path / "arpes_chunks", launcher, nproc=2, mode="chunks"
        )

        assert len(launcher.launched) == 2
        full = np.linspace(params.e_min_eV, params.e_max_eV, params.ne)
        assert result.dataset.sizes["energy"] == params.ne
        assert np.allclose(sorted(result.dataset["energy"].values), sorted(full), atol=1e-6)
        assert result.tensor.value.shape == (8, 9)

    def test_wait_false_returns_handle(self, tmp_path, cu_pot):
        bin_dir = tmp_path / "bin"
        _touch_bin(bin_dir, "kkrspec9.7")
        launcher = FakeLauncher(bin_dir=bin_dir, arpes_spc_fixture=ARPES_SPC_FULL)

        params = ArpesParams(ne=8, nt=9, np_=1, dataset="_Cu_ARPES2")
        handle = run_arpes(
            str(cu_pot), params, tmp_path / "arpes_async", launcher, nproc=1,
            mode="mpi", wait=False,
        )
        assert isinstance(handle, ArpesRunHandle)
        assert handle.is_done() is True  # FakeHandle.poll() -> 0 immediately
        result = handle.collect()
        assert result.tensor.value.shape == (8, 9)


# ---------------------------------------------------------------------------
# resolve_surface_geometry (core, shared by GUI B3 + CLI e2e -- design doc §0)
# ---------------------------------------------------------------------------


class TestResolveSurfaceGeometry:
    def test_vte2_cif_frame_converts_hkl_and_picks_te_site(self):
        geom = resolve_surface_geometry(VTE2_POT, (-2, 0, 1), cif_lattice_matrix=_vte2_cif_lattice())
        assert geom.hkl_abas == (0, 1, -1)
        pot_geom = parse_pot_geometry(VTE2_POT)
        site = next(s for s in pot_geom.sites if s[0] == geom.iq_at_surf)
        assert site[2].startswith("Te")
        assert geom.atoms_per_plane_max >= 1
        assert "plane 0" in geom.planes_summary

    def test_iq_at_surf_pin_overrides_auto_pick(self):
        geom = resolve_surface_geometry(
            VTE2_POT, (-2, 0, 1), cif_lattice_matrix=_vte2_cif_lattice(), iq_at_surf=3
        )
        assert geom.iq_at_surf == 3

    def test_no_cif_lattice_assumes_hkl_already_abas(self):
        geom = resolve_surface_geometry(VTE2_POT, (0, 1, -1))
        assert geom.hkl_abas == (0, 1, -1)


class TestRunArpesCifFrame:
    def test_hkl_frame_cif_rewrites_inp_with_crys_vecs_and_int_iq(self, tmp_path):
        bin_dir = tmp_path / "bin"
        _touch_bin(bin_dir, "kkrspec9.7")
        launcher = FakeLauncher(bin_dir=bin_dir, arpes_spc_fixture=ARPES_SPC_FULL)

        params = ArpesParams(
            ne=8, nt=9, np_=1, hkl=(-2, 0, 1), hkl_frame="cif", iq_at_surf=None,
            dataset="_VTe2_cif",
        )
        result = run_arpes(
            str(VTE2_POT), params, tmp_path / "arpes_cif", launcher, nproc=1, mode="mpi",
            cif_lattice=_vte2_cif_lattice(),
        )

        assert result.geometry is not None
        assert result.geometry.hkl_abas == (0, 1, -1)
        assert result.params.hkl == (0, 1, -1)
        assert result.params.hkl_frame == "abas"
        assert isinstance(result.params.iq_at_surf, int)

        inp_path = tmp_path / "arpes_cif" / params.dataset / f"{params.dataset}.inp"
        sections = read_inp_keywords(inp_path)
        assert sections["TASK"]["MILLER_HKL"] == "{0,1,-1}"
        assert sections["TASK"]["CRYS_VECS"] == ""
        assert sections["TASK"]["IQ_AT_SURF"] == str(result.geometry.iq_at_surf)

    def test_hkl_frame_cif_without_cif_lattice_raises(self, tmp_path):
        bin_dir = tmp_path / "bin"
        _touch_bin(bin_dir, "kkrspec9.7")
        launcher = FakeLauncher(bin_dir=bin_dir, arpes_spc_fixture=ARPES_SPC_FULL)

        params = ArpesParams(ne=8, nt=9, np_=1, hkl=(-2, 0, 1), hkl_frame="cif", dataset="_VTe2_nolat")
        with pytest.raises(ValueError, match="cif_lattice"):
            run_arpes(str(VTE2_POT), params, tmp_path / "arpes_nolat", launcher, nproc=1, mode="mpi")


# ---------------------------------------------------------------------------
# Remote workdir mapping (local staging dir != cluster run dir)
# ---------------------------------------------------------------------------


def _potfil_value(inp_path) -> str:
    kw = read_inp_keywords(inp_path)
    for section in kw.values():
        if "POTFIL" in section:
            return section["POTFIL"]
    raise AssertionError(f"no POTFIL line in {inp_path}")


class TestRemoteWorkdir:
    def test_run_scf_remote_uploads_and_fetches(self, tmp_path):
        launcher = FakeRemoteLauncher(scf_log_fixture=SCF_LOG_TAIL)
        structure = _cu_structure()
        params = ScfParams(dataset="scf")
        local_workdir = tmp_path / "scf_local"
        remote_workdir = "/cluster/scratch/scf_job1"

        result = run_scf(
            structure, params, local_workdir, launcher, nproc=1,
            remote_workdir=remote_workdir,
        )

        assert launcher.launched[0].workdir == remote_workdir
        assert launcher.uploads[0][0] == remote_workdir
        assert "scf.inp" in launcher.uploads[0][1]
        assert Path(result.pot_path) == local_workdir / "scf.pot_new"
        assert Path(result.pot_path).exists()

    def test_run_scf_remote_requires_remote_workdir(self, tmp_path):
        launcher = FakeRemoteLauncher(scf_log_fixture=SCF_LOG_TAIL)
        structure = _cu_structure()
        params = ScfParams(dataset="scf2")
        with pytest.raises(ValueError):
            run_scf(structure, params, tmp_path / "scf_local2", launcher, nproc=1)

    def test_run_arpes_remote_downloads_pot_and_rewrites_potfil(self, tmp_path, cu_pot):
        launcher = FakeRemoteLauncher(arpes_spc_fixture=ARPES_SPC_FULL, pot_src=str(cu_pot))
        params = ArpesParams(ne=8, nt=9, np_=1, dataset="_Cu_ARPESRemote")
        local_workdir = tmp_path / "arpes_local"
        remote_workdir = "/cluster/scratch/arpes_job1"
        remote_pot = "/cluster/scratch/scf_job1/scf.pot_new"  # cluster-only, not local

        result = run_arpes(
            remote_pot, params, local_workdir, launcher, nproc=1, mode="mpi",
            remote_workdir=remote_workdir,
        )

        # downloaded a local copy under the local workdir
        downloaded = local_workdir / "scf.pot_new"
        assert downloaded.exists()

        # uploaded into the remote subdir, inp + the downloaded pot (by basename)
        remote_dir, names = launcher.uploads[0]
        assert remote_dir == f"{remote_workdir}/{params.dataset}"
        assert f"{params.dataset}.inp" in names
        assert "scf.pot_new" in names

        # POTFIL in the uploaded .inp is rewritten to the pot's bare filename
        inp_path = local_workdir / params.dataset / f"{params.dataset}.inp"
        assert _potfil_value(inp_path) == "scf.pot_new"

        assert launcher.launched[0].workdir == remote_dir
        assert result.tensor.value.shape == (8, 9)

    def test_run_arpes_remote_requires_remote_workdir(self, tmp_path, cu_pot):
        launcher = FakeRemoteLauncher(arpes_spc_fixture=ARPES_SPC_FULL, pot_src=str(cu_pot))
        params = ArpesParams(ne=8, nt=9, np_=1, dataset="_Cu_ARPESRemote2")
        with pytest.raises(ValueError):
            run_arpes(
                "/cluster/scratch/scf_job1/scf.pot_new", params, tmp_path / "arpes_local2",
                launcher, nproc=1, mode="mpi",
            )


# ---------------------------------------------------------------------------
# KKRWrapper.run_simulation kwargs mapping
# ---------------------------------------------------------------------------


class TestKwargsMapping:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("p-pol", "P"),
            ("s-pol", "S"),
            ("CR", "C+"),
            ("CL", "C-"),
            ("Linear Horizontal (p-pol)", "P"),
            ("Linear Vertical (s-pol)", "S"),
            ("Circular Right (CR)", "C+"),
            ("Circular Left (CL)", "C-"),
            ("Linear Arbitrary", "P"),
            ("P", "P"),
            ("C+", "C+"),
        ],
    )
    def test_polarization_mapping(self, text, expected):
        params = _build_arpes_params({"polarization": text})
        assert params.pol_p == expected

    def test_k_bounds_mapping(self):
        params = _build_arpes_params(
            {"k_bounds": {"X": [-10.0, 10.0, 21], "Y": [0.0, 5.0, 3]}}
        )
        assert params.theta_e == (-10.0, 10.0)
        assert params.nt == 21
        assert params.phi_e == (0.0, 5.0)
        assert params.np_ == 3

    def test_energy_mapping_variants(self):
        p1 = _build_arpes_params({"e_range": [-3.0, 2.0, 11]})
        assert (p1.e_min_eV, p1.e_max_eV, p1.ne) == (-3.0, 2.0, 11)

        p2 = _build_arpes_params({"e_min": -1.0, "e_max": 1.0, "e_steps": 5})
        assert (p2.e_min_eV, p2.e_max_eV, p2.ne) == (-1.0, 1.0, 5)

        p3 = _build_arpes_params({"k_bounds": {"E": [-4.0, 0.0, 9]}})
        assert (p3.e_min_eV, p3.e_max_eV, p3.ne) == (-4.0, 0.0, 9)

    def test_direct_field_passthrough(self):
        params = _build_arpes_params({"n_layer": 30, "nlat_g_vec": 19, "nl": 4})
        assert params.n_layer == 30
        assert params.nlat_g_vec == 19
        assert params.nl == 4

    def test_photon_energy_and_work_function_and_hkl(self):
        params = _build_arpes_params(
            {"photon_energy": 55.0, "work_function": 4.8, "hkl": (1, 1, 1)}
        )
        assert params.hv_eV == pytest.approx(55.0)
        assert params.ework_eV == pytest.approx(4.8)
        assert params.hkl == (1, 1, 1)


# ---------------------------------------------------------------------------
# KKRWrapper.run_simulation end-to-end (FakeLauncher)
# ---------------------------------------------------------------------------


class TestKKRWrapper:
    def test_missing_pot_raises(self):
        wrapper = KKRWrapper()
        with pytest.raises(ValueError, match="converged potential"):
            wrapper.run_simulation({}, {})

    def test_returns_intensity_broadened_shape(self, tmp_path, cu_pot):
        bin_dir = tmp_path / "bin"
        _touch_bin(bin_dir, "kkrspec9.7", "kkrspec9.7MPI")
        launcher = FakeLauncher(bin_dir=bin_dir, arpes_spc_fixture=ARPES_SPC_FULL)

        kwargs = {
            "pot_path": str(cu_pot),
            "k_bounds": {"X": [-20.0, 20.0, 9], "Y": [0.0, 0.0, 1]},
            "e_range": [-6.0, 1.0, 8],
            "launcher": launcher,
            "workdir": str(tmp_path / "kkr_run"),
            "nproc": 2,
            "mode": "mpi",
        }
        wrapper = KKRWrapper()
        out = wrapper.run_simulation({}, kwargs)

        assert out["intensity_broadened"].shape == (9, 1, 8)  # nt, np, ne
        assert out["tensor"] is not None
        assert out["datatree"] is not None
        assert isinstance(out["dataset"], xr.Dataset)
        assert out["params"].nt == 9

    def test_async_returns_handle_then_collect(self, tmp_path, cu_pot):
        bin_dir = tmp_path / "bin"
        _touch_bin(bin_dir, "kkrspec9.7")
        launcher = FakeLauncher(bin_dir=bin_dir, arpes_spc_fixture=ARPES_SPC_FULL)

        kwargs = {
            "pot_path": str(cu_pot),
            "k_bounds": {"X": [-20.0, 20.0, 9], "Y": [0.0, 0.0, 1]},
            "e_range": [-6.0, 1.0, 8],
            "launcher": launcher,
            "workdir": str(tmp_path / "kkr_run_async"),
            "nproc": 1,
            "mode": "mpi",
            "async_": True,
        }
        wrapper = KKRWrapper()
        out = wrapper.run_simulation({}, kwargs)

        assert "handle" in out
        result = out["handle"].collect()
        assert result.tensor.value.shape == (8, 9)


# ---------------------------------------------------------------------------
# Smoke: real Cu ARPES via LocalLauncher (Gate C says report wall time).
# ---------------------------------------------------------------------------

SMOKE_BIN_DIR = Path("/home/claude/sprkkr_bench/bin")
SMOKE_POT = Path("/home/claude/sprkkr_bench/cu_run/scf/Cu.pot_new")


@pytest.mark.skipif(
    not (SMOKE_BIN_DIR.exists() and SMOKE_POT.exists() and os.environ.get("SPRKKR_SMOKE") == "1"),
    reason="real kkrspec9.7 binary + SPRKKR_SMOKE=1 required",
)
def test_smoke_real_run_arpes(tmp_path):
    from tensorspec.core.dft.sprkkr.jobs import LocalLauncher

    params = ArpesParams(ne=2, nt=3, np_=1, dataset="_SmokeWorkflowARPES")
    launcher = LocalLauncher(bin_dir=SMOKE_BIN_DIR)

    start = time.time()
    result = run_arpes(str(SMOKE_POT), params, tmp_path / "smoke_arpes", launcher, nproc=1, mode="mpi")
    wall_s = time.time() - start
    print(f"\nSPRKKR_SMOKE workflow.run_arpes wall time: {wall_s:.1f}s")

    assert result.tensor.value.shape == (2, 3)


# ---------------------------------------------------------------------------
# Pointwise deflector cut (single-point jobs per lab angle)
# ---------------------------------------------------------------------------


class TestPointwiseFanout:
    def test_angle_points_launch_one_job_per_point(self, tmp_path, cu_pot):
        bin_dir = tmp_path / "bin"
        _touch_bin(bin_dir, "kkrspec9.7")
        
        # Create angle points: 5 points
        pts = angle_points(
            (-15, 15), 5, hv_eV=84, work_function_eV=4.5, deflector_deg=-10.3
        )
        assert len(pts) == 5
        
        # Build spc_by_dataset with single-point params for each
        base_params = ArpesParams(ne=8, nt=1, np_=1, dataset="_Pointwise")
        spc_by_dataset = {}
        for p in pts:
            dataset_name = f"{base_params.dataset}_p{p.index:04d}"
            params = ArpesParams(
                ne=base_params.ne, nt=1, np_=1,
                theta_e=(p.theta_e_deg, p.theta_e_deg),
                phi_e=(p.phi_e_deg, p.phi_e_deg),
                dataset=dataset_name
            )
            spc_by_dataset[dataset_name] = params
        
        launcher = FakeLauncher(bin_dir=bin_dir, spc_by_dataset=spc_by_dataset)
        
        params = ArpesParams(ne=8, nt=5, np_=1, theta_e=(-15.0, 15.0), dataset="_Pointwise")
        result = run_arpes(
            str(cu_pot), params, tmp_path / "pointwise_run", launcher,
            nproc=1, angle_points=pts, deflector_deg=-10.3
        )
        
        assert len(launcher.launched) == 5
        assert result.dataset is not None

    def test_sub_params_are_single_point(self, tmp_path, cu_pot):
        bin_dir = tmp_path / "bin"
        _touch_bin(bin_dir, "kkrspec9.7")

        pts = angle_points(
            (-15, 15), 5, hv_eV=84, work_function_eV=4.5, deflector_deg=-10.3
        )

        base_params = ArpesParams(ne=8, nt=1, np_=1, dataset="_PointSingle")
        spc_by_dataset = {}
        for p in pts:
            dataset_name = f"{base_params.dataset}_p{p.index:04d}"
            params = ArpesParams(
                ne=base_params.ne, nt=1, np_=1,
                theta_e=(p.theta_e_deg, p.theta_e_deg),
                phi_e=(p.phi_e_deg, p.phi_e_deg),
                dataset=dataset_name
            )
            spc_by_dataset[dataset_name] = params

        launcher = FakeLauncher(bin_dir=bin_dir, spc_by_dataset=spc_by_dataset)

        params = ArpesParams(ne=8, nt=5, np_=1, theta_e=(-15.0, 15.0), dataset="_PointSingle")
        result = run_arpes(
            str(cu_pot), params, tmp_path / "pointwise_run2", launcher,
            nproc=1, angle_points=pts, deflector_deg=-10.3
        )

        # Check each launched job's .inp file has NT=1, NP=1
        for job in launcher.launched:
            inp_path = Path(job.workdir) / job.inp_name
            kw = read_inp_keywords(inp_path)
            assert kw["SPEC_EL"]["NT"] == "1"
            assert kw["SPEC_EL"]["NP"] == "1"

    def test_sub_dataset_names_unique(self, tmp_path, cu_pot):
        bin_dir = tmp_path / "bin"
        _touch_bin(bin_dir, "kkrspec9.7")
        
        pts = angle_points(
            (-15, 15), 5, hv_eV=84, work_function_eV=4.5, deflector_deg=-10.3
        )
        
        base_params = ArpesParams(ne=8, nt=1, np_=1, dataset="_PointUnique")
        spc_by_dataset = {}
        for p in pts:
            dataset_name = f"{base_params.dataset}_p{p.index:04d}"
            params = ArpesParams(
                ne=base_params.ne, nt=1, np_=1,
                theta_e=(p.theta_e_deg, p.theta_e_deg),
                phi_e=(p.phi_e_deg, p.phi_e_deg),
                dataset=dataset_name
            )
            spc_by_dataset[dataset_name] = params
        
        launcher = FakeLauncher(bin_dir=bin_dir, spc_by_dataset=spc_by_dataset)
        
        params = ArpesParams(ne=8, nt=5, np_=1, theta_e=(-15.0, 15.0), dataset="_PointUnique")
        result = run_arpes(
            str(cu_pot), params, tmp_path / "pointwise_run3", launcher,
            nproc=1, angle_points=pts, deflector_deg=-10.3
        )
        
        # Extract dataset names from launched jobs
        dataset_names = [Path(job.inp_name).stem for job in launcher.launched]
        assert len(dataset_names) == len(set(dataset_names))
        assert all("_p" in name for name in dataset_names)

    def test_pointwise_sidecar_written(self, tmp_path, cu_pot):
        bin_dir = tmp_path / "bin"
        _touch_bin(bin_dir, "kkrspec9.7")
        
        pts = angle_points(
            (-15, 15), 5, hv_eV=84, work_function_eV=4.5, deflector_deg=-10.3
        )
        
        base_params = ArpesParams(ne=8, nt=1, np_=1, dataset="_PointSidecar")
        spc_by_dataset = {}
        for p in pts:
            dataset_name = f"{base_params.dataset}_p{p.index:04d}"
            params = ArpesParams(
                ne=base_params.ne, nt=1, np_=1,
                theta_e=(p.theta_e_deg, p.theta_e_deg),
                phi_e=(p.phi_e_deg, p.phi_e_deg),
                dataset=dataset_name
            )
            spc_by_dataset[dataset_name] = params
        
        launcher = FakeLauncher(bin_dir=bin_dir, spc_by_dataset=spc_by_dataset)
        
        workdir = tmp_path / "pointwise_run4"
        params = ArpesParams(ne=8, nt=5, np_=1, theta_e=(-15.0, 15.0), dataset="_PointSidecar")
        result = run_arpes(
            str(cu_pot), params, workdir, launcher,
            nproc=1, angle_points=pts, deflector_deg=-10.3
        )
        
        sidecar_path = workdir / "pointwise_points.json"
        assert sidecar_path.exists()
        
        read_pts, meta = read_points_json(str(sidecar_path))
        assert len(read_pts) == 5
        assert meta.get("deflector_deg") == -10.3

    def test_pointwise_result_theta_axis_is_lab_slit_angle(self, tmp_path, cu_pot):
        bin_dir = tmp_path / "bin"
        _touch_bin(bin_dir, "kkrspec9.7")
        
        pts = angle_points(
            (-15, 15), 5, hv_eV=84, work_function_eV=4.5, deflector_deg=-10.3
        )
        
        base_params = ArpesParams(ne=8, nt=1, np_=1, dataset="_PointAxis")
        spc_by_dataset = {}
        for p in pts:
            dataset_name = f"{base_params.dataset}_p{p.index:04d}"
            params = ArpesParams(
                ne=base_params.ne, nt=1, np_=1,
                theta_e=(p.theta_e_deg, p.theta_e_deg),
                phi_e=(p.phi_e_deg, p.phi_e_deg),
                dataset=dataset_name
            )
            spc_by_dataset[dataset_name] = params
        
        launcher = FakeLauncher(bin_dir=bin_dir, spc_by_dataset=spc_by_dataset)
        
        params = ArpesParams(ne=8, nt=5, np_=1, theta_e=(-15.0, 15.0), dataset="_PointAxis")
        result = run_arpes(
            str(cu_pot), params, tmp_path / "pointwise_run5", launcher,
            nproc=1, angle_points=pts, deflector_deg=-10.3
        )
        
        # Check that theta axis matches lab slit angles
        lab_slit_angles = [p.slit_deg for p in pts]
        result_theta = result.dataset["theta"].values
        assert np.allclose(sorted(result_theta), sorted(lab_slit_angles), atol=1e-6)
        
        # Check that phi axis is the deflector value
        result_phi = result.dataset["phi"].values
        assert len(result_phi) == 1
        assert np.isclose(result_phi[0], -10.3, atol=1e-6)


# ---------------------------------------------------------------------------
# KKRWrapper pointwise mode kwargs mapping (kkr_wrapper.py)
# ---------------------------------------------------------------------------


class TestPointwiseKwargs:
    def test_pointwise_off_by_default(self):
        from tensorspec.core.arpes.one_step.kkr_wrapper import _build_angle_points

        params = ArpesParams()
        points = _build_angle_points({}, params)
        assert points == []

    def test_deflector_falls_back_to_phi_e(self):
        from tensorspec.core.arpes.one_step.kkr_wrapper import _map_deflector

        params = ArpesParams(phi_e=(5.0, 10.0))
        deflector = _map_deflector({}, params)
        assert deflector == 5.0

    def test_deflector_angle_key_wins(self):
        from tensorspec.core.arpes.one_step.kkr_wrapper import _map_deflector

        params = ArpesParams(phi_e=(5.0, 10.0))
        deflector = _map_deflector({"deflector_angle": -10.3}, params)
        assert deflector == -10.3

    def test_wrapper_pointwise_returns_lab_theta_axis(self, tmp_path, cu_pot):
        bin_dir = tmp_path / "bin"
        _touch_bin(bin_dir, "kkrspec9.7")

        # Create angle points: 5 points
        pts = angle_points(
            (-15, 15), 5, hv_eV=84, work_function_eV=4.5, deflector_deg=-10.3
        )

        # Build spc_by_dataset with single-point params for each
        base_params = ArpesParams(ne=8, nt=1, np_=1, dataset="_KWrapperPW")
        spc_by_dataset = {}
        for p in pts:
            dataset_name = f"{base_params.dataset}_p{p.index:04d}"
            params = ArpesParams(
                ne=base_params.ne, nt=1, np_=1,
                theta_e=(p.theta_e_deg, p.theta_e_deg),
                phi_e=(p.phi_e_deg, p.phi_e_deg),
                dataset=dataset_name
            )
            spc_by_dataset[dataset_name] = params

        launcher = FakeLauncher(
            bin_dir=bin_dir, arpes_spc_fixture=ARPES_SPC_FULL, spc_by_dataset=spc_by_dataset
        )

        # Run through KKRWrapper with pointwise=True
        wrapper = KKRWrapper()
        kwargs = {
            "pot_path": str(cu_pot),
            "k_bounds": {"X": [-15.0, 15.0, 5], "Y": [0.0, 0.0, 1]},
            "e_range": [-6.0, 1.0, 8],
            "launcher": launcher,
            "workdir": str(tmp_path / "kkr_pointwise_run"),
            "nproc": 1,
            "mode": "mpi",
            "pointwise": True,
            "slit_angle": 0.0,
            "deflector_angle": -10.3,
            "manip_theta": 0.0,
            "manip_azimuth": 0.0,
            "manip_tilt": 0.0,
            "phi_offset_deg": 0.0,
            "ref_energy_eV": 0.0,
            "dataset": "_KWrapperPW",
        }
        out = wrapper.run_simulation({}, kwargs)

        # Check theta axis is lab slit angles
        lab_slit_angles = [p.slit_deg for p in pts]
        assert np.allclose(sorted(out["theta"]), sorted(lab_slit_angles), atol=1e-6)

        # Check phi axis is the deflector value
        assert len(out["phi"]) == 1
        assert np.isclose(out["phi"][0], -10.3, atol=1e-6)

        # Check angle_points is in the output
        assert "angle_points" in out
        assert len(out["angle_points"]) == 5
