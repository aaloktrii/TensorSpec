#!/usr/bin/env python3
"""Remote tight-binding band diagonalization (runs on cluster, writes tb_bands_result.npz)."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np


def _solve_bands(engine, k_vecs, job: dict):
    """Call solve_bands; fail clearly if cluster TensorSpec lacks Grizzly API."""
    kwargs = {
        "custom_hopping": job.get("custom_hopping") or {},
        "onsite_e": float(job.get("onsite_e", 0.0)),
        "use_soc": bool(job.get("use_soc", False)),
        "soc_strength": float(job.get("soc_strength", 0.5)),
        "w90_filepath": job.get("_w90_path"),
        "cutoffs": job.get("cutoffs"),
        "tb_mode": job.get("tb_mode", "Simple Scalar"),
        "orbital_shifts": job.get("orbital_shifts"),
        "need_eigenvectors": bool(job.get("need_eigenvectors", True)),
        "diag_engine": str(job.get("diag_engine", "chinook")),
        "diag_device": str(job.get("diag_device", "cpu")),
        "qe_fermi": float(job.get("fermi_energy", job.get("qe_fermi", 0.0))),
    }
    params = inspect.signature(engine.solve_bands).parameters
    missing = [
        k
        for k in ("need_eigenvectors", "diag_engine", "diag_device")
        if k in kwargs and k not in params
    ]
    want_grizzly = kwargs.get("diag_engine") == "grizzly"
    if want_grizzly and missing:
        raise RuntimeError(
            "Remote TensorSpec is outdated (no Grizzly band diag). "
            "On cluster: cd ~/TensorSpec && git fetch && git checkout TensorSpec_GUI "
            "&& git pull origin TensorSpec_GUI"
        )
    filtered = {k: v for k, v in kwargs.items() if k in params}
    return engine.solve_bands(k_vecs, **filtered)


def main() -> int:
    parser = argparse.ArgumentParser(description="TensorSpec remote TB band runner")
    parser.add_argument("--job", default="tb_job.json", help="Job specification JSON")
    parser.add_argument(
        "--out",
        default="tb_bands_result.npz",
        help="Output NPZ (eigenvalues, optional eigenvectors, orb_labels)",
    )
    args = parser.parse_args()

    job_path = Path(args.job)
    if not job_path.is_file():
        print(f"FATAL: missing job file {job_path}", flush=True)
        return 1

    job = json.loads(job_path.read_text())
    run_dir = job_path.parent.resolve()

    from pymatgen.core import Structure

    from tensorspec.core.dft.chinook_tb import ChinookTightBindingEngine

    engine = ChinookTightBindingEngine()
    engine.crystal_structure = Structure.from_dict(job["structure"])

    k_vecs = np.asarray(job["k_vecs"], dtype=float)
    w90_name = job.get("w90_basename")
    w90_path = str(run_dir / w90_name) if w90_name else None
    job["_w90_path"] = w90_path

    use_soc = bool(job.get("use_soc", False))
    onsite_e = float(job.get("onsite_e", 0.0))
    qe_fermi = float(job.get("fermi_energy", job.get("qe_fermi", 0.0)))
    cache_name = job.get("w90_cache_basename")
    if w90_path and cache_name:
        cache_path = run_dir / cache_name
        if cache_path.is_file():
            import pickle

            try:
                with cache_path.open("rb") as f:
                    payload = pickle.load(f)
                source_key = engine._w90_source_key(
                    w90_path, use_soc, onsite_e, qe_fermi=qe_fermi
                )
                engine.seed_w90_parsed(
                    source_key,
                    payload["tb_dict"],
                    payload["basis_args"],
                    payload.get("A_qe"),
                )
                n_hop = len(payload.get("tb_dict", {}).get("list", []))
                print(
                    f"W90 TB cache loaded on server ({n_hop} hops, skip hr.dat parse)",
                    flush=True,
                )
            except Exception as exc:
                print(f"WARN: W90 cache load failed, will parse hr.dat: {exc}", flush=True)

    t0 = time.perf_counter()
    eigenvalues, eigenvectors, orb_labels = _solve_bands(engine, k_vecs, job)
    fermi_energy = qe_fermi
    # W90 parse folds QE EF into H — keep eigenvalues EF-relative (no second subtract).
    # Manual SK jobs leave fermi_energy at 0 unless UI set it without hr.dat.
    if not w90_path and fermi_energy:
        eigenvalues = np.asarray(eigenvalues, dtype=float) - fermi_energy
    else:
        eigenvalues = np.asarray(eigenvalues, dtype=float)

    # Export H_dict *before* writing the result npz. Client polls for the npz and
    # otherwise downloads while w90_tb_cache.pkl is still a half-written .tmp
    # (~2 GiB / 20M hops) — ARPES then sees empty H_dict.
    if w90_path:
        _export_w90_cache_for_client(engine, job, run_dir, w90_path)

    out_path = Path(args.out)
    if eigenvectors is None:
        np.savez(
            out_path,
            eigenvalues=eigenvalues,
            orb_labels=np.array(orb_labels, dtype=object),
            fermi_energy=fermi_energy,
            backend=str(job.get("diag_engine", "chinook")),
        )
    else:
        np.savez(
            out_path,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            orb_labels=np.array(orb_labels, dtype=object),
            fermi_energy=fermi_energy,
            backend=str(job.get("diag_engine", "chinook")),
        )

    elapsed = time.perf_counter() - t0
    print(
        f"OK: {eigenvalues.shape[0]} k-pts x {eigenvalues.shape[1]} bands "
        f"in {elapsed:.2f}s -> {out_path}",
        flush=True,
    )
    return 0


def _write_job_dir_cache_compat(
    out: Path,
    *,
    key: str,
    tb_dict: dict,
    basis_args: dict,
    A_qe,
    source: str,
    hop_tol: float,
) -> None:
    """Write client-keyed cache; inline fallback if cluster tensorspec is stale."""
    try:
        from tensorspec.core.dft.w90_tb_cache import write_job_dir_cache

        write_job_dir_cache(
            out,
            key=key,
            tb_dict=tb_dict,
            basis_args=basis_args,
            A_qe=A_qe,
            source=source,
            hop_tol=hop_tol,
        )
        return
    except ImportError:
        pass
    # Uploaded runner can be newer than the remote host's installed tensorspec.
    import pickle

    payload = {
        "key": str(key),
        "tb_dict": tb_dict,
        "basis_args": basis_args,
        "A_qe": A_qe,
        "hop_tol": float(hop_tol),
        "saved_at": time.time(),
        "source": source or str(out),
    }
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(out)


def _export_w90_cache_for_client(engine, job: dict, run_dir: Path, w90_path: str) -> None:
    """Write w90_tb_cache.pkl keyed by the *client* cache key from the job JSON."""
    import shutil

    try:
        from tensorspec.core.dft.w90_tb_cache import (
            REMOTE_CACHE_NAME,
            cache_key,
            ensure_cache_key,
            local_cache_path,
        )
    except ImportError:
        REMOTE_CACHE_NAME = "w90_tb_cache.pkl"
        cache_key = None
        ensure_cache_key = None
        local_cache_path = None

    client_key = job.get("w90_cache_key")
    if not client_key:
        print(
            "WARN: job missing w90_cache_key — Mac cannot attach H_dict for ARPES",
            flush=True,
        )
        return

    cache_name = job.get("w90_cache_basename") or REMOTE_CACHE_NAME
    out = run_dir / cache_name

    # Fast path: copy cluster ~/.tensorspec_cache/w90_tb/<cluster_key>.pkl and
    # rewrite payload key — avoids re-pickling ~20M hops (~5 min).
    use_soc = bool(job.get("use_soc", False))
    onsite_e = float(job.get("onsite_e", 0.0))
    hop_tol = float(job.get("hop_tol", 1e-6))
    qe_fermi = float(job.get("fermi_energy", job.get("qe_fermi", 0.0)))
    if cache_key is not None and local_cache_path is not None and ensure_cache_key is not None:
        try:
            cluster_src = local_cache_path(
                cache_key(w90_path, use_soc, onsite_e, hop_tol, qe_fermi=qe_fermi)
            )
            if cluster_src.is_file():
                print(
                    f"Copying cluster W90 cache -> {cache_name} "
                    f"({cluster_src.stat().st_size / (1024 * 1024):.1f} MiB)...",
                    flush=True,
                )
                shutil.copy2(cluster_src, out)
                # Skip ensure_cache_key on multi-GB pickles (full reload).
                # Mac load_parsed_tb tolerates path-key mismatch when hops present.
                if out.is_file() and out.stat().st_size > 0:
                    size_mb = out.stat().st_size / (1024 * 1024)
                    print(
                        f"Wrote {cache_name} for client sync (copied, {size_mb:.1f} MiB, "
                        f"key={client_key[:8]}…)",
                        flush=True,
                    )
                    return
                print("WARN: cluster cache copy empty — will re-pickle", flush=True)
        except OSError as exc:
            print(f"WARN: cluster cache copy failed ({exc}) — will re-pickle", flush=True)

    h_dict = getattr(engine, "H_dict", None)
    if not isinstance(h_dict, dict):
        # Fall back to in-memory parse cache after seed/parse.
        for _tb, _basis in getattr(engine, "_w90_parse_cache", {}).values():
            h_dict = _tb
            break
    if not isinstance(h_dict, dict):
        print("WARN: no H_dict on engine — skip ARPES cache export", flush=True)
        return
    n_hop = len(h_dict.get("list") or h_dict.get("H") or [])
    if n_hop == 0:
        print("WARN: H_dict has zero hoppings — skip ARPES cache export", flush=True)
        return

    basis_args = getattr(engine, "_cached_basis_args", None)
    if basis_args is None:
        for _tb, basis_args in getattr(engine, "_w90_parse_cache", {}).values():
            break
    if not isinstance(basis_args, dict):
        print("WARN: no basis_args — skip ARPES cache export", flush=True)
        return

    _write_job_dir_cache_compat(
        out,
        key=str(client_key),
        tb_dict=h_dict,
        basis_args=basis_args,
        A_qe=getattr(engine, "A_qe", None),
        source=w90_path,
        hop_tol=hop_tol,
    )
    size_mb = out.stat().st_size / (1024 * 1024)
    print(
        f"Wrote {cache_name} for client sync ({n_hop} hops, {size_mb:.1f} MiB, "
        f"key={client_key[:8]}…)",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
