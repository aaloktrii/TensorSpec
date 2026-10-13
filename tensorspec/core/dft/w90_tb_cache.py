"""Disk cache for parsed wannier90_hr.dat → Chinook tb_dict (band diag hot path)."""

from __future__ import annotations

import hashlib
import os
import pickle
import time
from pathlib import Path
from typing import Any, Optional, Tuple

CACHE_DIR = Path(os.path.expanduser("~/.tensorspec_cache/w90_tb"))
CACHE_VERSION = 8  # v8: use_ws_distance MDRS hop remap
REMOTE_CACHE_NAME = "w90_tb_cache.pkl"


def cache_key(
    w90_filepath: str,
    use_soc: bool,
    onsite_e: float,
    hop_tol: float = 1e-6,
    qe_fermi: float = 0.0,
) -> str:
    path = os.path.abspath(w90_filepath)
    try:
        mtime = os.path.getmtime(path)
        size = os.path.getsize(path)
    except OSError:
        mtime, size = 0, 0
    # Sibling .win mtime so flipping use_ws_distance / mp_grid busts cache.
    win_mtime = 0.0
    work = os.path.dirname(path)
    for name in (
        "wannier90.win",
        os.path.basename(path).replace("_hr.dat", ".win"),
    ):
        wp = os.path.join(work, name)
        if os.path.isfile(wp):
            try:
                win_mtime = os.path.getmtime(wp)
            except OSError:
                win_mtime = 0.0
            break
    raw = (
        f"v{CACHE_VERSION}|{path}|{mtime}|{size}|win:{win_mtime}|{use_soc}|"
        f"{float(onsite_e):.12g}|{float(hop_tol):.12g}|{float(qe_fermi):.12g}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def local_cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.pkl"


def load_parsed_tb(
    w90_filepath: str,
    use_soc: bool,
    onsite_e: float,
    hop_tol: float = 1e-6,
    qe_fermi: float = 0.0,
) -> Optional[Tuple[dict, dict, Optional[Any]]]:
    """Return (tb_dict, basis_args, A_qe) or None."""
    key = cache_key(w90_filepath, use_soc, onsite_e, hop_tol, qe_fermi=qe_fermi)
    path = local_cache_path(key)
    if not path.is_file():
        return None
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
        tb = payload.get("tb_dict") or {}
        n_hop = len(tb.get("list") or tb.get("H") or [])
        if n_hop == 0:
            return None
        # File path is keyed by Mac (or cluster) cache_key(); Hybrid may copy a
        # cluster-keyed pickle into the Mac path — accept hoppings regardless.
        if payload.get("key") != key:
            payload["key"] = key
        return tb, payload["basis_args"], payload.get("A_qe")
    except Exception:
        return None


def save_parsed_tb(
    w90_filepath: str,
    use_soc: bool,
    onsite_e: float,
    tb_dict: dict,
    basis_args: dict,
    A_qe,
    hop_tol: float = 1e-6,
    qe_fermi: float = 0.0,
) -> Path:
    key = cache_key(w90_filepath, use_soc, onsite_e, hop_tol, qe_fermi=qe_fermi)
    path = local_cache_path(key)
    payload = {
        "key": key,
        "tb_dict": tb_dict,
        "basis_args": basis_args,
        "A_qe": A_qe,
        "hop_tol": float(hop_tol),
        "saved_at": time.time(),
        "source": os.path.abspath(w90_filepath),
    }
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return path


def remote_cache_payload_path() -> str:
    return REMOTE_CACHE_NAME


def write_job_dir_cache(
    path: Path | str,
    *,
    key: str,
    tb_dict: dict,
    basis_args: dict,
    A_qe: Any = None,
    source: str = "",
    hop_tol: float = 1e-6,
) -> Path:
    """Write W90 TB pickle into a job directory for Hybrid upload/download.

    ``key`` must be the *client* cache key (from the Mac path), not the
    cluster path hash — otherwise ``load_parsed_tb`` rejects the payload.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
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
    return out


def ensure_cache_key(path: Path | str, expected_key: str) -> bool:
    """Rewrite payload['key'] to ``expected_key`` if mismatched. Return True if usable."""
    p = Path(path)
    if not p.is_file() or not expected_key:
        return False
    try:
        with p.open("rb") as f:
            payload = pickle.load(f)
        tb = payload.get("tb_dict") or {}
        n_hop = len(tb.get("list") or tb.get("H") or [])
        if n_hop == 0:
            return False
        if payload.get("key") == expected_key:
            return True
        payload["key"] = expected_key
        tmp = p.with_suffix(p.suffix + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(p)
        return True
    except Exception:
        return False
