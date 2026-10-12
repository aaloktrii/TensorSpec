"""Local SPR-KKR user settings (bin dir, vault root, default nproc). Zero Qt.

Resolution order for each key: explicit arg > env var > ~/.tensorspec_sprkkr.json > default.
Env: SPRKKR_BIN, SPRKKR_VAULT, SPRKKR_NPROC.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

SETTINGS_PATH = Path("~/.tensorspec_sprkkr.json").expanduser()


@dataclass
class SprkkrSettings:
    bin_dir: str = str(Path("~/bin").expanduser())
    vault_root: str = str(Path("~/.tensorspec_sprkkr_vault").expanduser())
    nproc: int = max(1, (os.cpu_count() or 2) // 2)
    suffix: str = "9.7"
    workdir_root: str = "scratch/sprkkr_gui_run"


def load_settings(path: Path = SETTINGS_PATH) -> SprkkrSettings:
    s = SprkkrSettings()
    try:
        if path.is_file():
            data = json.loads(path.read_text())
            for k, v in data.items():
                if hasattr(s, k):
                    setattr(s, k, v)
    except Exception:
        pass
    if os.environ.get("SPRKKR_BIN"):
        s.bin_dir = os.environ["SPRKKR_BIN"]
    if os.environ.get("SPRKKR_VAULT"):
        s.vault_root = os.environ["SPRKKR_VAULT"]
    if os.environ.get("SPRKKR_NPROC"):
        try:
            s.nproc = int(os.environ["SPRKKR_NPROC"])
        except ValueError:
            pass
    return s


def save_settings(s: SprkkrSettings, path: Path = SETTINGS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(s), indent=2))


def local_binaries_present(bin_dir: str, suffix: str = "9.7") -> dict:
    """{'kkrscf': path|None, 'kkrspec': path|None, 'kkrscf_mpi': ..., 'kkrspec_mpi': ...}."""
    out = {}
    for name in ("kkrscf", "kkrspec"):
        p = Path(bin_dir) / f"{name}{suffix}"
        pm = Path(bin_dir) / f"{name}{suffix}MPI"
        out[name] = str(p) if p.is_file() else None
        out[f"{name}_mpi"] = str(pm) if pm.is_file() else None
    return out
