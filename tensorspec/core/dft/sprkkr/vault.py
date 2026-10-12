"""Converged-potential vault: reuse a `.pot` instead of re-running SCF.

No Qt. No pymatgen import at module load time -- `pot_key()` only ever
touches duck-typed attributes it needs off the `structure` object
(`.lattice.matrix`, iteration yielding `.species_string` / `.frac_coords`
sites), so pymatgen never has to be imported here at all; callers already
have a pymatgen Structure and pass it straight in. No workspace import
either -- the GUI layer mirrors vault entries into
`global_workspace.push_remote_run` itself, later, on top of this module.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .params import ScfParams


def pot_key(structure: Any, scf_params: ScfParams) -> str:
    """Stable content hash identifying a converged potential.

    Built from the structure's lattice matrix (rounded 6 dp) + sorted
    (species, frac_coords rounded 6 dp) site list + every ScfParams field.
    Two Structure objects with identical content (any instance, any
    ordering of sites) hash the same; changing any ScfParams field (e.g.
    `nl`) changes the hash.
    """
    matrix = np.round(np.asarray(structure.lattice.matrix, dtype=float), 6).tolist()

    sites = []
    for site in structure:
        species = str(getattr(site, "species_string", None) or site.species)
        frac = tuple(round(float(x), 6) for x in site.frac_coords)
        sites.append([species, list(frac)])
    sites.sort(key=lambda s: (s[0], s[1]))

    payload = {
        "matrix": matrix,
        "sites": sites,
        "scf": asdict(scf_params),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


@dataclass
class VaultEntry:
    key: str
    name: str
    pot_path: Optional[str] = None
    cluster: Optional[str] = None
    remote_path: Optional[str] = None
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    meta: Dict[str, Any] = field(default_factory=dict)


class Vault:
    """Flat JSON-indexed store of converged potentials under `root`.

    `root/index.json` maps entry name -> VaultEntry fields. A locally
    registered `.pot` file is copied into `root/<key>/<basename>`; a
    remote-only entry (cluster + remote_path, pot_path=None) just records
    where it lives.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        if not self.index_path.exists():
            self._write_index({})

    def _read_index(self) -> Dict[str, dict]:
        if not self.index_path.exists():
            return {}
        text = self.index_path.read_text()
        if not text.strip():
            return {}
        return json.loads(text)

    def _write_index(self, data: Dict[str, dict]) -> None:
        self.index_path.write_text(json.dumps(data, indent=2, sort_keys=True))

    def register(
        self,
        key: str,
        name: str,
        pot_path: Optional[str] = None,
        cluster: Optional[str] = None,
        remote_path: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> VaultEntry:
        stored_pot_path = None
        if pot_path is not None:
            src = Path(pot_path)
            dest_dir = self.root / key
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / src.name
            shutil.copy2(src, dest)
            stored_pot_path = str(dest)

        entry = VaultEntry(
            key=key,
            name=name,
            pot_path=stored_pot_path,
            cluster=cluster,
            remote_path=remote_path,
            meta=dict(meta or {}),
        )

        data = self._read_index()
        data[name] = asdict(entry)
        self._write_index(data)
        return entry

    def lookup(self, key: str) -> Optional[VaultEntry]:
        data = self._read_index()
        for row in data.values():
            if row.get("key") == key:
                return VaultEntry(**row)
        return None

    def get(self, name: str) -> Optional[VaultEntry]:
        data = self._read_index()
        row = data.get(name)
        return VaultEntry(**row) if row is not None else None

    def list(self) -> List[VaultEntry]:
        data = self._read_index()
        return [VaultEntry(**row) for row in data.values()]

    def remove(self, name: str) -> bool:
        data = self._read_index()
        row = data.pop(name, None)
        if row is None:
            return False
        self._write_index(data)
        pot_path = row.get("pot_path")
        if pot_path:
            entry_dir = Path(pot_path).parent
            if entry_dir.exists() and entry_dir.parent == self.root:
                shutil.rmtree(entry_dir, ignore_errors=True)
        return True
