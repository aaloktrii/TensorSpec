"""Optional live-mount smoke via ``TENSORSPEC_MAESTRO_SMOKE_DIR``.

Set the env var to a directory containing Maestro H5 files plus a
``smoke_cases.json`` listing what to check::

    [{"glob": "*_01234.h5", "kind": "xy_fine_4d", "ndim": 4, "labels": ["Y", "X"]}]

Each case is discovered by basename glob and opened with :func:`open_maestro`
for plan/kind metadata only -- no full tensor load. Skipped when the env var
is unset, the JSON is missing, or no matching file is present. Synthetic
fixtures elsewhere cover the two-motor line-scan kinds; live smoke only
verifies lazy open against the user's own mount.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tensorspec.core.io.loaders.maestro.lazy import open_maestro

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

_SMOKE_DIR = os.environ.get("TENSORSPEC_MAESTRO_SMOKE_DIR")

def _load_smoke_cases() -> list[tuple[str, str, int, list[str]]]:
    """(basename glob, expected kind, ndim, leading labels) from smoke_cases.json."""
    if not _SMOKE_DIR:
        return []
    spec = Path(_SMOKE_DIR) / "smoke_cases.json"
    if not spec.is_file():
        return []
    try:
        cases = json.loads(spec.read_text())
    except (OSError, ValueError):
        return []
    return [
        (str(c["glob"]), str(c["kind"]), int(c["ndim"]), list(c["labels"]))
        for c in cases
    ]


_SMOKE_CASES = _load_smoke_cases() or [("*.h5", "", 0, [])]

_SKIP_REASON = (
    "Set TENSORSPEC_MAESTRO_SMOKE_DIR to a folder with Maestro H5 files"
    " and a smoke_cases.json"
)


def _find_smoke_file(pattern: str) -> Path | None:
    if not _SMOKE_DIR:
        return None
    root = Path(_SMOKE_DIR)
    if not root.is_dir():
        return None
    matches = sorted(root.glob(pattern))
    return matches[0] if matches else None


pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("pattern", "expected_kind", "ndim", "label_prefix"),
    _SMOKE_CASES,
    ids=[pat.replace("*", "").replace(".h5", "") for pat, *_ in _SMOKE_CASES],
)
def test_live_open_maestro_smoke(pattern, expected_kind, ndim, label_prefix):
    path = _find_smoke_file(pattern) if expected_kind else None
    if path is None:
        pytest.skip(_SKIP_REASON)

    desc = open_maestro(str(path))
    try:
        assert desc.kind == expected_kind
        assert len(desc.shape) == ndim
        assert desc.labels[: len(label_prefix)] == label_prefix
        assert desc.shape[-2] > 0 and desc.shape[-1] > 0
        block = desc.read_block(0)
        assert block.ndim == 2
        assert block.shape[0] > 0 and block.shape[1] > 0
    finally:
        desc.close()
