from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile

from tensorspec.core.data_models import TensorData

I0_COLUMN_ALIASES = ("I0", "I_0", "i0", "beam_current", "BeamCurrent", "current", "I0_nA")

_POL_TAGS = ("CP", "CM", "LH", "LV")


def _json_val(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    return value


def _jsonify_series(series: dict[str, list]) -> dict[str, list]:
    return {key: [_json_val(v) for v in vals] for key, vals in series.items()}


def _match_i0_column(columns: list[str]) -> str | None:
    alias_fold = {alias.casefold(): alias for alias in I0_COLUMN_ALIASES}
    for col in columns:
        if col.casefold() in alias_fold:
            return col
    return None


def _coerce_i0_value(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    match = pd.Series([str(value)]).str.extract(
        r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    ).iloc[0, 0]
    return None if pd.isna(match) else float(match)


def infer_pol_from_name(name: str) -> str:
    """Return 'CP'|'CM'|'LH'|'LV'|'unknown' via case-insensitive substring."""
    folded = name.casefold()
    for tag in _POL_TAGS:
        if tag.casefold() in folded:
            return tag
    return "unknown"


def _stem_matches(path: Path, preferred_stem: str) -> bool:
    preferred_fold = preferred_stem.casefold()
    stem_fold = path.stem.casefold()
    return stem_fold == preferred_fold or preferred_fold in stem_fold


def _auto_attach_csv(directory: Path, preferred_stem: str | None = None) -> dict[str, Any] | None:
    """
    Attach beamline CSV only when unambiguous:
    - exactly one *.csv in directory, or
    - exactly one path whose stem matches preferred_stem.
    """
    candidates = find_beamline_csv(directory)
    if not candidates:
        return None

    if len(candidates) == 1:
        chosen = candidates[0]
    elif preferred_stem:
        matching = [path for path in candidates if _stem_matches(path, preferred_stem)]
        if len(matching) == 1:
            chosen = matching[0]
        else:
            return None
    else:
        return None

    csv_meta = load_beamline_csv(chosen)
    csv_meta["beamline_csv"] = str(chosen)
    return csv_meta


def find_beamline_csv(directory: Path, preferred_stem: str | None = None) -> list[Path]:
    """
    Return candidate *.csv paths in directory (non-recursive).
    If preferred_stem set, sort stem-matching paths first.
    Empty list if none.
    """
    directory = Path(directory)
    candidates = sorted(directory.glob("*.csv"))
    if not preferred_stem:
        return candidates

    matching = [path for path in candidates if _stem_matches(path, preferred_stem)]
    non_matching = [path for path in candidates if path not in matching]
    return matching + non_matching


def load_beamline_csv(path: Path | str) -> dict[str, Any]:
    """
    Parse CSV with pandas. Return JSON-serializable dict:
      columns: list[str]
      rows: list[dict]  # optional if huge — prefer columnar:
      series: dict[str, list]  # column -> values
      I0: float | list[float] | None  # first matching alias; scalar if len==1 else list
      beam_current: same as I0 when alias matches current*
    """
    path = Path(path)
    df = pd.read_csv(path)
    columns = [str(col) for col in df.columns]
    series = _jsonify_series({col: df[col].tolist() for col in columns})
    rows = [{str(k): _json_val(v) for k, v in row.items()} for row in df.to_dict(orient="records")]

    i0_col = _match_i0_column(columns)
    i0_value: float | list[float] | None = None
    beam_current: float | list[float] | None = None
    if i0_col is not None:
        raw_vals = [_coerce_i0_value(value) for value in series[i0_col]]
        if len(raw_vals) == 1:
            i0_value = raw_vals[0]
        else:
            i0_value = raw_vals
        if "current" in i0_col.casefold():
            beam_current = i0_value

    return {
        "columns": columns,
        "rows": rows,
        "series": series,
        "I0": i0_value,
        "beam_current": beam_current,
    }


def package_stack(
    frames: np.ndarray,
    frame_names: list[str],
    *,
    source: str,
    loader: str,
    csv_meta: dict[str, Any] | None = None,
) -> TensorData:
    """Shared builder: axes, labels, pol tags, merge csv_meta into metadata."""
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim == 2:
        frames = frames[np.newaxis, ...]
    if frames.ndim != 3:
        raise ValueError(f"Expected stack shape (n, y, x), got {frames.shape}")

    n_frames, ny, nx = frames.shape
    pol = [infer_pol_from_name(name) for name in frame_names]

    metadata: dict[str, Any] = {
        "frame_names": frame_names,
        "pol": pol,
        "pair_id": None,
        "source": source,
        "loader": loader,
    }

    if csv_meta is not None:
        metadata["csv_attached"] = True
        metadata["beamline_csv"] = csv_meta.get("beamline_csv", csv_meta.get("source"))
        metadata["I0"] = csv_meta.get("I0")
        if csv_meta.get("beam_current") is not None:
            metadata["beam_current"] = csv_meta["beam_current"]
        metadata["beamline_table"] = {
            "columns": csv_meta.get("columns"),
            "series": csv_meta.get("series"),
        }
    else:
        metadata["csv_attached"] = False
        metadata["I0"] = None

    return TensorData(
        value=frames,
        axes=[np.arange(n_frames), np.arange(ny), np.arange(nx)],
        labels=["frame", "y", "x"],
        units=["", "px", "px"],
        data_type="Experimental PEEM",
        metadata=metadata,
    )


def _read_tif_frames(path: Path) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        if not tif.pages:
            raise ValueError(f"No TIFF pages in {path}")

        if len(tif.pages) == 1:
            frame = tif.pages[0].asarray()
            if frame.ndim == 3:
                return frame
            if frame.ndim == 2:
                return frame[np.newaxis, ...]
            raise ValueError(f"Unexpected TIFF page shape {frame.shape} in {path}")

        frames_list: list[np.ndarray] = []
        ref_shape: tuple[int, ...] | None = None
        for page_idx, page in enumerate(tif.pages):
            frame = page.asarray()
            if frame.ndim != 2:
                raise ValueError(
                    f"Expected 2D TIFF page in {path}, page {page_idx} shape {frame.shape}"
                )
            if ref_shape is None:
                ref_shape = frame.shape
            elif frame.shape != ref_shape:
                raise ValueError(
                    f"TIFF pages differ in {path}: page {page_idx} {frame.shape} vs {ref_shape}"
                )
            frames_list.append(frame)
        return np.stack(frames_list, axis=0)


def load_tif_stack(path: Path | str, *, csv_path: Path | str | None = None) -> TensorData:
    """Multipage TIF → TensorData shape (n_frames, ny, nx), float64."""
    path = Path(path)
    frames = _read_tif_frames(path)
    n_frames = frames.shape[0]
    frame_names = [f"frame_{i}" for i in range(n_frames)]

    csv_meta: dict[str, Any] | None = None
    if csv_path is not None:
        csv_meta = load_beamline_csv(csv_path)
        csv_meta["beamline_csv"] = str(Path(csv_path))
    else:
        csv_meta = _auto_attach_csv(path.parent, preferred_stem=path.stem)

    return package_stack(
        frames,
        frame_names,
        source=str(path),
        loader="tif_stack",
        csv_meta=csv_meta,
    )


def load_tif_sequence(directory: Path | str, *, csv_path: Path | str | None = None) -> TensorData:
    """Sorted *.tif/*.tiff in directory → TensorData (n_frames, ny, nx)."""
    directory = Path(directory)
    tif_paths = sorted(directory.glob("*.tif")) + sorted(directory.glob("*.tiff"))

    seen: set[str] = set()
    unique_paths: list[Path] = []
    for candidate in tif_paths:
        key = candidate.name.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(candidate)

    if not unique_paths:
        raise ValueError(f"No TIF files found in {directory}")

    frames_list: list[np.ndarray] = []
    frame_names: list[str] = []
    ref_shape: tuple[int, ...] | None = None
    for tif_path in unique_paths:
        frame = tifffile.imread(tif_path)
        if frame.ndim != 2:
            raise ValueError(f"Expected 2D frame in {tif_path.name}, got shape {frame.shape}")
        if ref_shape is None:
            ref_shape = frame.shape
        elif frame.shape != ref_shape:
            raise ValueError(f"TIFF pages differ in {directory}: {tif_path.name} {frame.shape} vs {ref_shape}")
        frames_list.append(frame)
        frame_names.append(tif_path.stem)

    frames = np.stack(frames_list, axis=0)

    csv_meta: dict[str, Any] | None = None
    if csv_path is not None:
        csv_meta = load_beamline_csv(csv_path)
        csv_meta["beamline_csv"] = str(Path(csv_path))
    else:
        csv_meta = _auto_attach_csv(directory, preferred_stem=directory.name)

    return package_stack(
        frames,
        frame_names,
        source=str(directory),
        loader="tif_sequence",
        csv_meta=csv_meta,
    )
