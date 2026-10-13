"""PEEM service, XMCD-imaging steps: Loading UE49 / UView (BNL ESM) / P3B (ALS) data into one PEEM dataset (core/io/peem_beamline_loaders).

Mixin of ``PeemService`` (gui/services/peem_service.py). Moved verbatim from the former
gui/services/peem_xmcd_service.py."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from tensorspec.core.data_models import TensorData
from tensorspec.core.peem.bg import resolve_energy
from tensorspec.core.io import ue49_loaders as loaders
from tensorspec.core.io import peem_beamline_loaders
from tensorspec.core.workspace import global_workspace
from tensorspec.gui.services.peem_steps.common import _record


class PeemLoadSteps:
    dead_map_path: str | None = None
    norm_image_path: str | None = None

    # ------------------------------------------------------------- loading
    def set_detector_maps(self, dead_map_path: str | None, norm_image_path: str | None) -> dict:
        for p in (dead_map_path, norm_image_path):
            if p and not Path(p).is_file():
                raise ValueError(f"file not found: {p}")
        self.dead_map_path = dead_map_path or None
        self.norm_image_path = norm_image_path or None
        return {"dead_map": self.dead_map_path, "norm_image": self.norm_image_path}

    @staticmethod
    def find_ue49_scans(day_dir: str | Path, prefix: str | None = None) -> list[str]:
        return [str(p) for p in loaders.find_scans(day_dir, prefix)]

    def load_beamline(
        self,
        paths: list[str | Path],
        name: str | None = None,
        *,
        i0_normalize: bool | None = None,
        tpx3: bool = True,
        drop_bad_frames: bool = False,
        allow_mixed_fov: bool = False,
    ) -> dict[str, Any]:
        """Any of the three formats -> PEEM dataset in the workspace.

        i0_normalize None = beamline default (UE49 on, ALS off, UView has no I0).
        Detector maps (set_detector_maps), tpx3 and drop_bad_frames apply to UE49 only.
        """
        tensor = peem_beamline_loaders.load_beamline(
            paths, i0_normalize=i0_normalize, allow_mixed_fov=allow_mixed_fov,
            dead_map_path=self.dead_map_path, norm_image_path=self.norm_image_path,
            tpx3=tpx3, drop_bad_frames=drop_bad_frames)
        return self._push(tensor, name)

    def load_ue49(
        self,
        txt_paths: list[str | Path],
        name: str | None = None,
        *,
        tpx3: bool = True,
        i0_normalize: bool = True,
        drop_bad_frames: bool = False,
        name_filter: str | None = None,
        allow_mixed_fov: bool = False,
    ) -> dict[str, Any]:
        tensor = loaders.load_scans(
            txt_paths,
            dead_map_path=self.dead_map_path,
            norm_image_path=self.norm_image_path,
            tpx3=tpx3,
            i0_normalize=i0_normalize,
            drop_bad_frames=drop_bad_frames,
            name_filter=name_filter,
            allow_mixed_fov=allow_mixed_fov,
        )
        tensor.metadata.update(format="ue49", default_convention=peem_beamline_loaders.DEFAULT_CONVENTION["ue49"])
        return self._push(tensor, name)

    def load_uview(self, folders: list[str | Path], name: str | None = None,
                   allow_mixed_fov: bool = False) -> dict[str, Any]:
        """Elmitec UView acquisition folders (BNL ESM, one polarization each) -> PEEM dataset."""
        for f in folders:
            if peem_beamline_loaders.detect_format(f)[0] != "uview":
                raise ValueError(f"{Path(f).name}: not a UView acquisition")
        return self.load_beamline(folders, name, allow_mixed_fov=allow_mixed_fov)

    @staticmethod
    def _auto_name(md: dict[str, Any]) -> str:
        scans = md["scans"]
        if md.get("format") == "uview":  # XPEEM_20um_..._Cr2p_CL_0deg + ..._CR_0deg -> 20um_579eV_0deg_CL-CR
            toks = [t for s in scans for t in md["pol_by_scan"][s]]
            base = re.sub(r"(?:^|_)(CL|CR|LH|LV)(?=_|$)", "", scans[0])
            base = re.sub(r"^XPEEM_|_\d+CA_\d+AN|_Cr2p", "", base)
            return f"{base}_{'-'.join(dict.fromkeys(toks))}"
        return "_".join(scans)

    def _push(self, tensor: TensorData, name: str | None) -> dict[str, Any]:
        md = tensor.metadata
        label = (name or self._auto_name(md)).replace(" ", "_").replace("+", "p")
        label = re.sub(r"[^A-Za-z0-9_.-]", "_", label)[:64]
        from tensorspec.gui.services.peem_service import SAFE_NAME  # peem_service imports this module

        if not SAFE_NAME.match(label):
            raise ValueError(f"invalid dataset name {label!r}")
        global_workspace.push_spectroscopy_data(label, tensor)
        pols = md["pol"]
        out = {
            "name": label,
            "format": md.get("format"),
            "beamline": md.get("beamline"),
            "default_convention": md.get("default_convention"),
            "shape": [int(s) for s in tensor.value.shape],
            "scans": md["scans"],
            "pol_counts": {p: pols.count(p) for p in sorted(set(pols))},
            "fov_um": md.get("fov_um"),
            "um_per_px": md.get("um_per_px"),
            "temperature_K": md.get("temperature_K"),
            "i0_normalized": md.get("i0_normalized"),
            "dropped_frames": md.get("dropped_frames"),
            "pol_by_scan": md.get("pol_by_scan"),
            "energy_range": self._energy_range(tensor),
        }
        for k in ("pol_source", "i0_check", "pol_conflicts", "missing_frames"):
            if md.get(k):
                out[k] = md[k]
        _record(label, "load", {k: out[k] for k in ("format", "beamline", "shape", "fov_um", "um_per_px", "i0_normalized")}
                | {"scans": [str(x) for x in md["scans"]][:8], "pol_counts": json.dumps(out["pol_counts"])})
        return out

    @staticmethod
    def _energy_range(tensor: TensorData) -> list[float] | None:
        energy, src = resolve_energy(tensor.value.shape[0], tensor.metadata or {})
        return None if src == "index" else [float(np.min(energy)), float(np.max(energy))]

    def unload(self, name: str) -> None:
        """Remove a dataset (raw, processed and all maps) from the workspace and free its memory."""
        global_workspace.remove(name)
