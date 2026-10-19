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
        pol: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Any of the three formats -> PEEM dataset in the workspace.

        pol: {path: CP / CM / LH / LV} for items whose data do not record the polarization
        (core/io/peem_beamline_loaders.detect_pol); file and folder names are not needed.

        i0_normalize None = beamline default (UE49 on, ALS off, UView has no I0).
        Detector maps (set_detector_maps), tpx3 and drop_bad_frames apply to UE49 only.
        """
        tensor = peem_beamline_loaders.load_beamline(
            paths, i0_normalize=i0_normalize, allow_mixed_fov=allow_mixed_fov,
            dead_map_path=self.dead_map_path, norm_image_path=self.norm_image_path,
            tpx3=tpx3, drop_bad_frames=drop_bad_frames, pol=pol)
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

    @staticmethod
    def inspect_folder(path: str | Path) -> dict[str, Any]:
        """What a folder holds and how to load it (core/io/peem_folder.inspect_folder): kind beamline /
        tif / images / container, the paths to load, and the files that are not frames ("ignored")."""
        from tensorspec.core.io import peem_folder

        return peem_folder.inspect_folder(path)

    def load_images(self, path: str | Path, name: str | None = None) -> dict[str, Any]:
        """A folder of PNG / JPEG frames -> PEEM dataset (same summary as load_path)."""
        from tensorspec.core.io import peem_folder
        from tensorspec.gui.services.peem_service import _load_summary, _safe_label  # imports this module

        tensor = peem_folder.load_image_folder(path)
        label = _safe_label(re.sub(r"[^A-Za-z0-9_.-]", "_", name or Path(path).name), "images")
        global_workspace.push_spectroscopy_data(label, tensor)
        _record(label, "load", {"loader": "image_sequence", "source": str(path), "shape": list(tensor.value.shape)})
        summary = _load_summary(label, tensor, csv_prompt=False, csv_candidates=[])
        summary["format"] = "images"
        return summary

    # ------------------------------------------------------------- map files (.nc)
    OLD_SIGNS = {"pcd": ("convention", "(CL-CR)/(CL+CR)"), "pld": ("formula", "(LH-LV)/(LH+LV)")}

    def load_maps_nc(self, path: str | Path, name: str | None = None) -> dict[str, Any]:
        """A NetCDF map file (core/io/peem_maps_nc: batch maps.nc, analysis outputs, any 2-D variables)
        -> PEEM dataset. /analysis/nc holds every map (data = the first asymmetry-type one; each
        variable's type in attrs 'var_types'); with channel means (mean_CP/CM, CL/CR, LH/LV) also
        /analysis/stack, and a pcd / pld variable as /analysis/pcd | pld (batch maps with the older
        sign flipped to the suite convention, noted in attrs), so the map tools and Compare (7.)
        work as on a loaded stack. Frames (Frame view): the 2-D maps, or a 3-D variable."""
        from tensorspec.core.io import peem_maps_nc
        from tensorspec.core.io.peem_loaders import package_stack
        from tensorspec.core.peem import products
        from tensorspec.gui.services.peem_service import _load_summary, _safe_label  # imports this module

        r = peem_maps_nc.read_maps_nc(path)
        label = _safe_label(re.sub(r"[^A-Za-z0-9_.-]", "_", name or peem_maps_nc.dataset_name(path))[:64], "maps")
        maps, types, um = r["maps"], r["types"], r["um_per_px"]
        flips = {}
        for var, (key, old) in self.OLD_SIGNS.items():
            if var in maps and r["attrs"].get(key) == old:
                maps[var] = -maps[var]
                flips[var] = f"sign flipped to the suite convention (file: {old})"
        if r["stacks"]:
            k3 = next(iter(r["stacks"]))
            frames, names = r["stacks"][k3], [f"{k3}_{i}" for i in range(len(r["stacks"][k3]))]
            pol = None
        else:
            names = list(maps)
            frames = np.stack([maps[k] for k in names])
            pol = [{"mean_cp": "CP", "mean_cl": "CP", "mean_cm": "CM", "mean_cr": "CM", "mean_lh": "LH",
                    "mean_lv": "LV"}.get(k.lower(), "?") for k in names]
        tensor = package_stack(frames, names, source=str(path), loader="maps_nc", pol=pol, dtype=np.float32)
        tensor.metadata.update({"um_per_px": um, "file_attrs": {k: str(v)[:200] for k, v in r["attrs"].items()},
                                "map_types": types})
        global_workspace.push_spectroscopy_data(label, tensor)
        pair = r["pair"]
        if pair and pair[2] in ("xmcd", "xld"):
            ka, kb, kind = pair
            ta, tb = ("CP", "CM") if kind == "xmcd" else ("LH", "LV")
            a, b = maps[ka], maps[kb]
            ds = products.map_dataset("stack", a + b, extra={f"mean_{ta.lower()}": a, f"mean_{tb.lower()}": b},
                                      attrs={"channel_a": ta, "channel_b": tb, "pair": kind,
                                             "from_step": f"loaded from {Path(path).name}"}, um_per_px=um)
            self._write(label, "stack", ds)
        for var in ("pcd", "pld"):
            if var in maps:
                extra = {k: maps[k] for k in maps if k.lower().startswith("mean_")}
                ds = products.map_dataset(var, maps[var], extra=extra, um_per_px=um,
                                          attrs={"from_step": f"loaded from {Path(path).name}",
                                                 "sign_note": flips.get(var, "as in the file")})
                self._write(label, var, ds)
        primary = r["primary"]
        ds = products.map_dataset("nc", maps[primary], extra={k: v for k, v in maps.items()}, um_per_px=um,
                                  attrs={"source": str(path), "primary": primary, "var_types": json.dumps(types),
                                         "sign_notes": json.dumps(flips)},
                                  long_name=f"{primary} ({Path(path).name})")
        self._write(label, "nc", ds)
        _record(label, "load", {"loader": "maps_nc", "source": str(path), "maps": list(maps)[:20],
                                "primary": primary, "um_per_px": um, "sign_flips": json.dumps(flips)})
        summary = _load_summary(label, tensor, csv_prompt=False, csv_candidates=[])
        summary.update({"format": "maps (.nc)", "maps": list(maps), "primary": primary, "map_types": types,
                        "um_per_px": um, "sign_flips": flips})
        return summary

    def nc_outputs(self, name: str) -> list[tuple[str, str]]:
        """[(label, variable)] of the maps loaded from a .nc file (data = the primary map)."""
        at = self.get_product_attrs(name, "nc")
        prim = at.get("primary", "")
        return [(f"{prim} (primary)", "data")] + [(v, v) for v in self.map_variables(name, "nc")
                                                  if v not in ("data", prim)]

    def map_type(self, name: str, kind: str, var: str = "data") -> str:
        """'signed', 'intensity' or 'domain' of a loaded .nc variable (kind 'nc'); '' otherwise."""
        if kind != "nc":
            return ""
        at = self.get_product_attrs(name, "nc")
        types = json.loads(at.get("var_types", "{}"))
        return types.get(at.get("primary", "") if var in ("", "data") else var, "")

    @staticmethod
    def list_datasets() -> list[str]:
        """Names of the loaded PEEM datasets (anything with frames in /raw), in load order."""
        return [n for n in list(getattr(global_workspace, "_data", {}))
                if global_workspace.pull_tensor_data(n, "raw") is not None]

    def unload(self, name: str) -> None:
        """Remove a dataset (raw, processed and all maps) from the workspace and free its memory."""
        global_workspace.remove(name)
