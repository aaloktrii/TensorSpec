"""PEEM service: maps and spectra from the PEEM tab's own corrected channels.

Mixin of ``PeemService`` (gui/services/peem_service.py). The PEEM tab loads, pairs (Stack Pairs)
and drift-corrects into ``/processed`` (pair, channel=2, y, x). These steps start from that cube and
use each frame's own photon energy (Stack Pairs pairs frames in order, so when the CP and CM sweeps
differ, a pair can hold two energies):

    channel_stack(name, e_lo, e_hi)   one image per polarization (mean of its frames, optionally
                                      within a photon-energy window), CM refined onto CP
                                      -> /analysis/stack; then asymmetry, image analysis, ROIs,
                                      domain maps and export work as for any map
    to_sum_rule(name)                 the corrected frames regrouped by photon energy (one CP and
                                      one CM image per energy) -> new dataset <name>_sumrule, paired
                                      + separated with the PEEM tab's own pair / separate, energy per
                                      pair in its beamline table, so background and sum rule get eV
    sum_rule_datasets()               the datasets made by to_sum_rule (the Sum Rule window's list)
    channel_image(name)               mean image of the corrected pairs (to draw the region on)
    roi_spectra(name, roi)            CP / CM spectra averaged over a region, per photon energy
    domain_width(name, kind, var, roi)  mean domain width = region area / domain-wall length
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
from scipy import ndimage

from tensorspec.core.data_models import TensorData
from tensorspec.core.peem import dichroism, products
from tensorspec.core.peem.bg import resolve_energy
from tensorspec.core.peem.engine import average_by_energy
from tensorspec.core.peem.roi import roi_to_mask
from tensorspec.core.workspace import global_workspace
from tensorspec.gui.services.peem_steps.common import _record, _require_raw

PAIRS = {"xmcd": ("CP", "CM"), "xld": ("LH", "LV")}


def _paired_cube(name: str):
    cube = global_workspace.pull_tensor_data(name, "processed")
    if cube is None or np.ndim(cube.value) != 4:
        raise ValueError("Run 2. Stack Pairs first (and 3. Drift on the processed data, if needed).")
    tags = [str(t) for t in (cube.metadata or {}).get("channel_tags", [])]
    pair = next((p for p, t in PAIRS.items() if set(t) == set(tags)), None)
    if pair is None:
        raise ValueError(f"channels {tags}: expected CP / CM or LH / LV")
    return cube, tags, pair


def _channels(name: str) -> dict[str, Any]:
    """The corrected frames of both channels with each frame's own photon energy (None without an
    energy axis): {pair, ta, tb, fa, fb, ea, eb, mixed (median |E_a - E_b| of a pair, eV)}."""
    cube, tags, pair = _paired_cube(name)
    ta, tb = PAIRS[pair]
    ia, ib = tags.index(ta), tags.index(tb)
    raw = _require_raw(name)
    md = raw.metadata or {}
    n = cube.value.shape[0]
    ea = eb = None
    per_pair, src = resolve_energy(n, md)
    if src != "index":  # one table row per pair (datasets made by to_sum_rule)
        ea = eb = np.asarray(per_pair, float)
    else:
        energy, src = resolve_energy(len(raw.value), md)
        sources = (cube.metadata or {}).get("pair_sources") or []
        if src != "index" and len(sources) == n:
            idx = [{c["tag"]: c["frame_index"] for c in s["channels"]} for s in sources]
            ea = np.array([energy[d[ta]] for d in idx], float)
            eb = np.array([energy[d[tb]] for d in idx], float)
    mixed = float(np.median(np.abs(ea - eb))) if ea is not None else 0.0
    return {"pair": pair, "ta": ta, "tb": tb, "fa": cube.value[:, ia], "fb": cube.value[:, ib],
            "ea": ea, "eb": eb, "mixed": mixed}


def _step(e: np.ndarray) -> float:
    u = np.unique(np.round(e, 4))
    return float(np.median(np.diff(u))) if len(u) > 1 else 0.0


def _mixing_note(ch: dict) -> str | None:
    if ch["ea"] is None or ch["mixed"] <= max(_step(ch["ea"]), 1e-3) / 2:
        return None
    return (f"Stack Pairs paired {ch['ta']} / {ch['tb']} frames of different photon energies (median "
            f"{ch['mixed']:.2f} eV apart: the two sweeps differ); frames are regrouped by their own energy")


class PeemChannelSteps:
    def channel_stack(self, name: str, e_lo: float | None = None, e_hi: float | None = None,
                      refine: bool = True) -> dict[str, Any]:
        """Stack the corrected CP / CM (or LH / LV) frames into one image each -> /analysis/stack.
        e_lo / e_hi: keep only frames in this photon-energy window (each frame by its own energy)."""
        ch = _channels(name)
        ta, tb = ch["ta"], ch["tb"]
        ka = np.ones(len(ch["fa"]), bool)
        kb = np.ones(len(ch["fb"]), bool)
        if ch["ea"] is not None and (e_lo is not None or e_hi is not None):
            lo, hi = (-np.inf if e_lo is None else e_lo), (np.inf if e_hi is None else e_hi)
            ka, kb = (ch["ea"] >= lo) & (ch["ea"] <= hi), (ch["eb"] >= lo) & (ch["eb"] <= hi)
            if not ka.any() or not kb.any():
                raise ValueError(f"no {ta} or {tb} frame between {e_lo} and {e_hi} eV")
        fa, fb = np.asarray(ch["fa"][ka], float), np.asarray(ch["fb"][kb], float)
        if refine:
            mean_a, mean_b, shift = dichroism.stack_means(fa, fb, dichroism._central_box(fa.shape))
        else:
            mean_a, mean_b, shift = fa.mean(0), fb.mean(0), (0.0, 0.0)
        na, nb = int(ka.sum()), int(kb.sum())
        attrs = {"pair": ch["pair"], "channel_a": ta, "channel_b": tb, "channel_a_name": ta, "channel_b_name": tb,
                 "n_a": na, "n_b": nb, "source_node": "processed (paired)",
                 "refine_shift_dy": float(shift[0]), "refine_shift_dx": float(shift[1])}
        if ch["ea"] is not None:
            e = np.concatenate([ch["ea"][ka], ch["eb"][kb]])
            attrs["energy_window_eV"] = [float(e.min()), float(e.max())]
        ds = products.map_dataset(
            "stack", mean_a + mean_b,
            extra={f"mean_{ta.lower()}": mean_a, f"mean_{tb.lower()}": mean_b,
                   f"sum_{ta.lower()}": mean_a * na, f"sum_{tb.lower()}": mean_b * nb},
            attrs=attrs, um_per_px=self._um_per_px(name), long_name="stacked images, sum of both polarizations")
        self._write(name, "stack", ds)
        out = {"name": name, "step": "stack channels", "pair": ch["pair"], "n_pairs": min(na, nb),
               "n_frames": {ta: na, tb: nb}, "energy_window_eV": attrs.get("energy_window_eV"),
               "B_onto_A_shift_px": [round(float(shift[0]), 3), round(float(shift[1]), 3)], "next": "asymmetry"}
        _record(name, "stack channels", out)
        return out

    def map_to_tools(self, name: str, kind: str) -> dict[str, Any]:
        """A map made on the panel's section 4 (make_pcd_map / make_pld_map) -> the map tools: its two
        registered mean images become /analysis/stack, and the asymmetry step rebuilds /analysis/<kind>
        from them (same formula and convention), adding the variables the tools use (difference ...).
        Image analysis, ROIs / domains and Compare then work on it as on a section-5 map."""
        ds = global_workspace.pull_analysis_data(name, kind)
        if ds is None or kind not in ("pcd", "pld"):
            raise ValueError(f"no /analysis/{kind} map: run Make PCD / Make PLD first")
        ta, tb = ("CP", "CM") if kind == "pcd" else ("LH", "LV")
        at = dict(ds.attrs)
        a = np.asarray(ds[f"mean_{ta.lower()}"].values, float)
        b = np.asarray(ds[f"mean_{tb.lower()}"].values, float)
        na, nb = int(at.get(f"n_{ta.lower()}", 1)), int(at.get(f"n_{tb.lower()}", 1))
        attrs = {"pair": "xmcd" if kind == "pcd" else "xld", "channel_a": ta, "channel_b": tb,
                 "channel_a_name": ta, "channel_b_name": tb, "n_a": na, "n_b": nb,
                 "source_node": f"Make {kind.upper()} ({at.get('source_node', '')})",
                 "registered": bool(at.get("registered", True))}
        stack = products.map_dataset(
            "stack", a + b, extra={f"mean_{ta.lower()}": a, f"mean_{tb.lower()}": b,
                                   f"sum_{ta.lower()}": a * na, f"sum_{tb.lower()}": b * nb},
            attrs=attrs, um_per_px=self._um_per_px(name), long_name="stacked images, sum of both polarizations")
        self._write(name, "stack", stack)
        out = self.asymmetry(name, convention=at.get("convention") or None) if kind == "pcd" else self.asymmetry(name)
        _record(name, "map to tools", {"kind": kind, "n_frames": {ta: na, tb: nb}})
        return out

    def to_sum_rule(self, name: str, energy_bin_eV: float | None = None) -> dict[str, Any]:
        """The PEEM tab's corrected frames -> new dataset ``<name>_sumrule`` for the Sum Rule window.

        Frames are regrouped by their own photon energy (core/peem/engine.average_by_energy; bin
        width ``energy_bin_eV``, None = the energy step): one CP and one CM image per energy, as
        /raw = CP(E1), CM(E1), CP(E2) ... with one beamline-table row per energy (what the background /
        sum-rule steps read for a channel). Then paired and separated (PeemService.pair / separate).
        Without an energy axis the corrected pairs are handed over as they are. The original
        dataset is not changed."""
        ch = _channels(name)
        ta, tb = ch["ta"], ch["tb"]
        raw = _require_raw(name)
        ny, nx = ch["fa"].shape[1:]
        note = _mixing_note(ch)
        if ch["ea"] is not None:
            frames = np.concatenate([ch["fa"], ch["fb"]])
            pol = [ta] * len(ch["fa"]) + [tb] * len(ch["fb"])
            energy = np.concatenate([ch["ea"], ch["eb"]])
            stack, pol_out, points, counts, width = average_by_energy(frames, pol, energy, (ta, tb), energy_bin_eV)
            table = {"columns": ["energy"], "series": {"energy": [float(p) for p in points]}}
        else:
            stack = np.empty((2 * len(ch["fa"]), ny, nx), np.float32)
            stack[0::2], stack[1::2] = ch["fa"], ch["fb"]
            pol_out, points, counts, width, table = [ta, tb] * len(ch["fa"]), None, None, None, None
        n = len(stack) // 2
        md = {k: v for k, v in (raw.metadata or {}).items()
              if k in ("format", "beamline", "default_convention", "fov_um", "um_per_px", "source", "loader")}
        md.update({"pol": list(pol_out), "frame_names": [f"{t}_{i:04d}" for i in range(n) for t in (ta, tb)],
                   "derived_by": "PEEM tab: corrected frames for the Sum Rule window", "derived_from": name})
        if table:
            md["beamline_table"] = table
        label = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{name}_sumrule")[:64]
        global_workspace.push_spectroscopy_data(label, TensorData(
            value=np.asarray(stack, np.float32), axes=[np.arange(len(stack), dtype=float), raw.axes[1], raw.axes[2]],
            labels=["frame", "y", "x"], units=list(raw.units), data_type="Experimental PEEM", metadata=md))
        paired = self.pair(label, "CP_CM" if ch["pair"] == "xmcd" else "LH_LV")
        sep = self.separate(label)
        out = {"name": label, "from": name, "pair": ch["pair"], "n_pairs": paired["n_pairs"], "channels": sep["channels"],
               "energy_range_eV": [float(min(points)), float(max(points))] if points else None,
               "energy_bin_eV": round(float(width), 4) if width else None}
        if counts:
            out["frames_per_energy"] = {ta: [c[0] for c in counts], tb: [c[1] for c in counts]}
        notes = [x for x in (note, None if points else
                             "no photon-energy axis: background and sum rule will run on the pair index") if x]
        if notes:
            out["note"] = "; ".join(notes)
        _record(name, "to sum rule", {k: v for k, v in out.items() if k not in ("name", "frames_per_energy")}
                | {"dataset": label})
        return out

    @staticmethod
    def sum_rule_datasets() -> list[str]:
        """Datasets made by to_sum_rule (corrected frames, one energy per pair), newest last."""
        names = []
        for name, item in list(getattr(global_workspace, "_data", {}).items()):
            if item.get("type") != "spectroscopy_tree":
                continue
            raw = global_workspace.pull_tensor_data(name, "raw")
            if raw is not None and str((raw.metadata or {}).get("derived_by", "")).startswith("PEEM tab"):
                names.append(name)
        return names

    def channel_image(self, name: str) -> np.ndarray:
        """Mean of both channels over all corrected pairs: the image the region is drawn on."""
        cube, _, _ = _paired_cube(name)
        return np.asarray(cube.value, float).mean(axis=(0, 1))

    def roi_spectra(self, name: str, roi: dict | None = None) -> dict[str, Any]:
        """CP / CM spectra (mean over ``roi``, whole image if None) per photon energy, with
        XAS = (CP + CM) / 2 and XMCD = CM − CP (LV − LH for linear). Frames are grouped by their own
        energy, so mismatched CP / CM sweeps still give matching points."""
        ch = _channels(name)
        ta, tb = ch["ta"], ch["tb"]
        ny, nx = ch["fa"].shape[1:]
        mask = None if roi is None else roi_to_mask(ny, nx, roi)
        mean = (lambda f: np.asarray(f, float)[:, mask].mean(axis=1)) if mask is not None else \
               (lambda f: np.asarray(f, float).mean(axis=(1, 2)))
        a, b = mean(ch["fa"]), mean(ch["fb"])
        if ch["ea"] is not None:
            vals = np.concatenate([a, b])[:, None, None]
            pol = [ta] * len(a) + [tb] * len(b)
            stack, _, points, _, _ = average_by_energy(vals, pol, np.concatenate([ch["ea"], ch["eb"]]), (ta, tb))
            a, b, x, src = stack[0::2, 0, 0], stack[1::2, 0, 0], np.asarray(points, float), "frames"
        else:
            x, src = np.arange(len(a), dtype=float), "index"
        order = np.argsort(x)
        out = {"name": name, "pair": ch["pair"], "channels": [ta, tb], "energy": x[order].tolist(),
               "energy_source": src, ta: a[order].tolist(), tb: b[order].tolist(),
               "xas": ((a + b) / 2)[order].tolist(), "dichroism": (b - a)[order].tolist(),
               "n_pixels": int(mask.sum()) if mask is not None else int(ny * nx)}
        note = _mixing_note(ch)
        if note:
            out["note"] = note
        return out

    def domain_width(self, name: str, kind: str = "pcd", var: str = "data", roi: dict | None = None,
                     smooth_px: float = 2.0) -> dict[str, Any]:
        """Mean domain width = area / domain-wall length inside ``roi`` (whole map if None). Walls are
        where the smoothed map crosses its median inside the region (Cauchy-Crofton length): right
        for labyrinths; straight stripes along an image axis read about a third high."""
        a = np.asarray(self.get_map(name, kind, var), float)
        a = ndimage.gaussian_filter(np.nan_to_num(a, nan=float(np.nanmedian(a))), smooth_px)
        mask = np.ones(a.shape, bool) if roi is None else roi_to_mask(a.shape[0], a.shape[1], roi)
        inner = ndimage.binary_erosion(mask, iterations=2)
        s = np.sign(a - np.median(a[mask]))
        fx = ((s[:, 1:] != s[:, :-1]) & inner[:, 1:] & inner[:, :-1]).sum()
        fy = ((s[1:] != s[:-1]) & inner[1:] & inner[:-1]).sum()
        upp = self._um_per_px(name)
        wall_px = (fx + fy) * np.pi / 4  # Cauchy-Crofton length estimate
        area_px = float(inner.sum())
        width_px = area_px / wall_px if wall_px else float("nan")
        out = {"name": name, "kind": kind, "var": var, "domain_width_px": round(float(width_px), 2),
               "period_px": round(float(2 * width_px), 2), "area_px": int(area_px)}
        if upp:
            out.update(domain_width_um=round(float(width_px * upp), 4), period_um=round(float(2 * width_px * upp), 4))
        _record(name, "domain width", {k: v for k, v in out.items() if k != "name"})
        return out
