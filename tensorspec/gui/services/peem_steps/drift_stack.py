"""PEEM service, XMCD-imaging steps: Drift correct -> stack -> asymmetry, one-call pcd / pld / pmld, hand-over to the stack tab.

Mixin of ``PeemService`` (gui/services/peem_service.py). Moved verbatim from the former
gui/services/peem_xmcd_service.py."""
from __future__ import annotations

import re
from typing import Any

import numpy as np

from tensorspec.core.data_models import TensorData
from tensorspec.core.peem.bg import resolve_energy
from tensorspec.core.peem import dichroism, drift, products, registration
from tensorspec.core.peem.engine import average_by_energy
from tensorspec.core.workspace import global_workspace
from tensorspec.gui.services.peem_steps.common import (_beamline_formula, _box, _drift_attrs, _frames_for_pol, _mask,
                                                       _read_edits, _record, _require_raw, _working_frames)


class PeemDriftStackSteps:
    # ------------------------------------------------------------- maps
    def _um_per_px(self, name: str) -> float | None:
        return (_require_raw(name).metadata or {}).get("um_per_px")

    def _write(self, name: str, kind: str, ds) -> None:
        if not global_workspace.write_analysis_data(name, kind, ds):
            raise ValueError(f"PEEM data '{name}' not found.")

    def compute_pcd(
        self,
        name: str,
        node: str = "raw",
        convention: str | None = None,
        register: bool = True,
        reg_roi: dict | None = None,
        crop_roi: dict | None = None,
    ) -> dict[str, Any]:
        """node: "raw", "processed" (drift-corrected or paired cube) or "separated" (/processed/CP|CM).

        register: drift correction (joint sub-pixel registration + edge refinement).
        convention None = the loaded beamline's default (metadata["default_convention"]).
        """
        if convention is None:
            convention = (_require_raw(name).metadata or {}).get("default_convention") or dichroism.UNIVERSAL_CONVENTION
        cp, _ = _frames_for_pol(name, "CP", "processed/CP" if node == "separated" else node)
        cm, _ = _frames_for_pol(name, "CM", "processed/CM" if node == "separated" else node)
        res = dichroism.pcd_map(cp, cm, convention=convention, register=register, box=_box(reg_roi))
        a = res["map"]
        if crop_roi is not None:
            a = np.where(_mask(a.shape, crop_roi), a, np.nan)
        ds = products.map_dataset(
            "pcd", a,
            extra={"mean_cp": res["mean_cp"], "mean_cm": res["mean_cm"]},
            attrs={
                "convention": convention,
                "formula": _beamline_formula(name, convention),
                "n_cp": int(cp.shape[0]), "n_cm": int(cm.shape[0]),
                "registered": bool(register),
                "cm_shift_dy": float(res["total_cm_shift"][0]),
                "cm_shift_dx": float(res["total_cm_shift"][1]),
                "refine_shift_dy": float(res["helicity_shift"][0]),
                "refine_shift_dx": float(res["helicity_shift"][1]),
                "source_node": node,
                **(_drift_attrs(a.shape, reg_roi, res["cp_shifts"], res["cm_shifts"]) if register else {}),
            },
            um_per_px=self._um_per_px(name),
            long_name="PEEM circular dichroism asymmetry",
        )
        self._write(name, "pcd", ds)
        return {"name": name, "kind": "pcd", **self._drift_summary(ds), **self._map_summary(a),
                **products.stats_table(ds)}

    # ------------------------------------------- step by step: drift -> stack -> asymmetry
    DRIFT_METHODS = drift.DRIFT_METHODS  # moved to core/peem/drift.py

    NEEDS_BOX = drift.NEEDS_BOX

    def drift_correct(
        self,
        name: str,
        reg_roi: dict | None = None,
        *,
        method: str | None = None,
        filt: str | None = None,
        ref_index: int = 0,
        search_radius: int = 20,
        max_shift: float | None = None,
        interp_factor: int = 4,
        transform: str = "translation",
        skip_pre: bool = False,
        pair: str = "xmcd",
    ) -> dict[str, Any]:
        """Step 1: align ALL frames (both polarizations together, one reference frame) after the
        stack clean-up edits, and write the drift-corrected frames to /processed.

        method  central / marquee / roi           sub-pixel methods of this suite
                igor_auto          autoImageCorrelation of each frame vs the reference
                igor_pair          driftCorrection: one shift between the mean images of the
                                   two polarizations (``pair``), applied to all B frames
                igor_marquee       marqueeDriftCorrection (interp_factor, skip_pre)
                igor_registration  marqueeImageRegistration (transform: translation /
                                   rotation / skew / isoscale)
        filt    none / smooth_edges / edges / highpass / find_edges / sharpen
        """
        method = method or ("marquee" if reg_roi else "central")
        if method not in self.DRIFT_METHODS:
            raise ValueError(f"method must be one of {sorted(self.DRIFT_METHODS)}")
        if method in self.NEEDS_BOX and reg_roi is None:
            raise ValueError("Draw the marquee / ROI box first (Draw box).")
        raw = _require_raw(name)
        frames, wpol, keep = _working_frames(name)
        n = len(frames)
        ref_index = int(ref_index)
        names = [str(v) for v in (raw.metadata or {}).get("frame_names", [])]
        names = [names[i] for i in keep] if len(names) == len(raw.value) else [""] * n
        skip = np.array(["pre" in nm.lower() for nm in names]) if skip_pre else None
        aligned, shifts, score, extra, box = drift.align_stack(
            frames, wpol, method, _box(reg_roi), filt=filt, ref_index=ref_index, search_radius=search_radius,
            max_shift=max_shift, interp_factor=interp_factor, transform=transform, skip=skip, pair=pair)
        md = dict(raw.metadata or {})
        md.update({
            "pol": wpol, "raw_frame_index": [int(i) for i in keep],
            "frame_names": names,
            "drift_corrected": True,
            "drift_method": method, "drift_method_label": self.DRIFT_METHODS[method],
            "drift_filter": filt or "none", "drift_reference_frame": ref_index,
            "drift_search_radius_px": int(search_radius) if method in ("roi", "ncc_roi") else None,
            "drift_max_shift_px": float(max_shift) if (max_shift and method in ("central", "marquee")) else None,
            "drift_dy_px": [float(v) for v in shifts[:, 0]], "drift_dx_px": [float(v) for v in shifts[:, 1]],
            "drift_score": [float(v) for v in score] if score is not None else None,
            "drift_box_y0y1x0x1": [int(v) for v in box],
            "drift_box_source": self.DRIFT_METHODS[method],
            "drift_extra": extra,
            "stack_edits_applied": _read_edits(name),
        })
        tensor = TensorData(value=aligned.astype(np.float32),
                            axes=[np.arange(n, dtype=float), raw.axes[1], raw.axes[2]], labels=raw.labels,
                            units=raw.units, data_type=raw.data_type, metadata=md)
        if not global_workspace.write_processed_data(name, tensor):
            raise ValueError(f"PEEM data '{name}' not found.")
        r = np.hypot(shifts[:, 0], shifts[:, 1])
        src = md.get("pol_source") or {}
        out = {"name": name, "step": "drift correct",
               "frames": f"{n} ({', '.join(f'{src.get(k, k)} {wpol.count(k)}' for k in dict.fromkeys(wpol))}) "
                         f"aligned together to frame {ref_index}",
               "method": self.DRIFT_METHODS[method], "filter": filt or "none",
               "box_y0y1x0x1": [int(v) for v in box],
               "drift_max_px": round(float(r.max()), 2),
               "drift_last_frame_px": [round(float(shifts[-1, 0]), 2), round(float(shifts[-1, 1]), 2)],
               **{k: v for k, v in extra.items() if not isinstance(v, list) or len(v) <= 4},
               "next": "stack (one image per polarization)"}
        if len(keep) < len(raw.value):
            out["excluded_frames"] = sorted(set(range(len(raw.value))) - set(int(i) for i in keep))
        if score is not None:
            worst = int(np.argmin(score))
            out["ncc_min"] = round(float(score.min()), 3)
            out["ncc_worst_frame"] = worst
            if score.min() < 0.5:
                out["warning"] = (f"frame {worst} matched the template poorly (NCC {score.min():.2f}): "
                                  "use a box with sharper features, a larger search radius or a smoothed-edge filter")
        _record(name, "drift correct", out)
        return out

    def manual_shift(self, name: str, frame: int, dy: float, dx: float) -> dict[str, Any]:
        """Igor manualDrift: shift one drift-corrected frame by (dy, dx) px (adds to its drift).
        Re-running the drift correction replaces it."""
        proc = global_workspace.pull_tensor_data(name, "processed")
        if proc is None or not (proc.metadata or {}).get("drift_corrected"):
            raise ValueError("Run the drift correction first; manual shifts adjust its result.")
        v = np.asarray(proc.value).copy()
        if not 0 <= int(frame) < len(v):
            raise ValueError(f"frame must be 0..{len(v) - 1}")
        v[int(frame)] = registration.apply_shift(v[int(frame)], float(dy), float(dx)).astype(v.dtype)
        md = dict(proc.metadata)
        dys, dxs = list(md["drift_dy_px"]), list(md["drift_dx_px"])
        dys[int(frame)] += float(dy)
        dxs[int(frame)] += float(dx)
        man = dict(md.get("manual_shifts") or {})
        prev = man.get(str(int(frame)), [0.0, 0.0])
        man[str(int(frame))] = [prev[0] + float(dy), prev[1] + float(dx)]
        md.update({"drift_dy_px": dys, "drift_dx_px": dxs, "manual_shifts": man})
        global_workspace.write_processed_data(name, TensorData(value=v, axes=proc.axes, labels=proc.labels,
                                                               units=proc.units, data_type=proc.data_type, metadata=md))
        out = {"name": name, "frame": int(frame), "total_drift_px": [round(dys[int(frame)], 2), round(dxs[int(frame)], 2)],
               "manual_shifts": man}
        _record(name, "manual shift", out)
        return out

    def stack(self, name: str, pair: str = "xmcd") -> dict[str, Any]:
        """Step 2: average the drift-corrected frames of each polarization into one image, then
        align the B image onto the A image on edge images (opposite polarizations show reversed
        magnetic contrast; edges are the same). Writes /analysis/stack: data = A + B (the two
        stacked images added), mean_<a> / mean_<b> (stacked images), sum_<a> / sum_<b> (sums of
        the frames)."""
        tags = {"xmcd": ("CP", "CM"), "xld": ("LH", "LV")}.get(pair)
        if tags is None:
            raise ValueError('pair must be "xmcd" or "xld"')
        proc = global_workspace.pull_tensor_data(name, "processed")
        pmd = (proc.metadata or {}) if proc is not None else {}
        if proc is None or not pmd.get("drift_corrected"):
            raise ValueError("Run step 1 (drift correct) first.")
        ta, tb = tags
        fa, _ = _frames_for_pol(name, ta, "processed")
        fb, _ = _frames_for_pol(name, tb, "processed")
        box = tuple(pmd["drift_box_y0y1x0x1"])
        mean_a, mean_b, refine = dichroism.stack_means(fa, fb, box)
        src = (_require_raw(name).metadata or {}).get("pol_source") or {}
        pol = [str(t) for t in pmd.get("pol", [])]
        ia = [i for i, t in enumerate(pol) if t == ta]
        ib = [i for i, t in enumerate(pol) if t == tb]
        dy, dx = np.asarray(pmd["drift_dy_px"]), np.asarray(pmd["drift_dx_px"])
        ds = products.map_dataset(
            "stack", mean_a + mean_b,
            extra={f"mean_{ta.lower()}": mean_a, f"mean_{tb.lower()}": mean_b,
                   # sums of the frames (Igor co-adds are sums); same alignment as the means
                   f"sum_{ta.lower()}": mean_a * len(ia), f"sum_{tb.lower()}": mean_b * len(ib)},
            attrs={"pair": pair, "channel_a": ta, "channel_b": tb,
                   "channel_a_name": src.get(ta, ta), "channel_b_name": src.get(tb, tb),
                   "n_a": len(ia), "n_b": len(ib), "source_node": "processed",
                   "refine_shift_dy": float(refine[0]), "refine_shift_dx": float(refine[1]),
                   "drift_dy_px": [float(v) for v in np.concatenate([dy[ia], dy[ib]])],
                   "drift_dx_px": [float(v) for v in np.concatenate([dx[ia], dx[ib]])],
                   "drift_n_a": len(ia), "drift_box_y0y1x0x1": list(box),
                   "drift_box_source": pmd["drift_box_source"]},
            um_per_px=self._um_per_px(name), long_name="stacked images, sum of both polarizations",
        )
        self._write(name, "stack", ds)
        out = {"name": name, "step": "2. stack", "pair": pair,
               "images": f"{src.get(ta, ta)}: {len(ia)} frames -> 1 image; {src.get(tb, tb)}: {len(ib)} frames -> 1 image",
               "B_onto_A_shift_px": [round(float(refine[0]), 3), round(float(refine[1]), 3)],
               "next": "3. asymmetry"}
        _record(name, "stack", out)
        return out

    def asymmetry(self, name: str, convention: str | None = None, crop_roi: dict | None = None) -> dict[str, Any]:
        """Step 3: asymmetry of the stacked images (/analysis/stack).

        XMCD -> /analysis/pcd: data = sign * (A-B)/(A+B), difference = sign * (A-B)
        (sign from convention; None = the beamline default). XLD -> /analysis/pld."""
        st = global_workspace.pull_analysis_data(name, "stack")
        if st is None:
            raise ValueError("Run step 2 (stack) first.")
        at = dict(st.attrs)
        ta, tb = at["channel_a"], at["channel_b"]
        a = np.asarray(st[f"mean_{ta.lower()}"].values, float)
        b = np.asarray(st[f"mean_{tb.lower()}"].values, float)
        if at["pair"] == "xmcd":
            if convention is None:
                convention = (_require_raw(name).metadata or {}).get("default_convention") or dichroism.UNIVERSAL_CONVENTION
            sign = dichroism.CONVENTIONS[convention]
            kind, formula = "pcd", _beamline_formula(name, convention)
        else:
            sign, kind, formula = -1.0, "pld", dichroism.XLD_FORMULA  # A = LH, B = LV -> (LV-LH)/(LV+LH)
        asym = sign * dichroism.asymmetry(a, b)
        diff = np.where(np.isfinite(asym), sign * (a - b), np.nan)
        if crop_roi is not None:
            m = _mask(asym.shape, crop_roi)
            asym, diff = np.where(m, asym, np.nan), np.where(m, diff, np.nan)
        keep = {k: v for k, v in at.items() if k.startswith(("drift_", "refine_shift", "n_", "source_node"))}
        ds = products.map_dataset(
            kind, asym, extra={"difference": diff, f"mean_{ta.lower()}": a, f"mean_{tb.lower()}": b},
            attrs={"formula": formula, "difference_formula": formula.split("/")[0],
                   "convention": convention or "", "registered": True,
                   "from_step": "drift correct -> stack -> asymmetry", **keep},
            um_per_px=self._um_per_px(name),
            long_name="PEEM circular dichroism asymmetry" if kind == "pcd" else "PEEM linear dichroism asymmetry",
        )
        self._write(name, kind, ds)
        out = {"name": name, "step": "3. asymmetry", "kind": kind, "formula": formula, **self._map_summary(asym)}
        _record(name, "asymmetry", out)
        return out

    subtract = asymmetry  # earlier name of step 3

    def compute_pld(self, name: str, node: str = "raw", register: bool = True,
                    reg_roi: dict | None = None, crop_roi: dict | None = None) -> dict[str, Any]:
        lh, _ = _frames_for_pol(name, "LH", "processed/LH" if node == "separated" else node)
        lv, _ = _frames_for_pol(name, "LV", "processed/LV" if node == "separated" else node)
        res = dichroism.pld_map(lh, lv, register=register, box=_box(reg_roi))
        a = res["map"]
        if crop_roi is not None:
            a = np.where(_mask(a.shape, crop_roi), a, np.nan)
        ds = products.map_dataset(
            "pld", a, extra={"mean_lh": res["mean_lh"], "mean_lv": res["mean_lv"]},
            attrs={"formula": dichroism.XLD_FORMULA, "n_lh": int(lh.shape[0]), "n_lv": int(lv.shape[0]),
                   "registered": bool(register), "source_node": node,
                   **(_drift_attrs(a.shape, reg_roi, res["lh_shifts"], res["lv_shifts"]) if register else {})},
            um_per_px=self._um_per_px(name), long_name="PEEM linear dichroism asymmetry",
        )
        self._write(name, "pld", ds)
        out = {"name": name, "kind": "pld", **self._drift_summary(ds), **self._map_summary(a)}
        _record(name, "pld", out)
        return out

    def compute_pmld(
        self,
        name: str,
        e_a: float,
        e_b: float,
        half_width: float = 0.0,
        node: str = "raw",
        register: bool = True,
        reg_roi: dict | None = None,
        crop_roi: dict | None = None,
    ) -> dict[str, Any]:
        """pld at E_A and E_B, then pmld = (pld_A - pld_B)/2, pmnld = (pld_A + pld_B)/2."""
        if node not in ("raw", "processed"):
            raise ValueError("pmld needs the frame cube ('raw' or drift-corrected 'processed')")
        raw = _require_raw(name)
        energy, src = resolve_energy(raw.value.shape[0], raw.metadata or {})
        if src == "index":
            raise ValueError("pmld needs a photon-energy column (load a Mono_S energy scan)")
        lh, i_lh = _frames_for_pol(name, "LH", node)
        lv, i_lv = _frames_for_pol(name, "LV", node)
        res = dichroism.pmld_pmnld_maps(lh, lv, energy[i_lh], energy[i_lv], e_a, e_b,
                                        half_width=half_width, register=register, box=_box(reg_roi))
        upp = self._um_per_px(name)
        crop = None if crop_roi is None else _mask(res["pmld"].shape, crop_roi)
        out = {}
        common = {"e_a": float(e_a), "e_b": float(e_b), "half_width": float(half_width),
                  "registered": bool(register), "source_node": node}
        for kind, arr, extra, formula in (
            ("pld", res["pld_a"], {"pld_b": res["pld_b"]}, f"{dichroism.XLD_FORMULA} at E_A (pld_b at E_B)"),
            ("pmld", res["pmld"], {}, "[pld(E_A) - pld(E_B)]/2"),
            ("pmnld", res["pmnld"], {}, "[pld(E_A) + pld(E_B)]/2"),
        ):
            if crop is not None:
                arr = np.where(crop, arr, np.nan)
                extra = {k: np.where(crop, v, np.nan) for k, v in extra.items()}
            ds = products.map_dataset(kind, arr, extra=extra, attrs={**common, "formula": formula},
                                      um_per_px=upp)
            self._write(name, kind, ds)
            out[kind] = self._map_summary(arr)
        out = {"name": name, "kinds": ["pld", "pmld", "pmnld"], "stats": out, **common}
        _record(name, "pmld", out)
        return out

    def compute_l2_ld(
        self,
        name: str,
        e_a: float,
        e_b: float,
        half_width: float = 0.05,
        pol: str | None = None,
        node: str = "raw",
        register: bool = True,
        reg_roi: dict | None = None,
        crop_roi: dict | None = None,
    ) -> dict[str, Any]:
        """Single-polarization L2 linear dichroism -> /analysis/pld.

        pld = (I(E_A) - I(E_B)) / (I(E_A) + I(E_B)), from an A/B/A image series
        (e.g. CrL2_A / CrL2_B) or an energy stack. Contains natural LD too; use
        ``split_ld_by_reference`` with a T > T_N dataset for pmld / pmnld.
        """
        if node not in ("raw", "processed"):
            raise ValueError("pld from L2 energies needs the frame cube ('raw' or 'processed')")
        raw = _require_raw(name)
        energy, src = resolve_energy(raw.value.shape[0], raw.metadata or {})
        if src == "index":
            raise ValueError("no photon-energy column in this dataset")
        tags = [str(p) for p in (raw.metadata or {}).get("pol", [])]
        if pol is None:
            present = sorted(set(tags))
            if len(present) != 1:
                raise ValueError(f"several polarizations {present}; choose one with pol=")
            pol = present[0]
        frames, idx = _frames_for_pol(name, pol, node)
        e = energy[idx]
        sel_a = dichroism.nearest_frames(e, e_a, half_width)
        sel_b = dichroism.nearest_frames(e, e_b, half_width)
        if np.intersect1d(sel_a, sel_b).size:
            raise ValueError("E_A and E_B windows overlap")
        j = dichroism.joint_means(frames[sel_a], frames[sel_b], register, _box(reg_roi))
        mean_a, mean_b = j["mean_a"], j["mean_b"]
        dy, dx = j["refine_shift"]
        a = dichroism.asymmetry(mean_a, mean_b)
        if crop_roi is not None:
            a = np.where(_mask(a.shape, crop_roi), a, np.nan)
        temp = (raw.metadata or {}).get("temperature_K")
        ds = products.map_dataset(
            "pld", a, extra={"mean_a": mean_a, "mean_b": mean_b},
            attrs={"method": "l2_ratio_single_pol", "formula": "(I_A-I_B)/(I_A+I_B)", "pol": pol,
                   "e_a": float(e_a), "e_b": float(e_b), "half_width": float(half_width),
                   "n_a": int(sel_a.size), "n_b": int(sel_b.size), "registered": bool(register),
                   "ab_shift_dy": dy, "ab_shift_dx": dx, "temperature_K": temp, "source_node": node},
            um_per_px=self._um_per_px(name), long_name="PEEM L2 linear dichroism (single polarization)",
        )
        self._write(name, "pld", ds)
        return {"name": name, "kind": "pld", "pol": pol, "n_a": int(sel_a.size), "n_b": int(sel_b.size),
                "temperature_K": temp, **self._map_summary(a)}

    def split_ld_by_reference(
        self,
        name: str,
        reference_name: str,
        register: bool = True,
        reg_roi: dict | None = None,
        reference_smooth_sigma: float = 0.0,
    ) -> dict[str, Any]:
        """pmnld = reference pld (T > T_N), pmld = pld - reference, both written to ``name``.

        The result carries ``reliable`` / ``warning`` / ``noise_ratio``: check them.
        """
        pld = self.get_map(name, "pld")
        ref = self.get_map(reference_name, "pld")
        a_attrs = self.get_product_attrs(name, "pld")
        r_attrs = self.get_product_attrs(reference_name, "pld")
        for key in ("method", "pol"):
            if a_attrs.get(key) != r_attrs.get(key):
                raise ValueError(f"pld maps differ in {key}: {a_attrs.get(key)} vs {r_attrs.get(key)}")
        for key in ("e_a", "e_b"):
            if abs(float(a_attrs.get(key, 0)) - float(r_attrs.get(key, 0))) > 0.3:
                raise ValueError(f"pld maps use different {key}: {a_attrs.get(key)} vs {r_attrs.get(key)}")
        def align(n: str) -> np.ndarray | None:
            ds = global_workspace.pull_analysis_data(n, "pld")
            for var in ("mean_a", "mean_lh"):
                if var in ds:
                    return np.asarray(ds[var].values)
            return None

        res = dichroism.split_by_reference(pld, ref, register=register, box=_box(reg_roi),
                                           align_image=align(name), reference_align_image=align(reference_name),
                                           reference_smooth_sigma=reference_smooth_sigma)
        common = {"method": "temperature_reference", "reference": reference_name,
                  "temperature_K": a_attrs.get("temperature_K"),
                  "reference_temperature_K": r_attrs.get("temperature_K"),
                  "reference_shift_dy": res["reference_shift"][0],
                  "reference_shift_dx": res["reference_shift"][1],
                  "reference_smooth_sigma": float(reference_smooth_sigma),
                  "noise_ratio": res["noise_ratio"], "reference_correlation": res["correlation"],
                  "reliable": res["reliable"], "warning": res["warning"]}
        upp = self._um_per_px(name)
        self._write(name, "pmld", products.map_dataset(
            "pmld", res["pmld"], attrs={**common, "formula": "pld - pld_ref(T>T_N)"}, um_per_px=upp))
        self._write(name, "pmnld", products.map_dataset(
            "pmnld", res["pmnld"], attrs={**common, "formula": "pld_ref(T>T_N), registered"}, um_per_px=upp))
        return {"name": name, "kinds": ["pmld", "pmnld"], **common,
                "stats": {"pmld": self._map_summary(res["pmld"]), "pmnld": self._map_summary(res["pmnld"])}}

    @staticmethod
    def _drift_summary(ds) -> dict[str, Any]:
        dy, dx = ds.attrs.get("drift_dy_px"), ds.attrs.get("drift_dx_px")
        if dy is None:
            return {"drift": "off"}
        r = np.hypot(np.asarray(dy), np.asarray(dx))
        return {"drift_box": ds.attrs["drift_box_source"], "drift_box_y0y1x0x1": list(ds.attrs["drift_box_y0y1x0x1"]),
                "drift_max_px": round(float(r.max()), 2), "drift_last_frame_px": [round(float(dy[-1]), 2), round(float(dx[-1]), 2)]}

    def to_peem_stack(self, name: str, pair: str = "xmcd", energy_bin_eV: float | None = None) -> dict[str, Any]:
        """Hand the drift-corrected result of tab 1 to the PEEM-stack tab (background, sum rule).

        The drift-corrected frames (/processed, step 1) are grouped by photon energy and
        averaged per polarization, giving one A and one B image per energy. They become a new
        dataset "<name>_<pair>" whose raw frames are A(E1), B(E1), A(E2), B(E2), ... with the
        energy axis attached. Frames are grouped in energy bins of ``energy_bin_eV`` (None = the
        larger median step of the two polarizations, so separate CP / CM sweeps on different energy
        points still pair up); bins without both polarizations are dropped (core/peem/engine.
        average_by_energy). The new dataset is already paired (/processed) and separated
        (/processed/<A>, <B>), which is the input the PEEM-stack background and sum-rule steps use.
        """
        tags = {"xmcd": ("CP", "CM"), "xld": ("LH", "LV")}.get(pair)
        if tags is None:
            raise ValueError('pair must be "xmcd" or "xld"')
        raw = _require_raw(name)
        proc = global_workspace.pull_tensor_data(name, "processed")
        pmd = (proc.metadata or {}) if proc is not None else {}
        if proc is None or not pmd.get("drift_corrected"):
            raise ValueError("Run step 2 (drift correction) in the XMCD-imaging tab first.")
        frames = np.asarray(proc.value, dtype=np.float32)
        pol = [str(t) for t in pmd.get("pol", [])]
        energy, e_src = resolve_energy(len(raw.value), raw.metadata or {})
        keep = np.asarray(pmd.get("raw_frame_index", range(len(pol))), dtype=int)
        energy = np.asarray(energy, float)[keep]  # energies of the frames kept after clean-up
        has_e = e_src != "index"
        ta, tb = tags
        stack, pol_out, points, counts, energy_bin_eV = average_by_energy(
            frames, pol, energy if has_e else None, tags, energy_bin_eV)
        src = (raw.metadata or {}).get("pol_source") or {}
        md = {k: v for k, v in (raw.metadata or {}).items()
              if k in ("format", "beamline", "default_convention", "pol_source", "fov_um", "um_per_px",
                       "temperature_K", "loader", "i0_normalized")}
        md.update({
            "pol": pol_out,
            "frame_names": [f"{src.get(t, t)}@{e:.2f}eV" for e in points for t in (ta, tb)],
            "derived_from": name,
            "derived_by": "XMCD-imaging tab: drift-corrected frames averaged per polarization and energy",
            "drift_method": pmd.get("drift_method_label"),
            "frames_per_point": counts,
            "beamline_table": {"columns": ["energy"], "series": {"energy": points}} if has_e else None,
            "I0": None, "csv_attached": False,
        })
        tensor = TensorData(value=stack, axes=[np.arange(len(stack), dtype=float), raw.axes[1], raw.axes[2]],
                            labels=raw.labels, units=raw.units, data_type="Experimental PEEM", metadata=md)
        label = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{name}_{pair}")[:64]
        global_workspace.push_spectroscopy_data(label, tensor)
        self.pair(label, "CP_CM" if pair == "xmcd" else "LH_LV")
        sep = self.separate(label)
        out = {"name": label, "from": name, "pair": pair,
               "channels": [src.get(t, t) for t in (ta, tb)], "separated": sep["channels"],
               "n_energies": len(points), "n_frames": int(len(stack)), "pol_summary": {ta: len(points), tb: len(points)},
               "energy_range_eV": [min(points), max(points)] if has_e else None,
               "frames_per_point": counts,
               "energy_bin_eV": round(energy_bin_eV, 4) if has_e else None}
        if len(points) < 3:
            out["note"] = ("single photon energy: background subtraction and the sum rule need an energy "
                           "scan (several energies); use the XMCD-imaging tab's asymmetry for single-energy maps")
        return out

    def get_drift(self, name: str, kind: str = "pcd") -> dict[str, Any] | None:
        """Per-frame drift stored with a pcd / pld / stack map, or with /processed (kind="processed").
        None if there is none (e.g. computed without registration)."""
        if kind == "processed":
            proc = global_workspace.pull_tensor_data(name, "processed")
            at = dict((proc.metadata or {}) if proc is not None else {})
            if "drift_dy_px" in at:
                at.setdefault("drift_n_a", len(at["drift_dy_px"]))
                src = at.get("pol_source") or {}
                at["_pol_names"] = [src.get(str(t), str(t)) for t in at.get("pol", [])]
        else:
            ds = global_workspace.pull_analysis_data(name, kind)
            at = {} if ds is None else ds.attrs
        if "drift_dy_px" not in at:
            return None
        return {"pol": at.get("_pol_names"),
                "score": None if at.get("drift_score") is None else np.asarray(at["drift_score"], float),
                "method": at.get("drift_method_label") or at.get("drift_box_source"),
                "dy": np.asarray(at["drift_dy_px"], float), "dx": np.asarray(at["drift_dx_px"], float),
                "n_a": int(at["drift_n_a"]), "box": list(at["drift_box_y0y1x0x1"]), "source": at["drift_box_source"]}
