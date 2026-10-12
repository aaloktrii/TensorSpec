#!/usr/bin/env python3
"""
Export the VTe2 temperature-dependent Chinook dispersion cuts to MATLAB .mat files,
in the same layout as scripts/sprkkr/to_matlab.py (Sandy's MATLAB convention):

PER-GEOMETRY FILES  (16 = 4 structures x 4 geometries)      <out>/vte2_<T>_<cfg>_chinook.mat
    data.value : intensity [n_angle x n_defl x n_energy] = [200 x 5 x 200]
    data.x     : lab slit angle, deg          ROW vector 1 x 200
    data.y     : E - E_F, eV                  ROW vector 1 x 200
    data.z     : deflector angle, deg         ROW vector 1 x 5   (-10.3 -5 0 5 10.3)
    data.k_slit, data.k_defl : the same axes in 1/Angstrom (k = 0.512316*sqrt(hv-W)*sin(angle))
    data.info  : struct (geometry, physics knobs, provenance, row/column labels)

FERMI-SURFACE FILES (16 + 4 bundles)
    <out>/vte2_<T>_<cfg>_fermisurface_chinook.mat          -> data.value [n_slit x n_defl] = [200 x 301] at E = E_F,
        data.x slit angle (deg), data.y deflector angle (deg), data.k_slit, data.k_defl (1/A), data.info
    <out>/vte2_<T>_all4geometries_fermisurface_chinook.mat -> fs.panel(1..4) in figure order (same as the rows below)

PER-STRUCTURE BUNDLES (4)                                   <out>/vte2_<T>_all4geometries_chinook.mat
    grid.row(i)  i = 1..4 in figure order:
        1 slit horizontal, beam 55 deg   (th55_ph0)
        2 slit horizontal, beam 20 deg   (th20_ph0)
        3 slit vertical,   beam 55 deg   (th55_ph90)
        4 slit vertical,   beam 20 deg   (th20_ph90)
      each with the same value/x/y/z/k_slit/k_defl/info as above + row_label
    grid.column_labels, grid.deflectors_deg, grid.structure, grid.fd_temperature_K, grid.hv_eV ...
    grid.howto : one-line MATLAB recipe to draw panel (i, j)

Usage (on the Mac, from the TensorSpec_GUI folder):
    TensorSpec_env/bin/python scripts/chinook/to_matlab_vte2_temperature.py \
        --root scratch/vte2_temperature_dependent_chinook --out scratch/vte2_temperature_dependent_chinook/matlab
"""
import argparse
import glob
import math
import os
import re

import numpy as np

K_PER_SQRT_EV = 0.512316
CFGS = [  # (tag, row label, slit orientation, beam incidence)
    ("th55_ph0",  "slit horizontal, beam 55 deg", "horizontal", 55.0),
    ("th20_ph0",  "slit horizontal, beam 20 deg", "horizontal", 20.0),
    ("th55_ph90", "slit vertical, beam 55 deg",   "vertical",   55.0),
    ("th20_ph90", "slit vertical, beam 20 deg",   "vertical",   20.0),
]
DEFL = [-10.3, -5.0, 0.0, 5.0, 10.3]


def defl_tag(d):
    return f"{d:g}".replace("-", "m").replace(".", "p")


def short_name(folder):
    m = re.search(r"XRD(\d+)K", folder)
    if m:
        return f"XRD{m.group(1)}K"
    return "relaxed" if "_CDW_NM_" in folder else folder


def _clean(v):
    if isinstance(v, (list, tuple)):
        if v and all(isinstance(x, (int, float, np.floating, np.integer)) for x in v):
            return np.asarray(v, dtype=float)
        return [str(x) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if v is None:
        return ""
    return v


def load_geometry(root, folder, cfg):
    """Stack the 5 deflector cuts of one geometry -> value[angle, defl, energy] + axes + info."""
    cubes, info = [], {}
    x_deg = y_eV = None
    for d in DEFL:
        p = os.path.join(root, folder, cfg, "dispersion", f"{cfg}_defl_{defl_tag(d)}_viewer.npz")
        z = np.load(p, allow_pickle=True)
        I = np.asarray(z["intensity"], dtype=float)[:, 0, :]          # (angle, energy)
        meta = z["metadata"].item()
        th = np.asarray(meta["theta_lab_deg"], dtype=float)
        E = np.asarray(z["E"], dtype=float)
        if x_deg is None:
            x_deg, y_eV = th, E
            for k, v in meta.items():
                if k in ("theta_lab_deg", "phi_lab_deg", "deflector_deg", "wall_s", "kind"):
                    continue
                info[str(k)] = _clean(v)
        else:
            assert np.allclose(th, x_deg) and np.allclose(E, y_eV), p
        cubes.append(I)
    value = np.stack(cubes, axis=1)                                     # (angle, defl, energy)
    hv, wf = float(info["hv_eV"]), float(info["ework_eV"])
    k0 = K_PER_SQRT_EV * math.sqrt(hv - wf)
    k_slit = k0 * np.sin(np.radians(x_deg))
    k_defl = k0 * np.sin(np.radians(np.asarray(DEFL)))
    return value, x_deg, y_eV, np.asarray(DEFL, dtype=float), k_slit, k_defl, info


def geometry_struct(value, x, y, zdeg, k_slit, k_defl, info, cfg, row_label, slit_orient, beam):
    inf = dict(info)
    inf.update({
        "config": cfg,
        "row_label": row_label,
        "slit_orientation": slit_orient,
        "beam_incidence_deg": float(beam),
        "axes": "value[angle, defl, energy]; x = slit angle (deg), y = E - E_F (eV), z = deflector (deg)",
        "x_units": "deg (lab slit angle)", "y_units": "eV (E - E_F)", "z_units": "deg (deflector angle)",
        "k_units": "1/Angstrom (k_slit, k_defl = 0.512316*sqrt(hv - W)*sin(angle))",
        "colour_in_png": "one scale per row: vmax = 99.5th percentile of the row, sqrt (PowerNorm gamma 0.5)",
    })
    return {
        "value": value,
        "x": x.reshape(1, -1), "y": y.reshape(1, -1), "z": zdeg.reshape(1, -1),
        "k_slit": k_slit.reshape(1, -1), "k_defl": k_defl.reshape(1, -1),
        "sumI": float(value.sum()),
        "row_label": row_label,
        "config": cfg,
        "info": inf,
    }


def load_fermi_surface(root, folder, cfg):
    """E_F map of one geometry -> value[slit angle, deflector angle] + axes + info."""
    p = os.path.join(root, folder, cfg, "iso_energy", f"{cfg}_isoE_EF_viewer.npz")
    z = np.load(p, allow_pickle=True)
    I = np.asarray(z["intensity"], dtype=float)[:, :, 0]               # (slit angle, deflector)
    meta = z["metadata"].item()
    x_deg = np.asarray(meta["theta_lab_deg"], dtype=float)
    y_deg = np.asarray(meta["phi_lab_deg"], dtype=float)
    info = {str(k): _clean(v) for k, v in meta.items()
            if k not in ("theta_lab_deg", "phi_lab_deg", "kind")}
    hv, wf = float(info["hv_eV"]), float(info["ework_eV"])
    k0 = K_PER_SQRT_EV * math.sqrt(hv - wf)
    return I, x_deg, y_deg, k0 * np.sin(np.radians(x_deg)), k0 * np.sin(np.radians(y_deg)), info


def fermi_struct(value, x, y, k_slit, k_defl, info, cfg, row_label, slit_orient, beam):
    inf = dict(info)
    inf.update({
        "config": cfg, "row_label": row_label, "slit_orientation": slit_orient,
        "beam_incidence_deg": float(beam),
        "axes": "value[slit angle, deflector angle] at E = E_F; x = slit angle (deg), y = deflector (deg)",
        "x_units": "deg (lab slit angle)", "y_units": "deg (deflector angle)",
        "k_units": "1/Angstrom (k_slit, k_defl = 0.512316*sqrt(hv - W)*sin(angle))",
        "energy_note": (f"constant-energy slice at E - E_F = {float(info.get('iso_energy_eV', 0.0)):+.3f} eV, "
                        f"computed on the window {info.get('iso_window_eV')} [E0, E1, nE] so Lorentzian tails are included"),
        "colour_in_png": "each panel scaled to its own 99.5th percentile, sqrt (PowerNorm gamma 0.5)",
    })
    return {
        "value": value,
        "x": x.reshape(1, -1), "y": y.reshape(1, -1),
        "k_slit": k_slit.reshape(1, -1), "k_defl": k_defl.reshape(1, -1),
        "energy_eV": float(info.get("iso_energy_eV", 0.0)),
        "sumI": float(value.sum()),
        "row_label": row_label, "config": cfg, "info": inf,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="scratch/vte2_temperature_dependent_chinook")
    ap.add_argument("--out", required=True, help="output folder for the .mat files")
    args = ap.parse_args()
    from scipy.io import savemat

    os.makedirs(args.out, exist_ok=True)
    folders = sorted(d for d in os.listdir(args.root)
                     if os.path.isdir(os.path.join(args.root, d, "th55_ph0", "dispersion")))

    def tkey(f):
        m = re.search(r"XRD(\d+)K", f)
        return (0, int(m.group(1))) if m else (1, f)

    # ---------------- Fermi-surface (E_F) maps ----------------
    for folder in sorted(folders, key=tkey):
        name = short_name(folder)
        panels = []
        for cfg, row_label, slit_orient, beam in CFGS:
            p = os.path.join(args.root, folder, cfg, "iso_energy", f"{cfg}_isoE_EF_viewer.npz")
            if not os.path.isfile(p):
                print(f"[skip] no E_F map for {name} {cfg}")
                continue
            value, x, y, ks, kd, info = load_fermi_surface(args.root, folder, cfg)
            s = fermi_struct(value, x, y, ks, kd, info, cfg, row_label, slit_orient, beam)
            panels.append(s)
            out = os.path.join(args.out, f"vte2_{name}_{cfg}_fermisurface_chinook.mat")
            savemat(out, {"data": s}, do_compression=True)
            print(f"[mat] {out}  value[{'x'.join(map(str, value.shape))}]  sumI={s['sumI']:.3e}")
        if panels:
            fs = {
                "structure": name, "structure_folder": folder,
                "temperature_K": panels[0]["info"].get("temperature_K", ""),
                "fd_temperature_K": panels[0]["info"].get("fd_temperature_K", ""),
                "hv_eV": panels[0]["info"]["hv_eV"], "ework_eV": panels[0]["info"]["ework_eV"],
                "inner_potential_eV": panels[0]["info"]["inner_potential_eV"],
                "polarization": panels[0]["info"]["polarization"],
                "energy_eV": panels[0]["energy_eV"],
                "panel_labels": [s["row_label"] for s in panels],
                "panel_configs": [s["config"] for s in panels],
                "x": panels[0]["x"], "y": panels[0]["y"],
                "k_slit": panels[0]["k_slit"], "k_defl": panels[0]["k_defl"],
                "panel": np.array([(s["value"], s["x"], s["y"], s["k_slit"], s["k_defl"], s["energy_eV"],
                                    s["sumI"], s["row_label"], s["config"], s["info"]) for s in panels],
                                  dtype=[("value", object), ("x", object), ("y", object), ("k_slit", object),
                                         ("k_defl", object), ("energy_eV", object), ("sumI", object),
                                         ("row_label", object), ("config", object), ("info", object)]),
                "howto": ("panel i: imagesc(fs.panel(i).k_slit, fs.panel(i).k_defl, fs.panel(i).value'); axis xy image; "
                          "colormap(hot); caxis([0 prctile(fs.panel(i).value(:),99.5)]) -- "
                          "i=1..4 = slit horizontal beam 55/20, slit vertical beam 55/20 (same order as the summary PNG columns)"),
            }
            out = os.path.join(args.out, f"vte2_{name}_all4geometries_fermisurface_chinook.mat")
            savemat(out, {"fs": fs}, do_compression=True)
            print(f"[mat] {out}  ({len(panels)} E_F maps bundle)")

    # ---------------- dispersion cuts ----------------
    for folder in sorted(folders, key=tkey):
        name = short_name(folder)
        rows = []
        for cfg, row_label, slit_orient, beam in CFGS:
            value, x, y, zdeg, ks, kd, info = load_geometry(args.root, folder, cfg)
            s = geometry_struct(value, x, y, zdeg, ks, kd, info, cfg, row_label, slit_orient, beam)
            rows.append(s)
            out = os.path.join(args.out, f"vte2_{name}_{cfg}_chinook.mat")
            savemat(out, {"data": s}, do_compression=True)
            print(f"[mat] {out}  value[{'x'.join(map(str, value.shape))}]  sumI={s['sumI']:.3e}")
        fd_T = rows[0]["info"].get("fd_temperature_K", "")
        grid = {
            "structure": name,
            "structure_folder": folder,
            "temperature_K": rows[0]["info"].get("temperature_K", ""),
            "fd_temperature_K": fd_T,
            "hv_eV": rows[0]["info"]["hv_eV"], "ework_eV": rows[0]["info"]["ework_eV"],
            "inner_potential_eV": rows[0]["info"]["inner_potential_eV"],
            "polarization": rows[0]["info"]["polarization"],
            "deflectors_deg": np.asarray(DEFL, dtype=float).reshape(1, -1),
            "k_defl_1perA": rows[0]["k_defl"],
            "row_labels": [r["row_label"] for r in rows],
            "row_configs": [r["config"] for r in rows],
            "column_labels": [f"deflector {d:+.1f} deg" for d in DEFL],
            "x": rows[0]["x"], "y": rows[0]["y"], "k_slit": rows[0]["k_slit"],
            "row": np.array([(r["value"], r["x"], r["y"], r["z"], r["k_slit"], r["k_defl"],
                              r["sumI"], r["row_label"], r["config"], r["info"]) for r in rows],
                            dtype=[("value", object), ("x", object), ("y", object), ("z", object),
                                   ("k_slit", object), ("k_defl", object), ("sumI", object),
                                   ("row_label", object), ("config", object), ("info", object)]),
            "howto": ("panel (i,j): imagesc(grid.row(i).x, grid.row(i).y, squeeze(grid.row(i).value(:,j,:))'); "
                      "axis xy; colormap(hot); caxis([0 prctile(grid.row(i).value(:),99.5)]) -- "
                      "rows i=1..4 = slit horizontal beam 55/20, slit vertical beam 55/20; columns j=1..5 = deflector -10.3,-5,0,5,10.3 deg"),
        }
        out = os.path.join(args.out, f"vte2_{name}_all4geometries_chinook.mat")
        savemat(out, {"grid": grid}, do_compression=True)
        print(f"[mat] {out}  (4 rows x 5 deflectors bundle, FD T = {fd_T} K)")


if __name__ == "__main__":
    main()
