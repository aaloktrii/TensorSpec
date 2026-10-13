"""Does the matrix element matter for the temperature dependence?  Three variants of the same 3 XRD structures:
   bare  = spectral function of the bands at (k_par, k_z(84 eV)), no matrix element, no Fermi cutoff
   fd10K = full one-step matrix elements, Fermi-Dirac fixed at 10 K   (structure + ME change only)
   orig  = full matrix elements, Fermi-Dirac at the folder temperature (structure + ME + FD softening)
Numbers: relative change of each cut between temperatures, |I_T - I_100| / |I_100| (L2) and 1 - corr, over E <= -0.02 eV.
Figure: deflector 0, 4 geometries x [100 K | 300 K | 300-100 difference] for the three variants."""
import json, os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, TwoSlopeNorm

ROOT = os.path.expanduser("~/mnt/TensorSpec_GUI/scratch")
VAR = {"bare": "vte2_temperature_dependent_chinook_bare", "fd10K": "vte2_temperature_dependent_chinook_fd10K",
       "orig": "vte2_temperature_dependent_chinook"}
VLAB = {"bare": "bands only (no matrix element, no Fermi cutoff)", "fd10K": "full matrix elements, Fermi–Dirac fixed 10 K",
        "orig": "full matrix elements, Fermi–Dirac = T (original batch)"}
TS = {100: "deliverable_VTe2_CDW_XRD100K_NM_6x23x9_MLWF2000", 180: "deliverable_VTe2_CDW_XRD180K_NM_6x23x9_MLWF2000",
      300: "deliverable_VTe2_CDW_XRD300K_NM_6x23x9_MLWF2000"}
CFGS = [("th55_ph0", "slit horizontal, beam 55°"), ("th20_ph0", "slit horizontal, beam 20°"),
        ("th55_ph90", "slit vertical, beam 55°"), ("th20_ph90", "slit vertical, beam 20°")]
DEFL = ["m10p3", "m5", "0", "5", "10p3"]
EMAX = -0.02


def load(var, T, cfg, d):
    p = f"{ROOT}/{VAR[var]}/{TS[T]}/{cfg}/dispersion/{cfg}_defl_{d}_viewer.npz"
    z = np.load(p, allow_pickle=True)
    return z["kx"], z["E"], z["intensity"][:, 0, :].astype(np.float64), z["metadata"].item()


def metrics(I0, I1, E, emax=EMAX):
    m = E <= emax
    a, b = I0[:, m].ravel(), I1[:, m].ravel()
    rel = np.linalg.norm(b - a) / np.linalg.norm(a)
    corr = np.corrcoef(a, b)[0, 1]
    return rel, 1.0 - corr


def best_shift(I0, I1, E, emax=EMAX):
    """rigid energy shift (meV) of I1 that best matches I0, and the residual 1-corr after it (E <= emax window)"""
    dE = float(E[1] - E[0])
    best = (None, 9.0)
    for n in range(-15, 16):
        I1s = np.roll(I1, n, axis=1)
        lo, hi = max(0, n), I1.shape[1] + min(0, n)
        m = np.zeros(E.size, bool); m[lo:hi] = True; m &= (E <= emax)
        c = 1.0 - np.corrcoef(I0[:, m].ravel(), I1s[:, m].ravel())[0, 1]
        if c < best[1]:
            best = (n * dE * 1e3, c)
    return best


rows = []
for var in VAR:
    for cfg, _ in CFGS:
        for d in DEFL:
            kx, E, I100, md = load(var, 100, cfg, d)
            _, _, I180, _ = load(var, 180, cfg, d)
            _, _, I300, _ = load(var, 300, cfg, d)
            r18, c18 = metrics(I100, I180, E)
            r30, c30 = metrics(I100, I300, E)
            r30f, c30f = metrics(I100, I300, E, emax=1.0)   # whole window incl. the Fermi edge
            sh, csh = best_shift(I100, I300, E)
            rows.append(dict(variant=var, cfg=cfg, defl=d, rel_180=r18, rel_300=r30, dcorr_180=c18, dcorr_300=c30,
                             rel_300_full=r30f, dcorr_300_full=c30f, best_shift_meV=sh, dcorr_300_after_shift=csh,
                             me_mode=md.get("matrix_element_mode"), fd_T=md.get("fd_temperature_K"), fermi_edge=md.get("fermi_edge_applied")))
out_dir = f"{ROOT}/vte2_temperature_dependent_chinook/summary"
os.makedirs(out_dir, exist_ok=True)
with open(f"{out_dir}/T_dependence_bare_vs_fullME_metrics.json", "w") as fh:
    json.dump(rows, fh, indent=1)

print("relative L2 change of the cut (E <= -0.02 eV), 100 K -> 300 K  [100 K -> 180 K]   ;  1 - correlation(100 K, 300 K)")
print(f"{'geometry':10s} {'defl':6s} | {'bare':>22s} | {'full ME, FD 10 K':>22s} | {'full ME, FD = T':>22s}")
for cfg, _ in CFGS:
    for d in DEFL:
        cells = []
        for var in VAR:
            r = next(x for x in rows if x["variant"] == var and x["cfg"] == cfg and x["defl"] == d)
            cells.append(f"{100*r['rel_300']:5.1f}% [{100*r['rel_180']:4.1f}%] {r['dcorr_300']:.4f}")
        print(f"{cfg:10s} {d:6s} | " + " | ".join(cells))
print()
for var in VAR:
    sel = [x for x in rows if x["variant"] == var]
    print(f"{var:6s}: mean rel change 100->300 K = {100*np.mean([x['rel_300'] for x in sel]):.1f}%  "
          f"(min {100*min(x['rel_300'] for x in sel):.1f}, max {100*max(x['rel_300'] for x in sel):.1f});  "
          f"mean 1-corr = {np.mean([x['dcorr_300'] for x in sel]):.4f};  mode = {sel[0]['me_mode']}")
    print(f"        whole window incl. Fermi edge: rel {100*np.mean([x['rel_300_full'] for x in sel]):.1f}%, 1-corr {np.mean([x['dcorr_300_full'] for x in sel]):.4f}")
    print(f"        best rigid shift of the 300 K cut: {np.mean([x['best_shift_meV'] for x in sel]):+.1f} meV "
          f"(range {min(x['best_shift_meV'] for x in sel):+.0f}..{max(x['best_shift_meV'] for x in sel):+.0f}); "
          f"1-corr after the shift {np.mean([x['dcorr_300_after_shift'] for x in sel]):.4f}")

# ---- figure: deflector 0
fig, axes = plt.subplots(4, 9, figsize=(27, 15))
plt.subplots_adjust(left=0.03, right=0.99, top=0.91, bottom=0.04, wspace=0.12, hspace=0.3)
for ir, (cfg, clab) in enumerate(CFGS):
    for iv, var in enumerate(VAR):
        kx, E, I100, _ = load(var, 100, cfg, "0")
        _, _, I300, _ = load(var, 300, cfg, "0")
        ext = [kx[0], kx[-1], E[0], E[-1]]
        vmax = max(I100.max(), I300.max())
        diff = I300 - I100
        r = next(x for x in rows if x["variant"] == var and x["cfg"] == cfg and x["defl"] == "0")
        for ic, (img, ttl, kind) in enumerate([(I100, f"{var}: 100 K", "I"), (I300, f"{var}: 300 K", "I"),
                                               (diff, f"{var}: 300 K − 100 K   (ΔI/I = {100*r['rel_300']:.1f}%)", "D")]):
            ax = axes[ir, 3 * iv + ic]
            if kind == "I":
                ax.imshow(img.T, origin="lower", aspect="auto", extent=ext, cmap="magma", norm=PowerNorm(0.5, vmin=0, vmax=vmax))
            else:
                a = float(np.abs(diff).max())
                ax.imshow(img.T, origin="lower", aspect="auto", extent=ext, cmap="RdBu_r", norm=TwoSlopeNorm(0, -a, a))
            ax.set_title(ttl, fontsize=9)
            ax.tick_params(labelsize=7)
            if ic == 0 and iv == 0:
                ax.set_ylabel(f"{clab}\nE − E$_F$ (eV)", fontsize=9)
            if ir == 3:
                ax.set_xlabel("k$_{slit}$ (Å$^{-1}$)", fontsize=8)
fig.suptitle("VTe$_2$ bulk (−2 0 1), XRD 100 K vs 300 K structures, deflector 0°, hν 84 eV LH — does the matrix element carry temperature dependence?\n"
             "left: " + VLAB["bare"] + "    middle: " + VLAB["fd10K"] + "    right: " + VLAB["orig"]
             + "\n(intensity panels: one colour scale per pair; difference panels: symmetric blue/red scale per panel; ΔI/I = L2 change for E ≤ −0.02 eV)",
             fontsize=11)
fp = f"{out_dir}/compare_T_dependence_bare_vs_fullME_vs_FD.png"
fig.savefig(fp, dpi=100)
print("saved", fp)
