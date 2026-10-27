"""Dichroism maps from PEEM image pairs and energy stacks.

Output kinds (written to ``/analysis/<kind>`` by :mod:`products`):

Signs used for every beamline: XMCD = the UE49 convention (neg - pos)/(neg + pos)
= (CR - CL)/(CR + CL); XLD = (LV - LH)/(LV + LH). The internal "CP" channel is UE49 pos,
BNL CL and ALS EPU +0.9; "CM" is UE49 neg, BNL CR and ALS -0.9.

pcd    circular dichroism asymmetry  A_c = s * (I_CP - I_CM) / (I_CP + I_CM), s = -1
       (``convention="ue49"`` = ``"cm_minus_cp"``, the default everywhere);
       ``"cp_minus_cm"`` (s = +1) gives the opposite sign for comparisons.
pld    linear dichroism asymmetry    A_l(E) = (I_LV - I_LH) / (I_LV + I_LH)
pmld   magnetic linear dichroism, the part of A_l that reverses between the
       two L2 multiplet peaks E_A, E_B:  [A_l(E_A) - A_l(E_B)] / 2
pmnld  non-magnetic (natural / crystal-field) linear dichroism, the part
       common to both peaks:            [A_l(E_A) + A_l(E_B)] / 2

``l2_ratio_map`` is the single-polarization XMLD contrast
(I(E_A) - I(E_B)) / (I(E_A) + I(E_B)) used when only one polarization was taken
(UE49 CrL2_A / CrL2_B series). It contains natural LD as well; ``split_by_reference``
separates the two with a map taken above the ordering temperature:
pmnld = reference (T > T_N), pmld = pld - reference.
"""
from __future__ import annotations

import re
from collections import OrderedDict

import numpy as np

from tensorspec.core.peem import registration

CONVENTIONS = {"ue49": -1.0, "cm_minus_cp": -1.0, "cp_minus_cm": 1.0}
UNIVERSAL_CONVENTION = "ue49"  # (neg - pos)/(neg + pos) = (CR - CL)/(CR + CL), every beamline
XMCD_FORMULA = "(CM-CP)/(CM+CP)"  # (neg-pos)/(neg+pos)
XLD_FORMULA = "(LV-LH)/(LV+LH)"

# Opposite helicities / polarizations show reversed magnetic contrast, so plain
# cross-correlation between them can lock onto the anti-correlated domain
# pattern. Gradient magnitude is invariant to that sign flip (Igor's
# driftCorrection also defaults to "find edges").
CROSS_POL_FILTER = "edges"
FRAME_FILTER = "smooth"  # frame-to-frame drift: smoothed 1 px (UE49 TPX3 without flat field keeps a pixel pattern)


def asymmetry(a: np.ndarray, b: np.ndarray, min_sum: float = 0.0,
              min_rel_sum: float = 0.02) -> np.ndarray:
    """(a - b) / (a + b); NaN where a + b <= max(min_sum, min_rel_sum * median(a + b)).

    The relative floor blanks the unilluminated area outside the PEEM field of
    view, where a few counts would otherwise give |A| = 1.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    s = a + b
    finite = s[np.isfinite(s)]
    if min_rel_sum and finite.size:
        min_sum = max(min_sum, min_rel_sum * float(np.median(finite)))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (a - b) / s
    out[~np.isfinite(out) | (s <= min_sum)] = np.nan
    return out


def xld_asymmetry(lh: np.ndarray, lv: np.ndarray, min_sum: float = 0.0) -> np.ndarray:
    """Universal XLD sign: (LV - LH)/(LV + LH)."""
    return asymmetry(lv, lh, min_sum=min_sum)


def _central_box(shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    ny, nx = shape[-2:]
    return ny // 4, 3 * ny // 4, nx // 4, 3 * nx // 4


# Opt-in bound on registration shifts (fraction of the image size). Off by default:
# real UE49 triplets drift smoothly by up to ~28 px (June 2025 X_0004-0006), which a
# 5 % bound (25.7 px) would clamp.
MAX_SHIFT_FRAC = None


def joint_means(stack_a: np.ndarray, stack_b: np.ndarray, register: bool = True,
                box: tuple[int, int, int, int] | None = None,
                max_shift_frac: float | None = MAX_SHIFT_FRAC) -> dict:
    """Co-add two polarization (or energy) channels on one common reference.

    All frames of both channels are registered to the first frame of channel A
    (plain correlation in ``box``, default the central half, as Igor's stack
    drift correction does), averaged per channel, and the B mean is then
    refined onto the A mean with edge images. On the 20250613 Cr XMCD repeats
    this gave the most reproducible maps (repeat correlation 0.78 vs 0.70 for
    per-channel co-adds and 0.52 for the Igor maps).

    Known limit: across opposite polarizations the reversed domain pattern gives a
    false correlation peak half a domain period away. It wins only when the
    topographic contrast is weak compared with the dichroic contrast (synthetic
    periodic domains with 6 % topography; real UE49/BNL images carry 10-30 %
    flake contrast). For such samples pass ``max_shift_frac`` (e.g. 0.05) to bound
    the search below half a domain period.
    """
    a = np.asarray(stack_a, dtype=float)
    b = np.asarray(stack_b, dtype=float)
    a = a[None] if a.ndim == 2 else a
    b = b[None] if b.ndim == 2 else b
    if a.shape[1:] != b.shape[1:]:
        raise ValueError(f"channel image sizes differ: {a.shape[1:]} vs {b.shape[1:]}")
    shifts = np.zeros((a.shape[0] + b.shape[0], 2))
    refine = (0.0, 0.0)
    max_shift = max(2.0, max_shift_frac * min(a.shape[1:])) if max_shift_frac else None
    if register:
        box = box or _central_box(a.shape)
        aligned, shifts = registration.register_stack(np.concatenate([a, b]), 0, box=box, filt=FRAME_FILTER,
                                                      max_shift=max_shift)
        a, b = aligned[: a.shape[0]], aligned[a.shape[0]:]
    if register:
        mean_a, mean_b, refine = stack_means(a, b, box, max_shift)
    else:
        mean_a, mean_b = a.mean(axis=0), b.mean(axis=0)
    return {"mean_a": mean_a, "mean_b": mean_b, "frame_shifts": shifts, "refine_shift": refine,
            "max_shift_px": max_shift}


def stack_means(frames_a: np.ndarray, frames_b: np.ndarray, box: tuple[int, int, int, int],
                max_shift: float | None = None) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Average already drift-corrected frames per polarization into one image each, then refine
    the B image onto the A image on edge images (opposite polarizations show reversed magnetic
    contrast; edges are the same). Returns (mean_a, mean_b aligned, (dy, dx) applied to B)."""
    mean_a, mean_b = np.asarray(frames_a, float).mean(axis=0), np.asarray(frames_b, float).mean(axis=0)
    refine = registration.estimate_shift(mean_a, mean_b, box=box, filt=CROSS_POL_FILTER, max_shift=max_shift)
    return mean_a, registration.apply_shift(mean_b, *refine), refine


def _mean_image(x: np.ndarray, register: bool, box) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        return x, np.zeros((1, 2))
    if x.ndim != 3:
        raise ValueError("expected an image (y, x) or a stack (n, y, x)")
    return registration.smart_coadd(x, register=register, box=box)


def pcd_map(
    i_cp: np.ndarray,
    i_cm: np.ndarray,
    convention: str = "ue49",
    register: bool = True,
    box: tuple[int, int, int, int] | None = None,
    min_sum: float = 0.0,
) -> dict:
    """Circular dichroism map from CP / CM images or stacks.

    Stacks are co-added with drift correction (means, not Igor's sums, so
    unequal frame counts do not bias A). The CM mean is then registered onto
    the CP mean at sub-pixel precision before forming the asymmetry, since a
    residual offset between helicities creates false contrast at domain walls.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {sorted(CONVENTIONS)}")
    j = joint_means(i_cp, i_cm, register, box)
    cp, cm = j["mean_a"], j["mean_b"]
    n_cp = 1 if np.ndim(i_cp) == 2 else len(i_cp)
    a = CONVENTIONS[convention] * asymmetry(cp, cm, min_sum=min_sum)
    return {
        "map": a,
        "mean_cp": cp,
        "mean_cm": cm,
        "helicity_shift": j["refine_shift"],
        # mean CM displacement relative to CP: common-reference shift + refinement
        "total_cm_shift": tuple(np.mean(j["frame_shifts"][n_cp:], axis=0) - np.mean(j["frame_shifts"][:n_cp], axis=0)
                                + np.asarray(j["refine_shift"])),
        "cp_shifts": j["frame_shifts"][:n_cp],
        "cm_shifts": j["frame_shifts"][n_cp:],
        "convention": convention,
    }


def pld_map(
    i_lh: np.ndarray,
    i_lv: np.ndarray,
    register: bool = True,
    box: tuple[int, int, int, int] | None = None,
    min_sum: float = 0.0,
) -> dict:
    """Linear dichroism asymmetry (LV - LH)/(LV + LH) at one energy (universal XLD sign)."""
    j = joint_means(i_lh, i_lv, register, box)
    lh, lv = j["mean_a"], j["mean_b"]
    n_lh = 1 if np.ndim(i_lh) == 2 else len(i_lh)
    return {"map": xld_asymmetry(lh, lv, min_sum=min_sum), "mean_lh": lh, "mean_lv": lv,
            "polarization_shift": j["refine_shift"],
            "lh_shifts": j["frame_shifts"][:n_lh], "lv_shifts": j["frame_shifts"][n_lh:]}


def nearest_frames(energy: np.ndarray, e0: float, half_width: float = 0.0) -> np.ndarray:
    """Indices with |E - e0| <= half_width, or the single nearest frame."""
    energy = np.asarray(energy, dtype=float)
    idx = np.flatnonzero(np.abs(energy - e0) <= half_width) if half_width > 0 else np.array([], int)
    if idx.size == 0:
        idx = np.array([int(np.nanargmin(np.abs(energy - e0)))])
    return idx


def energy_image(stack: np.ndarray, energy: np.ndarray, e0: float, half_width: float = 0.0) -> np.ndarray:
    """Mean image over frames within ``half_width`` of ``e0``."""
    stack = np.asarray(stack, dtype=float)
    return stack[nearest_frames(energy, e0, half_width)].mean(axis=0)


def pmld_pmnld_maps(
    lh_stack: np.ndarray,
    lv_stack: np.ndarray,
    energy_lh: np.ndarray,
    energy_lv: np.ndarray,
    e_a: float,
    e_b: float,
    half_width: float = 0.0,
    register: bool = True,
    box: tuple[int, int, int, int] | None = None,
) -> dict:
    """Split linear dichroism at two multiplet energies into magnetic / non-magnetic parts.

    Every image is registered to the LH image at ``e_a`` first.
    """
    imgs = {
        ("LH", "A"): energy_image(lh_stack, energy_lh, e_a, half_width),
        ("LV", "A"): energy_image(lv_stack, energy_lv, e_a, half_width),
        ("LH", "B"): energy_image(lh_stack, energy_lh, e_b, half_width),
        ("LV", "B"): energy_image(lv_stack, energy_lv, e_b, half_width),
    }
    ref = imgs[("LH", "A")]
    shifts = {}
    if register:
        for key, img in imgs.items():
            if key == ("LH", "A"):
                shifts[key] = (0.0, 0.0)
                continue
            dy, dx = registration.estimate_shift(ref, img, box=box, filt=CROSS_POL_FILTER)
            shifts[key] = (dy, dx)
            imgs[key] = registration.apply_shift(img, dy, dx)
    ld_a = xld_asymmetry(imgs[("LH", "A")], imgs[("LV", "A")])
    ld_b = xld_asymmetry(imgs[("LH", "B")], imgs[("LV", "B")])
    return {
        "pld_a": ld_a,
        "pld_b": ld_b,
        "pmld": 0.5 * (ld_a - ld_b),
        "pmnld": 0.5 * (ld_a + ld_b),
        "e_a": float(e_a),
        "e_b": float(e_b),
        "shifts": {f"{p}_{e}": v for (p, e), v in shifts.items()},
    }


def l2_ratio_map(stack: np.ndarray, energy: np.ndarray, e_a: float, e_b: float,
                 half_width: float = 0.0, register: bool = True) -> np.ndarray:
    """Single-polarization XMLD contrast (I_A - I_B)/(I_A + I_B)."""
    ia = energy_image(stack, energy, e_a, half_width)
    ib = energy_image(stack, energy, e_b, half_width)
    if register:
        ib = registration.apply_shift(ib, *registration.estimate_shift(ia, ib, filt=CROSS_POL_FILTER))
    return asymmetry(ia, ib)


MAX_REFERENCE_SHIFT_FRAC = 0.15
MAX_REFERENCE_NOISE_RATIO = 1.5


def nan_gaussian(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing that ignores NaN pixels and keeps them NaN."""
    from scipy import ndimage

    a = np.asarray(a, dtype=float)
    if not sigma:
        return a.copy()
    m = np.isfinite(a)
    w = ndimage.gaussian_filter(m.astype(float), float(sigma))
    n = ndimage.gaussian_filter(np.where(m, a, 0.0), float(sigma))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(m, n / w, np.nan)


def _noise(a: np.ndarray) -> float:
    """Pixel noise estimate: robust std of the difference to a 1-px-smoothed copy."""
    d = a - nan_gaussian(a, 1.0)
    d = d[np.isfinite(d)]
    return float(1.4826 * np.median(np.abs(d - np.median(d)))) if d.size else float("nan")


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return float("nan")
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den else float("nan")


def split_by_reference(pld: np.ndarray, reference: np.ndarray, register: bool = True,
                       box: tuple[int, int, int, int] | None = None,
                       align_image: np.ndarray | None = None,
                       reference_align_image: np.ndarray | None = None,
                       reference_smooth_sigma: float = 0.0) -> dict:
    """Magnetic / natural split of an LD map using a T > T_N reference of the same area.

    The shift is measured on intensity images (``align_image`` /
    ``reference_align_image``, e.g. the E_A means), not on the dichroism maps:
    a few-frame reference LD map is mostly noise and does not register. Shifts
    larger than 15 % of the image are refused as a failed registration.

    Quality fields: ``noise_ratio`` (reference / map pixel noise) and
    ``correlation`` (map vs registered reference). ``reliable`` is False when
    the reference is noisier than the map by more than 1.5x, in which case
    pmld is dominated by reference noise; smooth the reference
    (``reference_smooth_sigma``) or acquire more T > T_N frames.
    """
    pld = np.asarray(pld, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if ref.shape != pld.shape:
        raise ValueError(f"reference {ref.shape} and map {pld.shape} differ; same FOV and binning needed")
    dy = dx = 0.0
    if register:
        fill = lambda a: np.where(np.isfinite(a), a, np.nanmedian(a))  # noqa: E731
        a = fill(np.asarray(align_image if align_image is not None else pld, dtype=float))
        b = fill(np.asarray(reference_align_image if reference_align_image is not None else ref, dtype=float))
        dy, dx = registration.estimate_shift(a, b, box=box or _central_box(a.shape))
        limit = MAX_REFERENCE_SHIFT_FRAC * min(pld.shape)
        if abs(dy) > limit or abs(dx) > limit:
            raise ValueError(f"reference registration failed (shift {dy:.1f}, {dx:.1f} px); "
                             "check that both datasets show the same area, or set a registration ROI")
        ref = registration.apply_shift(fill(ref), dy, dx)
        ref[~np.isfinite(pld)] = np.nan
    # pixel noise of the raw reference, scaled by the white-noise reduction of a
    # 2-D Gaussian (1 / (2 sqrt(pi) sigma)); measuring after smoothing would only
    # see the removed high-frequency part and overstate the gain
    ref_noise = _noise(ref)
    if reference_smooth_sigma and reference_smooth_sigma >= 0.5:
        ref_noise /= 2.0 * np.sqrt(np.pi) * float(reference_smooth_sigma)
    ref = nan_gaussian(ref, reference_smooth_sigma)
    map_noise = _noise(pld)
    noise_ratio = ref_noise / map_noise if map_noise > 0 else float("nan")
    reliable = bool(np.isfinite(noise_ratio) and noise_ratio <= MAX_REFERENCE_NOISE_RATIO)
    return {
        "pmld": pld - ref,
        "pmnld": ref,
        "reference_shift": (dy, dx),
        "noise_ratio": noise_ratio,
        "correlation": _corr(pld, ref),
        "reliable": reliable,
        "warning": "" if reliable else (
            f"reference is {noise_ratio:.1f}x noisier than the map; pmld is dominated by reference "
            "noise. Smooth the reference or take more frames above T_N."),
    }


def vector_map(xmcd_0: np.ndarray, xmcd_90: np.ndarray) -> dict:
    """In-plane magnetization angle and magnitude from two azimuths (Igor Calc2dmag)."""
    a0 = np.asarray(xmcd_0, dtype=float)
    a90 = np.asarray(xmcd_90, dtype=float)
    return {"angle_rad": np.arctan2(a90, a0), "magnitude": np.hypot(a0, a90)}


def group_by_key(names: list[str], strip: int = 5) -> "OrderedDict[str, list[int]]":
    """Igor autoCoaddXMCD grouping: key = name minus its last ``strip`` chars."""
    groups: "OrderedDict[str, list[int]]" = OrderedDict()
    for i, n in enumerate(names):
        key = n[:-strip] if len(n) > strip else n
        key = re.sub(r"[_\-]+$", "", key)
        groups.setdefault(key, []).append(i)
    return groups


def coadd_by_key(stack: np.ndarray, names: list[str], register: bool = True,
                 strip: int = 5) -> dict[str, np.ndarray]:
    """Mean image per filename key (e.g. Cr_pos / Cr_neg)."""
    stack = np.asarray(stack, dtype=float)
    return {k: _mean_image(stack[idx], register, None)[0]
            for k, idx in group_by_key(names, strip).items()}
