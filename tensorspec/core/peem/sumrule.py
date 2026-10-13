"""PEEM XMCD sum-rule engine (Thole/Carra classic form).

Dichroism dμ = μ+ − μ−; sum sμ = μ+ + μ−.

Integrals (trapezoid on energy axis):
  p = ∫_{L3} dμ dE
  q = ∫_{L3∪L2} dμ dE  (OR mask on energy; overlap counted once)
  r = ∫_{r_window} sμ dE

Moments (⟨T_z⟩=0 form):
  m_orb = −(4/3) · nₕ · q / r
  m_spin_plus_dipole = nₕ · (6p − 4q) / r

Require |r| > eps; nₕ > 0.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tensorspec.core.peem.bg import fit_background

_R_EPS = np.finfo(float).eps


def apply_i0(
    spectrum: np.ndarray, i0: np.ndarray | float | None
) -> tuple[np.ndarray, bool]:
    """Normalize spectrum by I0 when valid; return (spectrum, i0_applied)."""
    spectrum = np.asarray(spectrum, dtype=float)
    if i0 is None:
        return spectrum, False

    if np.isscalar(i0) or isinstance(i0, (int, float)):
        i0_val = float(i0)
        if not np.isfinite(i0_val) or i0_val == 0.0:
            return spectrum, False
        return spectrum / i0_val, True

    i0_arr = np.asarray(i0, dtype=float)
    if i0_arr.shape != spectrum.shape:
        return spectrum, False
    if not np.all(np.isfinite(i0_arr)) or np.any(i0_arr == 0.0):
        return spectrum, False
    return spectrum / i0_arr, True


def _trapz_window(
    energy: np.ndarray, y: np.ndarray, e0: float, e1: float
) -> float:
    lo, hi = (e0, e1) if e0 <= e1 else (e1, e0)
    mask = (energy >= lo) & (energy <= hi)
    e_win = energy[mask]
    y_win = y[mask]
    if e_win.size < 2:
        return 0.0
    trapz = getattr(np, "trapezoid", np.trapz)
    return float(trapz(y_win, e_win))


def _trapz_union_windows(
    energy: np.ndarray, y: np.ndarray, *windows: tuple[float, float]
) -> float:
    """Trapezoid over L3∪L2 via OR mask; overlap integrated once."""
    mask = np.zeros(energy.shape, dtype=bool)
    for e0, e1 in windows:
        lo, hi = (e0, e1) if e0 <= e1 else (e1, e0)
        mask |= (energy >= lo) & (energy <= hi)
    indices = np.flatnonzero(mask)
    if indices.size < 2:
        return 0.0

    trapz = getattr(np, "trapezoid", np.trapz)
    total = 0.0
    seg_start = 0
    for i in range(1, indices.size):
        if indices[i] != indices[i - 1] + 1:
            seg = indices[seg_start:i]
            if seg.size >= 2:
                total += float(trapz(y[seg], energy[seg]))
            seg_start = i
    seg = indices[seg_start:]
    if seg.size >= 2:
        total += float(trapz(y[seg], energy[seg]))
    return total


def integrate_windows(
    energy: np.ndarray,
    mu_plus: np.ndarray,
    mu_minus: np.ndarray,
    *,
    l3: tuple[float, float],
    l2: tuple[float, float],
    r_win: tuple[float, float],
) -> dict[str, float]:
    """Return p, q, r integrals over L3, L3∪L2 (OR mask), and r windows."""
    energy = np.asarray(energy, dtype=float)
    mu_plus = np.asarray(mu_plus, dtype=float)
    mu_minus = np.asarray(mu_minus, dtype=float)
    d_mu = mu_plus - mu_minus
    s_mu = mu_plus + mu_minus

    p = _trapz_window(energy, d_mu, l3[0], l3[1])
    q = _trapz_union_windows(energy, d_mu, l3, l2)
    r = _trapz_window(energy, s_mu, r_win[0], r_win[1])
    return {"p": p, "q": q, "r": r}


def moments(p: float, q: float, r: float, nh: float) -> dict[str, float]:
    """Compute m_orb and m_spin_plus_dipole from integrals."""
    if nh <= 0:
        raise ValueError("nh must be positive")
    if abs(r) <= _R_EPS:
        raise ValueError("r integral too small for moment calculation")

    m_orb = -(4.0 / 3.0) * nh * q / r
    m_spin_plus_dipole = nh * (6.0 * p - 4.0 * q) / r
    return {"m_orb": float(m_orb), "m_spin_plus_dipole": float(m_spin_plus_dipole)}


def _jitter_endpoint(
    value: float, delta: float, rng: np.random.Generator, e_min: float, e_max: float
) -> float:
    return float(np.clip(rng.uniform(value - delta, value + delta), e_min, e_max))


def _jitter_window(
    window: tuple[float, float],
    delta: float,
    rng: np.random.Generator,
    e_min: float,
    e_max: float,
) -> tuple[float, float]:
    return (
        _jitter_endpoint(window[0], delta, rng, e_min, e_max),
        _jitter_endpoint(window[1], delta, rng, e_min, e_max),
    )


def ensemble_sumrule(
    energy: np.ndarray,
    mu_plus: np.ndarray,
    mu_minus: np.ndarray,
    *,
    l3: tuple[float, float],
    l2: tuple[float, float],
    r_win: tuple[float, float],
    nh: float,
    window_delta: float,
    window_n: int,
    bg_e0: float | None = None,
    bg_e1: float | None = None,
    bg_method: str = "linear",
    bg_post_e0: float | None = None,
    bg_post_e1: float | None = None,
    bg_delta: float = 0.0,
    bg_n: int = 1,
    seed: int = 0,
) -> dict[str, Any]:
    """Dual ensemble: L-window jitter and optional BG pre-edge re-subtraction."""
    if window_n < 1:
        raise ValueError("window_n must be at least 1")
    if window_delta < 0:
        raise ValueError("window_delta must be non-negative")
    if bg_n < 1:
        raise ValueError("bg_n must be at least 1")
    if bg_delta < 0:
        raise ValueError("bg_delta must be non-negative")

    energy = np.asarray(energy, dtype=float)
    mu_plus = np.asarray(mu_plus, dtype=float)
    mu_minus = np.asarray(mu_minus, dtype=float)
    e_min, e_max = float(energy.min()), float(energy.max())
    rng = np.random.default_rng(seed)

    bg_active = bg_e0 is not None and bg_e1 is not None
    bg_method_norm = str(bg_method).strip().lower()
    two_step_bg = bg_method_norm == "two_step"
    if two_step_bg and bg_active and (bg_post_e0 is None or bg_post_e1 is None):
        raise ValueError("two_step background requires bg_post_e0 and bg_post_e1")

    p_samples: list[float] = []
    q_samples: list[float] = []
    r_samples: list[float] = []
    m_orb_samples: list[float] = []
    m_spin_samples: list[float] = []
    n_valid_bg = 0

    for _ in range(window_n):
        l3_j = _jitter_window(l3, window_delta, rng, e_min, e_max)
        l2_j = _jitter_window(l2, window_delta, rng, e_min, e_max)
        r_j = _jitter_window(r_win, window_delta, rng, e_min, e_max)

        bg_iters = range(bg_n) if bg_active else range(1)
        for _ in bg_iters:
            mp = mu_plus.copy()
            mm = mu_minus.copy()
            bg_ok = False
            if bg_active:
                e0_j = _jitter_endpoint(bg_e0, bg_delta, rng, e_min, e_max)
                e1_j = _jitter_endpoint(bg_e1, bg_delta, rng, e_min, e_max)
                fit_kwargs: dict[str, Any] = {"e0": e0_j, "e1": e1_j}
                if two_step_bg:
                    fit_kwargs["post_e0"] = _jitter_endpoint(
                        bg_post_e0, bg_delta, rng, e_min, e_max
                    )
                    fit_kwargs["post_e1"] = _jitter_endpoint(
                        bg_post_e1, bg_delta, rng, e_min, e_max
                    )
                try:
                    mp = mp - fit_background(
                        bg_method_norm, energy, mp, **fit_kwargs
                    )["bg"]
                    mm = mm - fit_background(
                        bg_method_norm, energy, mm, **fit_kwargs
                    )["bg"]
                    bg_ok = True
                except ValueError:
                    continue

            try:
                ints = integrate_windows(
                    energy, mp, mm, l3=l3_j, l2=l2_j, r_win=r_j
                )
                m = moments(ints["p"], ints["q"], ints["r"], nh)
            except ValueError:
                continue

            if bg_ok:
                n_valid_bg += 1
            p_samples.append(ints["p"])
            q_samples.append(ints["q"])
            r_samples.append(ints["r"])
            m_orb_samples.append(m["m_orb"])
            m_spin_samples.append(m["m_spin_plus_dipole"])

    if not p_samples:
        raise ValueError("ensemble produced no valid samples")

    def _mean_std(vals: list[float]) -> tuple[float, float]:
        arr = np.asarray(vals, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=0))

    p_mean, p_std = _mean_std(p_samples)
    q_mean, q_std = _mean_std(q_samples)
    r_mean, r_std = _mean_std(r_samples)
    m_orb_mean, m_orb_std = _mean_std(m_orb_samples)
    m_spin_mean, m_spin_std = _mean_std(m_spin_samples)

    return {
        "p_mean": p_mean,
        "p_std": p_std,
        "q_mean": q_mean,
        "q_std": q_std,
        "r_mean": r_mean,
        "r_std": r_std,
        "m_orb_mean": m_orb_mean,
        "m_orb_std": m_orb_std,
        "m_spin_plus_dipole_mean": m_spin_mean,
        "m_spin_plus_dipole_std": m_spin_std,
        "n_valid": len(p_samples),
        "n_valid_bg": n_valid_bg,
    }


def pick_source_kind(available_nodes: list[str] | set[str], tags: tuple[str, str]) -> str:
    """Resolve source kind: bg pair → separated → paired 4D."""
    nodes = set(available_nodes)
    t0, t1 = tags
    if f"processed/{t0}_bg" in nodes and f"processed/{t1}_bg" in nodes:
        return "bg"
    if f"processed/{t0}" in nodes and f"processed/{t1}" in nodes:
        return "separated"
    if "processed" in nodes:
        return "paired"
    raise ValueError("no sum-rule source pair found")


def analysis_dataset(
    energy: np.ndarray,
    mu_plus: np.ndarray,
    mu_minus: np.ndarray,
    integrals: dict[str, float],
    integral_stds: dict[str, float],
    moment_vals: dict[str, float],
    moment_stds: dict[str, float],
    ensemble: dict[str, Any],
    *,
    nh: float,
    l3: tuple[float, float],
    l2: tuple[float, float],
    r_win: tuple[float, float],
    i0_applied: bool,
    source_kind: str,
    tags: tuple[str, str],
    window_delta: float | None = None,
    window_n: int | None = None,
    bg_e0: float | None = None,
    bg_e1: float | None = None,
    bg_delta: float | None = None,
    bg_n: int | None = None,
    seed: int = 0,
    use_roi: bool = False,
    roi: dict | None = None,
):
    """Build xarray Dataset for /analysis/sumrule."""
    import xarray as xr

    energy = np.asarray(energy, dtype=float)
    mu_plus = np.asarray(mu_plus, dtype=float)
    mu_minus = np.asarray(mu_minus, dtype=float)
    d_mu = mu_plus - mu_minus

    attrs: dict[str, Any] = {
        "nh": float(nh),
        "l3_lo": float(l3[0]),
        "l3_hi": float(l3[1]),
        "l2_lo": float(l2[0]),
        "l2_hi": float(l2[1]),
        "r_lo": float(r_win[0]),
        "r_hi": float(r_win[1]),
        "p": float(integrals["p"]),
        "q": float(integrals["q"]),
        "r": float(integrals["r"]),
        "p_std": float(integral_stds["p"]),
        "q_std": float(integral_stds["q"]),
        "r_std": float(integral_stds["r"]),
        "m_orb": float(moment_vals["m_orb"]),
        "m_spin_plus_dipole": float(moment_vals["m_spin_plus_dipole"]),
        "m_orb_std": float(moment_stds["m_orb"]),
        "m_spin_plus_dipole_std": float(moment_stds["m_spin_plus_dipole"]),
        "i0_applied": bool(i0_applied),
        "source_kind": source_kind,
        "tag_plus": tags[0],
        "tag_minus": tags[1],
        "ensemble_n_valid": int(ensemble["n_valid"]),
        "ensemble_n_valid_bg": int(ensemble.get("n_valid_bg", 0)),
        "seed": int(seed),
        "use_roi": bool(use_roi),
    }
    if window_delta is not None:
        attrs["window_delta"] = float(window_delta)
    if window_n is not None:
        attrs["window_n"] = int(window_n)
    if bg_e0 is not None:
        attrs["bg_e0"] = float(bg_e0)
    if bg_e1 is not None:
        attrs["bg_e1"] = float(bg_e1)
    if bg_delta is not None:
        attrs["bg_delta"] = float(bg_delta)
    if bg_n is not None:
        attrs["bg_n"] = int(bg_n)
    if roi is not None:
        attrs["roi"] = roi

    return xr.Dataset(
        data_vars={
            "mu_plus": (("energy",), mu_plus),
            "mu_minus": (("energy",), mu_minus),
            "dichroism": (("energy",), d_mu),
        },
        coords={"energy": energy},
        attrs=attrs,
    )
