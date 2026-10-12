"""Jiang et al., Science 393, eadu1550 (2026) homobilayer twist."""
from __future__ import annotations

import math

import numpy as np
from pymatgen.core import Lattice, Structure

_Q = {
    ("hexagonal", "Γ"): "triangular",
    ("hexagonal", "K"): "honeycomb",
    ("hexagonal", "M"): "kagome",
    ("hexagonal", "nHSP"): "nHSP",
    ("square", "Γ"): "square",
    ("square", "M"): "square",
    ("square", "X"): "nested square",
    ("square", "nHSP"): "nHSP",
    ("rectangular", "HSP"): "rectangular",
    ("rectangular", "nHSP"): "nHSP",
    ("oblique", "HSP"): "oblique",
    ("oblique", "nHSP"): "nHSP",
}


def classify_bravais(ab: np.ndarray) -> str:
    ab = np.asarray(ab, dtype=float)
    a1, a2 = ab[0], ab[1]
    la, lb = float(np.linalg.norm(a1)), float(np.linalg.norm(a2))
    mean = 0.5 * (la + lb)
    rel = abs(la - lb) / mean if mean > 0 else 1.0
    cosang = float(np.dot(a1, a2) / (la * lb))
    cosang = max(-1.0, min(1.0, cosang))
    ang = math.degrees(math.acos(cosang))
    if rel < 1e-3 and min(abs(ang - 60.0), abs(ang - 120.0)) < 1.0:
        return "hexagonal"
    if rel < 1e-3 and abs(ang - 90.0) < 1.0:
        return "square"
    if rel >= 1e-3 and abs(ang - 90.0) < 1.0:
        return "rectangular"
    return "oblique"


def q_lattice_label(bravais: str, valley: str) -> str:
    key = (bravais, valley)
    if key not in _Q:
        raise ValueError(f"No Q lattice for bravais={bravais!r} valley={valley!r}.")
    return _Q[key]


def hexagonal_pair(m: int, n: int) -> tuple[float, np.ndarray]:
    if m <= n or n < 0:
        raise ValueError("hexagonal pair needs m > n >= 0.")
    N = m * m + m * n + n * n
    cos_th = (m * m + n * n + 4 * m * n) / (2.0 * N)
    theta = math.degrees(math.acos(cos_th))
    S = np.array([[m, -n], [n, m + n]], dtype=int)
    return theta, S


def square_pair(m: int, n: int) -> tuple[float, np.ndarray]:
    if m <= n or n <= 0:
        raise ValueError("square pair needs m > n > 0.")
    theta = math.degrees(2.0 * math.atan2(n, m))
    S = np.array([[m, n], [-n, m]], dtype=int)
    return theta, S


def _rot(theta_deg: float) -> np.ndarray:
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]], dtype=float)


def _residual(ab: np.ndarray, moire: np.ndarray, theta_deg: float) -> float:
    coeffs = (moire @ _rot(theta_deg)) @ np.linalg.inv(ab)
    return float(np.max(np.abs(coeffs - np.round(coeffs))))


_RESIDUAL_TOL = 1e-6
_MAX_SURVIVORS = 400


def _search_integer_S(ab: np.ndarray, theta_deg: float, max_cells: int) -> dict:
    """Find S (det in 1..max_cells) with rows of S@ab rotating into the layer lattice.

    A row vector v=(i,j)@ab is usable only if its rotation by theta lies in the layer
    lattice. Filter the O(B^2) coefficient vectors first, then pair the shortest
    survivors. Total iterations stay well under 200k at max_cells=2000.
    """
    B = int(math.isqrt(max_cells)) + 1
    rng = np.arange(-B, B + 1)
    ii, jj = np.meshgrid(rng, rng, indexing="ij")
    coef = np.stack([ii.ravel(), jj.ravel()], axis=1)
    coef = coef[np.any(coef != 0, axis=1)]
    vecs = coef.astype(float) @ ab
    back = (vecs @ _rot(theta_deg)) @ np.linalg.inv(ab)
    res = np.max(np.abs(back - np.round(back)), axis=1)
    keep = res < _RESIDUAL_TOL
    coef, vecs, res = coef[keep], vecs[keep], res[keep]
    if len(coef) < 2:
        raise ValueError(
            f"No unstrained commensurate cell for θ={theta_deg}° "
            f"within max_cells={max_cells} (coefficient bound {B})."
        )
    order = np.argsort(np.linalg.norm(vecs, axis=1), kind="stable")[:_MAX_SURVIVORS]
    coef, res = coef[order], res[order]
    x, y = coef[:, 0], coef[:, 1]
    det = x[:, None] * y[None, :] - y[:, None] * x[None, :]
    ok = (det > 0) & (det <= max_cells)
    if not np.any(ok):
        raise ValueError(
            f"No unstrained commensurate cell for θ={theta_deg}° "
            f"within max_cells={max_cells} (coefficient bound {B})."
        )
    pair_res = np.maximum(res[:, None], res[None, :])
    cand = np.argwhere(ok)
    keys = [
        (int(det[p, q]), float(pair_res[p, q]), p + q, p) for p, q in cand
    ]
    best_idx = min(range(len(keys)), key=keys.__getitem__)
    p, q = cand[best_idx]
    S = np.array([coef[p], coef[q]], dtype=int)
    n_cells = int(det[p, q])
    return {
        "bravais": classify_bravais(ab),
        "theta_used": float(theta_deg),
        "S": S,
        "n_cells": n_cells,
        "residual": _residual(ab, S.astype(float) @ ab, theta_deg),
    }


def _require_commensurate(residual: float, theta_deg: float) -> None:
    if residual >= _RESIDUAL_TOL:
        raise ValueError(
            f"No unstrained commensurate cell for θ={theta_deg}° "
            f"(residual {residual:.3e} >= {_RESIDUAL_TOL:.0e}); lattice is not ideal."
        )


def _pair_snap_result(
    ab: np.ndarray, kind: str, theta_deg: float, th: float, S_pos: np.ndarray
) -> dict:
    n_cells = int(abs(round(np.linalg.det(S_pos))))
    if theta_deg >= 0:
        moire = S_pos.astype(float) @ ab
        residual = _residual(ab, moire, th)
        _require_commensurate(residual, theta_deg)
        return {
            "bravais": kind,
            "theta_used": float(th),
            "S": S_pos,
            "n_cells": n_cells,
            "residual": residual,
        }
    moire = S_pos.astype(float) @ ab
    T = np.round((moire @ _rot(th)) @ np.linalg.inv(ab)).astype(int)
    theta_used = -th
    moire_t = T.astype(float) @ ab
    residual_t = _residual(ab, moire_t, theta_used)
    _require_commensurate(residual_t, theta_deg)
    return {
        "bravais": kind,
        "theta_used": float(theta_used),
        "S": T,
        "n_cells": int(abs(round(np.linalg.det(T)))),
        "residual": residual_t,
    }


def snap_commensurate(ab: np.ndarray, theta_deg: float, max_cells: int = 217) -> dict:
    ab = np.asarray(ab, dtype=float)
    if max_cells < 1:
        raise ValueError("max_cells must be >= 1.")
    kind = classify_bravais(ab)
    if abs(theta_deg) < 1e-8:
        S = np.eye(2, dtype=int)
        return {
            "bravais": kind,
            "theta_used": 0.0,
            "S": S,
            "n_cells": 1,
            "residual": 0.0,
        }
    if kind == "hexagonal":
        best = None
        # m^2 < N <= max_cells => m upper bound
        m_max = int(math.sqrt(max_cells)) + 2
        for m in range(1, m_max + 1):
            for n in range(0, m):
                N = m * m + m * n + n * n
                if N > max_cells or n == 0:
                    continue
                th, S = hexagonal_pair(m, n)
                err = abs(th - abs(theta_deg))
                if best is None or err < best[0] - 1e-12 or (
                    abs(err - best[0]) <= 1e-12 and N < best[1]
                ):
                    best = (err, N, th, S)
        if best is None:
            raise ValueError(
                f"No unstrained commensurate cell for θ={theta_deg}° "
                f"within max_cells={max_cells}."
            )
        return _pair_snap_result(ab, kind, theta_deg, best[2], best[3])
    if kind == "square":
        best = None
        m_max = int(math.sqrt(max_cells)) + 2
        for m in range(2, m_max + 1):
            for n in range(1, m):
                N = m * m + n * n
                if N > max_cells:
                    continue
                th, S = square_pair(m, n)
                err = abs(th - abs(theta_deg))
                if best is None or err < best[0] - 1e-12 or (
                    abs(err - best[0]) <= 1e-12 and N < best[1]
                ):
                    best = (err, N, th, S)
        if best is None:
            raise ValueError(
                f"No unstrained commensurate cell for θ={theta_deg}° "
                f"within max_cells={max_cells}."
            )
        return _pair_snap_result(ab, kind, theta_deg, best[2], best[3])
    return _search_integer_S(ab, theta_deg, max_cells)


def tile_layer_unstrained(
    moire_ab: np.ndarray,
    layer_ab: np.ndarray,
    species: list[str],
    carts: list[list[float]],
    tags: list[str],
) -> tuple[list[str], list[list[float]], list[str]]:
    """Tile one unstrained layer into a moiré cell and fold to [0, 1)."""
    moire_ab = np.asarray(moire_ab, dtype=float)
    layer_ab = np.asarray(layer_ab, dtype=float)
    S_float = moire_ab @ np.linalg.inv(layer_ab)
    S = np.round(S_float)
    if not np.allclose(S, S_float, atol=_RESIDUAL_TOL, rtol=0.0):
        raise ValueError(
            "Moiré cell is not commensurate with the layer lattice "
            f"(max deviation {float(np.max(np.abs(S - S_float))):.3e})."
        )
    det = int(abs(round(np.linalg.det(S))))
    if det < 1:
        return [], [], []

    Sinv = np.linalg.inv(S)
    n_max = int(np.max(np.abs(S))) + 3
    shifts = []
    for n1 in range(-2 * n_max, 2 * n_max + 1):
        for n2 in range(-2 * n_max, 2 * n_max + 1):
            u = np.array([n1, n2], dtype=float) @ Sinv
            if np.all(u >= -1e-10) and np.all(u < 1.0 - 1e-10):
                shifts.append(n1 * layer_ab[0] + n2 * layer_ab[1])

    moire_inv = np.linalg.inv(moire_ab)
    out_s, out_c, out_t = [], [], []
    seen = set()
    for shift in shifts:
        for symbol, cart, tag in zip(species, carts, tags):
            xy = np.asarray(cart[:2], dtype=float) + shift
            frac = xy @ moire_inv
            frac = frac - np.floor(frac + 1e-12)
            frac = np.where(frac > 1.0 - 1e-10, 0.0, frac)
            key = (round(float(frac[0]), 6), round(float(frac[1]), 6), symbol, tag)
            if key in seen:
                continue
            seen.add(key)
            xy_folded = frac @ moire_ab
            out_s.append(symbol)
            out_c.append([float(xy_folded[0]), float(xy_folded[1]), float(cart[2])])
            out_t.append(tag)
    return out_s, out_c, out_t


def _finalize_slab_structure(
    ab_2x2: np.ndarray,
    species: list[str],
    carts_xyz: list[list[float]],
    tags: list[str],
    vacuum_ang: float,
) -> Structure:
    """Build lattice from in-plane 2x2 + vacuum along c; center slab in c."""
    carts = np.asarray(carts_xyz, dtype=float)
    zmin, zmax = float(carts[:, 2].min()), float(carts[:, 2].max())
    zspan = max(zmax - zmin, 0.0)
    c = zspan + float(vacuum_ang)
    z_center = 0.5 * (zmin + zmax)
    carts[:, 2] = carts[:, 2] - z_center + 0.5 * c
    matrix = np.zeros((3, 3), dtype=float)
    matrix[:2, :2] = ab_2x2
    matrix[2, 2] = c
    lat = Lattice(matrix)
    if lat.a >= 499.0:
        raise ValueError(f"In-plane lattice too large for DFT cell: a={lat.a:.1f} Å")
    return Structure(
        lat, species, carts, coords_are_cartesian=True,
        site_properties={"layer_tag": tags},
    )


def _lattice_angle(ab: np.ndarray) -> float:
    a1, a2 = ab
    cosine = float(np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _placed_layer(
    struct: Structure, angle: float, z_shift: float, layer_number: int
) -> tuple[np.ndarray, list[str], list[list[float]], list[str]]:
    rotation = _rot(angle).T
    layer_ab = np.asarray(struct.lattice.matrix[:2, :2], dtype=float) @ rotation
    mean_z = float(np.mean(struct.cart_coords[:, 2]))
    species, carts, tags = [], [], []
    for site in struct:
        xy = np.asarray(site.coords[:2], dtype=float) @ rotation
        species.append(site.specie.symbol)
        carts.append([
            float(xy[0]),
            float(xy[1]),
            float(site.coords[2]) - mean_z + (float(z_shift) - 12.5),
        ])
        tags.append(f"{site.specie.symbol}_L{layer_number}")
    return layer_ab, species, carts, tags


def build_bernevig_bilayer(
    bottom: Structure,
    top: Structure,
    theta_deg: float,
    stacking: str,
    z_bottom: float,
    z_top: float,
    vacuum_ang: float,
    max_cells: int = 217,
    valley: str = "K",
) -> tuple[Structure, dict]:
    """Build an unstrained commensurate homobilayer at symmetric ±theta/2."""
    if vacuum_ang <= 15:
        raise ValueError("Vacuum must be greater than 15 Å.")
    if stacking not in {"AA", "AB"}:
        raise ValueError("stacking must be 'AA' or 'AB'.")
    if len(bottom) != len(top):
        raise ValueError("Bernevig builder requires equal-atom-count homobilayer inputs.")

    bottom_ab = np.asarray(bottom.lattice.matrix[:2, :2], dtype=float)
    top_ab = np.asarray(top.lattice.matrix[:2, :2], dtype=float)
    bottom_lengths = np.linalg.norm(bottom_ab, axis=1)
    top_lengths = np.linalg.norm(top_ab, axis=1)
    relative_mismatch = np.max(
        np.abs(bottom_lengths - top_lengths)
        / (0.5 * (bottom_lengths + top_lengths))
    )
    angle_mismatch = abs(_lattice_angle(bottom_ab) - _lattice_angle(top_ab))
    if relative_mismatch >= 1e-3 or angle_mismatch >= 0.05:
        raise ValueError("Bernevig builder requires an unstrained homobilayer.")

    hit = snap_commensurate(bottom_ab, theta_deg, max_cells)
    theta_used = float(hit["theta_used"])
    moire = hit["S"].astype(float) @ bottom_ab
    top_angle = theta_used if stacking == "AA" else theta_used + 180.0

    all_species, all_carts, all_tags = [], [], []
    for struct, angle, z_shift, layer_number in (
        (bottom, 0.0, z_bottom, 1),
        (top, top_angle, z_top, 2),
    ):
        layer_ab, species, carts, tags = _placed_layer(
            struct, angle, z_shift, layer_number
        )
        tiled_species, tiled_carts, tiled_tags = tile_layer_unstrained(
            moire, layer_ab, species, carts, tags
        )
        expected_layer = len(struct) * int(hit["n_cells"])
        if len(tiled_carts) != expected_layer:
            raise ValueError(
                f"Layer {layer_number} has {len(tiled_carts)} atoms, "
                f"expected {expected_layer} atoms."
            )
        all_species.extend(tiled_species)
        all_carts.extend(tiled_carts)
        all_tags.extend(tiled_tags)

    global_rotation = _rot(-theta_used / 2.0).T
    moire = moire @ global_rotation
    for cart in all_carts:
        cart[:2] = (np.asarray(cart[:2], dtype=float) @ global_rotation).tolist()

    expected = 2 * len(bottom) * int(hit["n_cells"])
    if len(all_carts) != expected:
        raise ValueError(
            f"Built {len(all_carts)} atoms, expected {expected} atoms."
        )
    slab = _finalize_slab_structure(
        moire, all_species, all_carts, all_tags, vacuum_ang
    )
    info = {
        "status": (
            "perfect_alignment"
            if theta_used == 0.0 and stacking == "AA"
            else "commensurate"
        ),
        "theta_requested": float(theta_deg),
        "theta_used": theta_used,
        "n_cells": int(hit["n_cells"]),
        "bravais": hit["bravais"],
        "valley": valley,
        "q_lattice": q_lattice_label(hit["bravais"], valley),
        "stacking": stacking,
        "residual": float(hit["residual"]),
        "S": hit["S"],
    }
    return slab, info
