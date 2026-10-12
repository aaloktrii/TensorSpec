"""Pure Python .spc result loading and stitching (no Qt, no GUI).

Handles both point-wise mode (via pointwise_points.json + stitch_points)
and fallback mode (via sibling .inp PHI extraction + stitch_spc).
"""

import os
from typing import Dict, Optional, List
import numpy as np
import xarray as xr

from .outputs import parse_spc, stitch_spc, read_points_json, stitch_points
from .inputs import read_inp_keywords


def spc_paths_to_results(local_paths: List[str], points_json: Optional[str] = None) -> Dict:
    """Parse (+stitch) SPR-KKR *_ARPES_data.spc files into a results dict.

    Point-wise mode (points_json):
    - Load AnglePoint list from pointwise_points.json
    - Parse each .spc with the correct phi_range per point (matched by _pNNNN suffix)
    - Stitch via stitch_points

    Fallback mode:
    - For each .spc, look for sibling .inp file
    - If found, extract PHI from SPEC_EL and parse with that phi_range
    - If not found, parse with default phi_range
    - Stitch via stitch_spc

    Returns dict with 4 keys: intensity_broadened, energy, theta, phi
    (dims transposed to theta, phi, energy)
    """
    # Point-wise mode: points_json + stitch_points
    if points_json and os.path.exists(points_json):
        points, meta = read_points_json(points_json)
        deflector_deg = meta.get("deflector_deg", 0.0)

        # Match each .spc file to its point by _pNNNN suffix
        datasets = []
        for p in points:
            # Find .spc file matching _pNNNN suffix
            suffix = f"_p{p.index:04d}_ARPES_data.spc"
            matching_paths = [lp for lp in local_paths if lp.endswith(suffix)]
            if not matching_paths:
                raise FileNotFoundError(f"No .spc file found matching {suffix}")
            lp = matching_paths[0]
            # Parse with the correct phi_range for this point
            ds = parse_spc(lp, phi_range=(p.phi_e_deg, p.phi_e_deg))
            datasets.append(ds)

        merged = stitch_points(datasets, points, deflector_deg)
    else:
        # Fallback: read PHI from sibling .inp files, or use default
        datasets = []
        for lp in local_paths:
            # Try to find sibling .inp file
            inp_path = lp.replace("_ARPES_data.spc", ".inp")
            phi_range = None
            if os.path.exists(inp_path):
                try:
                    keywords = read_inp_keywords(inp_path)
                    spec_el = keywords.get("SPEC_EL", {})
                    if "PHI" in spec_el:
                        phi_val = float(spec_el["PHI"])
                        phi_range = (phi_val, phi_val)
                except Exception:
                    pass

            # Parse with phi_range if found, else default
            if phi_range:
                ds = parse_spc(lp, phi_range=phi_range)
            else:
                ds = parse_spc(lp)
            datasets.append(ds)

        merged = stitch_spc(datasets) if len(datasets) > 1 else datasets[0]

    # dataset dims are (energy, theta, phi); GUI layout is (theta, phi, energy)
    intensity = np.transpose(merged["I_tot"].values, (1, 2, 0))
    return {
        'intensity_broadened': intensity,
        'energy': merged["energy"].values,
        'theta': merged["theta"].values,
        'phi': merged["phi"].values,
    }
