from __future__ import annotations

import re

from pymatgen.core import Structure

_FINAL_POS = re.compile(
    r"ATOMIC_POSITIONS\s*\(\s*crystal\s*\)\s*(.*?)\s*End final coordinates",
    re.IGNORECASE | re.DOTALL,
)


def parse_qe_relaxed_structure(relax_out_text: str, template: Structure) -> Structure:
    """Ionic relax: keep template lattice; replace frac coords from final crystal block."""
    m = _FINAL_POS.search(relax_out_text)
    if not m:
        raise ValueError(
            "No 'ATOMIC_POSITIONS (crystal) ... End final coordinates' in relax output"
        )
    species, fracs = [], []
    for line in m.group(1).strip().splitlines():
        p = line.split()
        if len(p) >= 4 and p[0][0].isalpha():
            species.append(p[0])
            fracs.append([float(p[1]), float(p[2]), float(p[3])])
    if len(species) != len(template):
        raise ValueError(f"Atom count {len(species)} != template {len(template)}")
    site_props = (
        {k: list(v) for k, v in template.site_properties.items()}
        if template.site_properties
        else {}
    )
    return Structure(template.lattice, species, fracs, site_properties=site_props or None)


def write_relaxed_cif(structure: Structure, path: str) -> str:
    structure.to(filename=path, fmt="cif")
    return path
