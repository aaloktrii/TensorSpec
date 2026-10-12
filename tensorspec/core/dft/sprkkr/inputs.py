"""Build real SPR-KKR .inp files via ase2sprkkr (no binary run here).

Thin wrapper: dataclass params -> ase2sprkkr InputParameters -> calc.save_input.
We launch kkrscf/kkrspec ourselves elsewhere (design doc §0: ase2sprkkr's own
run wrapper is flaky). This module is I/O only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from .params import ArpesParams, ScfParams

PathLike = Union[str, Path]


@dataclass
class ScfInputs:
    inp_path: Path
    pot_path: Path
    dataset: str


@dataclass
class ArpesInputs:
    inp_path: Path
    dataset: str
    expected_spc: str
    expected_log: str


def _apply_dict(ip, nested: dict) -> list:
    """Set ip.<SECTION>.<KEY> = value for every entry in nested dict.

    Returns a list of (section, key, value) that ase2sprkkr refused, so the
    caller can fall back to raw-text injection instead of losing the value.
    """
    rejected = []
    for section, keys in nested.items():
        try:
            sec_obj = getattr(ip, section)
        except AttributeError:
            for key, value in keys.items():
                rejected.append((section, key, value))
            continue
        for key, value in keys.items():
            try:
                setattr(sec_obj, key, value)
            except Exception:
                rejected.append((section, key, value))
    return rejected


def _format_raw_value(value) -> str:
    if isinstance(value, bool):
        return ""  # bare flag keyword, no value
    if isinstance(value, (list, tuple)):
        return "{" + ",".join(str(v) for v in value) + "}"
    return str(value)


def _inject_raw_lines(path: PathLike, rejected: list) -> None:
    """Fallback for keys ase2sprkkr's InputParameters refused to set.

    Appends `KEY=value` (or bare `KEY` for a flag) directly under the target
    section header in the already-written .inp text. Documented workaround
    per design doc / plan — should rarely trigger since every field in
    ScfParams/ArpesParams round-trips through ase2sprkkr cleanly today.
    """
    if not rejected:
        return
    path = Path(path)
    lines = path.read_text().splitlines()
    for section, key, value in rejected:
        raw = _format_raw_value(value)
        new_line = f"\t{key}={raw}" if raw else f"\t{key}"
        try:
            idx = next(i for i, l in enumerate(lines) if l.strip() == section)
            lines.insert(idx + 1, new_line)
        except StopIteration:
            lines.append("")
            lines.append(section)
            lines.append(new_line)
    path.write_text("\n".join(lines) + "\n")


def build_scf_inputs(structure, params: ScfParams, out_dir: PathLike) -> ScfInputs:
    """structure: pymatgen.core.Structure. Writes <dataset>.inp + <dataset>.pot."""
    from ase2sprkkr.input_parameters.input_parameters import InputParameters
    from ase2sprkkr.sprkkr.calculator import SPRKKR
    from pymatgen.io.ase import AseAtomsAdaptor

    params.validate()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    atoms = AseAtomsAdaptor.get_atoms(structure)
    calc = SPRKKR(atoms=atoms)
    ip = InputParameters.create_input_parameters("SCF")
    rejected = _apply_dict(ip, params.to_ase2sprkkr_dict())

    input_file = f"{params.dataset}.inp"
    potential_file = f"{params.dataset}.pot"
    calc.save_input(
        directory=str(out_dir),
        input_file=input_file,
        potential_file=potential_file,
        input_parameters=ip,
        atoms=atoms,
    )

    inp_path = out_dir / input_file
    _inject_raw_lines(inp_path, rejected)

    return ScfInputs(
        inp_path=inp_path,
        pot_path=out_dir / potential_file,
        dataset=params.dataset,
    )


def build_arpes_inputs(
    pot_path: PathLike, params: ArpesParams, out_dir: PathLike, extra_raw: dict = None
) -> ArpesInputs:
    """pot_path: converged (or starting) potential file. Writes <dataset>.inp.

    Uses SPRKKR(potential=pot_path), NOT atoms= (design doc §0 gotcha).

    ``extra_raw``: optional {SECTION: {KEY: value}} applied via the same
    ``_apply_dict`` path AFTER params' own dict, for keys with no ArpesParams
    field yet (e.g. SPEC_EL.TYP/BETA1/BETA2/ROTAXIS -- 2026-09-12 design-doc
    probe, see docs/superpowers/specs/2026-09-11-sprkkr-full-geometry-design.md
    §7). Diagnostic escape hatch, not a stable API -- values here still go
    through ase2sprkkr's own validation/grammar like any other field.
    """
    from ase2sprkkr.input_parameters.input_parameters import InputParameters
    from ase2sprkkr.sprkkr.calculator import SPRKKR

    params.validate()
    if not params.iq_at_surf:
        raise ValueError(
            "params.iq_at_surf must be resolved to an int before build_arpes_inputs "
            "(None/0 means 'auto' -- see workflow.resolve_surface_geometry / run_arpes)"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    calc = SPRKKR(potential=str(pot_path))
    ip = InputParameters.create_input_parameters("ARPES")
    rejected = _apply_dict(ip, params.to_ase2sprkkr_dict())
    if extra_raw:
        rejected = list(rejected) + _apply_dict(ip, extra_raw)

    input_file = f"{params.dataset}.inp"
    calc.save_input(
        directory=str(out_dir),
        input_file=input_file,
        input_parameters=ip,
    )

    inp_path = out_dir / input_file
    if getattr(params, "hkl_frame", "conventional") == "abas":
        rejected = list(rejected) + [("TASK", "CRYS_VECS", True)]
    _inject_raw_lines(inp_path, rejected)

    # Verified pattern (design doc §0): DATASET=_Cu_ARPES -> _Cu_ARPES_ARPES_data.spc
    expected_spc = f"{params.dataset}_ARPES_data.spc"
    expected_log = f"{params.dataset}_ARPES_SPEC.out"

    return ArpesInputs(
        inp_path=inp_path,
        dataset=params.dataset,
        expected_spc=expected_spc,
        expected_log=expected_log,
    )


_SECTION_RE = re.compile(r"^\S")


def read_inp_keywords(path: PathLike) -> dict:
    """Tiny .inp parser for tests + legacy compare.

    Returns {section: {KEY: raw_value_string}}. A bare flag line (no '=',
    e.g. `SCF` under TASK, or `CRYS_VEC`) is stored as {key: ""}. Array
    values keep their literal `{a,b}` text; callers normalize as needed.
    """
    sections: dict = {}
    current = None
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        if _SECTION_RE.match(line):
            current = line.strip()
            sections.setdefault(current, {})
            continue
        if current is None:
            continue
        stripped = line.strip()
        if "=" in stripped:
            key, _, value = stripped.partition("=")
            sections[current][key.strip()] = value.strip()
        else:
            sections[current][stripped] = ""
    return sections
