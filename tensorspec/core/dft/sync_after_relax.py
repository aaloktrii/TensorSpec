from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pymatgen.core import Structure

from tensorspec.core.dft.qe_generator import QEInputGenerator
from tensorspec.core.dft.qe_relax_io import parse_qe_relaxed_structure, write_relaxed_cif

META_FILENAME = "tensorspec_relax_meta.json"
RELAX_OUT = "relax.out"
RELAXED_CIF = "relaxed_structure.cif"


def sync_after_relax(
    out_dir: str,
    *,
    template_cif: str = "structure_template.cif",
    relax_out: str = RELAX_OUT,
    pseudo_dir: str | None = None,
) -> str:
    """Parse relax.out, write relaxed CIF, regenerate electronic QE/Wannier inputs."""
    out = Path(out_dir).resolve()
    relax_path = out / relax_out
    if not relax_path.is_file():
        raise FileNotFoundError(f"Missing relax output: {relax_path}")

    meta_path = out / META_FILENAME
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing meta sidecar: {meta_path}")

    template_path = out / template_cif
    if not template_path.is_file():
        raise FileNotFoundError(f"Missing template CIF: {template_path}")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    template = Structure.from_file(str(template_path))
    relax_text = relax_path.read_text(encoding="utf-8")
    relaxed = parse_qe_relaxed_structure(relax_text, template)

    relaxed_cif_path = out / RELAXED_CIF
    write_relaxed_cif(relaxed, str(relaxed_cif_path))

    gen = QEInputGenerator(template)
    gen.apply_structure(relaxed)

    resolved_pseudo = (
        pseudo_dir
        or os.environ.get("TENSORSPEC_PSEUDO_DIR")
        or str(out / "pseudo")
    )
    gen.app_pseudo_dir = resolved_pseudo

    kmesh = tuple(meta["kmesh"])
    gen.write_scf_input(
        str(out),
        ecutwfc=meta["ecutwfc"],
        ecutrho=meta["ecutrho"],
        kmesh=kmesh,
        use_soc=meta.get("use_soc", False),
        use_gpu=meta.get("use_gpu", False),
        vdw_dft_d3=meta.get("vdw_dft_d3", False),
    )
    gen.write_nscf_input(
        str(out),
        ecutwfc=meta["ecutwfc"],
        ecutrho=meta["ecutrho"],
        kmesh=kmesh,
        nbnd=meta.get("nbnd", 12),
        use_soc=meta.get("use_soc", False),
        use_gpu=meta.get("use_gpu", False),
        vdw_dft_d3=meta.get("vdw_dft_d3", False),
    )
    gen.write_wannier90_input(
        str(out),
        kmesh=kmesh,
        num_wann=meta.get("nbnd", 12),
        use_soc=meta.get("use_soc", False),
        mlwf_mode=meta.get("mlwf", False),
    )
    gen.write_pw2wan_input(str(out))

    return str(relaxed_cif_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parse QE relax.out and regenerate electronic inputs."
    )
    parser.add_argument("--out-dir", required=True, help="QE run directory")
    parser.add_argument(
        "--template-cif",
        default="structure_template.cif",
        help="Template CIF written at Generate time",
    )
    parser.add_argument(
        "--relax-out",
        default=RELAX_OUT,
        help="QE relax stdout file",
    )
    parser.add_argument(
        "--pseudo-dir",
        default=None,
        help="Directory containing .upf files (default: out-dir/pseudo)",
    )
    args = parser.parse_args(argv)

    try:
        path = sync_after_relax(
            args.out_dir,
            template_cif=args.template_cif,
            relax_out=args.relax_out,
            pseudo_dir=args.pseudo_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"FATAL: {exc}", flush=True)
        return 1

    print(f"Wrote {path} and regenerated scf/nscf/wannier inputs.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
