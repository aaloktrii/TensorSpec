"""Magnetic / relativistic options for QE + Wannier90 input generation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from pymatgen.core import Structure

MODES = ("off", "collinear", "afm", "noncolin", "soc")


@dataclass
class MagneticSpec:
    """How spin appears in pw.x / wannier90.win.

    ``site_labels``: per-atom QE type names (length = nat). When None, use
    chemical symbols. AFM uses labels like ``Fe_up`` / ``Fe_dn`` with the same UPF.
    ``starting_magnetization`` keys are those type labels (scalar for collinear/AFM;
    length-3 sequence for noncolin/soc).
    """

    mode: str = "off"
    starting_magnetization: dict[str, float | Sequence[float]] = field(default_factory=dict)
    site_labels: list[str] | None = None
    tot_magnetization: float | None = None

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"Unknown magnetic mode {self.mode!r}; expected one of {MODES}")

    @classmethod
    def off(cls) -> MagneticSpec:
        return cls(mode="off")

    @classmethod
    def from_use_soc(cls, use_soc: bool) -> MagneticSpec:
        """Back-compat for callers that only had a SOC checkbox."""
        return cls(mode="soc") if use_soc else cls.off()

    @property
    def use_soc(self) -> bool:
        return self.mode == "soc"

    @property
    def uses_fr_pseudos(self) -> bool:
        return self.mode == "soc"

    @property
    def spinors_wannier(self) -> bool:
        return self.mode == "soc"

    def base_symbol(self, type_label: str) -> str:
        """Map ``Fe_up`` → ``Fe`` for pseudopotential lookup."""
        if "_" in type_label:
            return type_label.split("_", 1)[0]
        return type_label

    def unique_types(self, structure: Structure) -> list[str]:
        if self.site_labels is not None:
            if len(self.site_labels) != len(structure):
                raise ValueError(
                    f"site_labels length {len(self.site_labels)} != nat {len(structure)}"
                )
            seen: list[str] = []
            for lab in self.site_labels:
                if lab not in seen:
                    seen.append(lab)
            types = seen
        else:
            types = [el.symbol for el in structure.composition.elements]
        # Align with ATOMIC_SPECIES / starting_magnetization(i) indexing
        if self.starting_magnetization:
            ordered = [t for t in types if t in self.starting_magnetization]
            ordered += [t for t in types if t not in ordered]
            return ordered
        return types

    def label_for_site(self, structure: Structure, index: int) -> str:
        if self.site_labels is not None:
            return self.site_labels[index]
        return structure[index].specie.symbol

    def system_flags(self, type_order: Sequence[str] | None = None) -> str:
        """Extra lines inside ``&SYSTEM`` (leading newlines, no trailing slash).

        ``type_order`` should match ATOMIC_SPECIES order so ``starting_magnetization(i)``
        indices align with QE types.
        """
        if self.mode == "off":
            return ""
        order = list(type_order) if type_order is not None else list(self.starting_magnetization)
        if self.mode == "collinear" or self.mode == "afm":
            lines = ["\n  nspin = 2"]
            for i, lab in enumerate(order, start=1):
                mag = self.starting_magnetization.get(lab, 0.0)
                if isinstance(mag, (list, tuple)):
                    raise ValueError(f"Collinear/AFM mag for {lab} must be a scalar, got {mag!r}")
                lines.append(f"\n  starting_magnetization({i}) = {float(mag)}")
            if self.mode == "collinear" and self.tot_magnetization is not None:
                lines.append(f"\n  tot_magnetization = {float(self.tot_magnetization)}")
            return "".join(lines)
        lines = ["\n  noncolin = .true."]
        if self.mode == "soc":
            lines.append("\n  lspinorb = .true.")
        for i, lab in enumerate(order, start=1):
            mag = self.starting_magnetization.get(lab, 0.0)
            if isinstance(mag, (list, tuple)):
                if len(mag) != 3:
                    raise ValueError(f"Noncolin mag for {lab} needs 3 components")
                mx, my, mz = (float(x) for x in mag)
                lines.append(f"\n  starting_magnetization({i}) = {mx} {my} {mz}")
            else:
                lines.append(f"\n  starting_magnetization({i}) = 0.0 0.0 {float(mag)}")
        return "".join(lines)


def afm_split_spec(
    structure: Structure,
    elements: Sequence[str],
    *,
    mag_up: float = 1.0,
    mag_dn: float = -1.0,
    assignment: str = "alternate",
) -> MagneticSpec:
    """Build AFM MagneticSpec: split each listed element into ``El_up`` / ``El_dn``.

    ``assignment``:
      - ``alternate``: even/odd occurrence index among sites of that element
      - ``layer``: use ``layer_tag`` ending ``_L1`` → up, else dn (needs site prop)
    """
    elements = [str(e) for e in elements]
    site_labels: list[str] = []
    mag: dict[str, float] = {}
    counters = {e: 0 for e in elements}
    tags = structure.site_properties.get("layer_tag")

    for i, site in enumerate(structure):
        sym = site.specie.symbol
        if sym not in elements:
            site_labels.append(sym)
            continue
        if assignment == "layer":
            if not tags:
                raise ValueError("AFM assignment='layer' needs structure site_properties['layer_tag']")
            tag = str(tags[i])
            up = tag.endswith("_L1") or tag.endswith("L1")
            lab = f"{sym}_up" if up else f"{sym}_dn"
        else:
            n = counters[sym]
            counters[sym] = n + 1
            lab = f"{sym}_up" if (n % 2 == 0) else f"{sym}_dn"
        site_labels.append(lab)
        if lab.endswith("_up"):
            mag[lab] = float(mag_up)
        else:
            mag[lab] = float(mag_dn)

    # include non-magnetic species with zero mag so ntyp indexing stays clear
    for el in structure.composition.elements:
        if el.symbol not in elements and el.symbol not in mag:
            mag[el.symbol] = 0.0

    return MagneticSpec(mode="afm", starting_magnetization=mag, site_labels=site_labels)


def collinear_spec(
    structure: Structure,
    mags: Mapping[str, float],
    *,
    tot_magnetization: float | None = None,
) -> MagneticSpec:
    """Collinear FM/ferrimag: ``mags`` keyed by element symbol."""
    full = {el.symbol: float(mags.get(el.symbol, 0.0)) for el in structure.composition.elements}
    return MagneticSpec(
        mode="collinear",
        starting_magnetization=full,
        tot_magnetization=tot_magnetization,
    )
