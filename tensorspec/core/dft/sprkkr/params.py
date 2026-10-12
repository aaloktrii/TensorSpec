"""Pure-data parameter objects for SPR-KKR SCF and ARPES runs.

No Qt. No ase2sprkkr imports at module load time beyond typing — the heavy
lifting (calling into ase2sprkkr) happens in ``inputs.py``. This module only
knows the *keyword* names, so keep it in sync with ``spec_inputs.f`` (see
design doc §0 for the verified real keyword list).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


def _range_value(pair: Tuple[float, float]):
    """Render a (min, max) pair the way ase2sprkkr / SPR-KKR expects.

    When min == max SPR-KKR wants a bare scalar (``PHI=0.0``), not a
    single-element range (``PHI={0.0,0.0}``). When they differ it wants the
    2-element range (``THETA={-20.0,20.0}``). Verified against
    tests/sprkkr/fixtures/REF_arpes.inp.
    """
    lo, hi = pair
    if lo == hi:
        return lo
    return [lo, hi]


@dataclass
class ScfParams:
    """Parameters for a SPR-KKR SCF (self-consistent potential) run."""

    nl: int = 3
    ne: int = 30
    nktab: int = 250
    niter: int = 200
    mix: float = 0.2
    tol: float = 1e-5
    vxc: str = "VWN"
    alg: str = "BROYDEN2"
    nonmag: bool = False
    dataset: str = "scf"

    def validate(self) -> None:
        if self.nl < 1:
            raise ValueError(f"nl must be >= 1, got {self.nl}")
        if self.ne < 1:
            raise ValueError(f"ne must be >= 1, got {self.ne}")
        if self.nktab < 1:
            raise ValueError(f"nktab must be >= 1, got {self.nktab}")
        if self.niter < 1:
            raise ValueError(f"niter must be >= 1, got {self.niter}")
        if not (0.0 < self.mix <= 1.0):
            raise ValueError(f"mix must be in (0, 1], got {self.mix}")
        if self.tol <= 0.0:
            raise ValueError(f"tol must be > 0, got {self.tol}")
        if not self.dataset:
            raise ValueError("dataset must be a non-empty string")

    def to_ase2sprkkr_dict(self) -> dict:
        """Nested {section: {REAL_KEYWORD: value}} for ase2sprkkr InputParameters."""
        d = {
            "CONTROL": {"DATASET": self.dataset},
            "TAU": {"NKTAB": self.nktab},
            "ENERGY": {"NE": self.ne},
            "SITES": {"NL": self.nl},
            "SCF": {
                "NITER": self.niter,
                "MIX": self.mix,
                "VXC": self.vxc,
                "ALG": self.alg,
                "TOL": self.tol,
            },
        }
        if self.nonmag:
            d["CONTROL"]["NONMAG"] = True
        return d


@dataclass
class ArpesParams:
    """Parameters for a SPR-KKR one-step ARPES run (kkrspec).

    Field list per design doc §5. Defaults reproduce
    tests/sprkkr/fixtures/REF_arpes.inp exactly (aside from DATASET/POTFIL).
    """

    hv_eV: float = 21.2
    pol_p: str = "P"  # P | S | C+ | C-
    theta_ph: float = 45.0
    phi_ph: float = 0.0
    theta_e: Tuple[float, float] = (-20.0, 20.0)
    nt: int = 21
    phi_e: Tuple[float, float] = (0.0, 0.0)
    np_: int = 1
    e_min_eV: float = -6.0
    e_max_eV: float = 1.0
    ne: int = 71
    ework_eV: float = 4.5
    imv_ini_eV: float = 0.05
    imv_fin_eV: float = 2.0
    hkl: Tuple[int, int, int] = (0, 0, 1)
    iq_at_surf: Optional[int] = 1  # None/0 -> auto-pick surface site (workflow.resolve_surface_geometry)
    n_layer: int = 50
    nlat_g_vec: int = 57
    n_laydbl: Tuple[float, float] = (10, 10)
    surf_bar: Tuple[float, float] = (0.25, 0.25)
    nl: int = 3
    nktab: int = 250
    spol: Optional[int] = None
    pol_e: str = "PZ"
    dataset: str = "arpes"
    hkl_frame: str = "conventional"  # "conventional" | "abas" (raw ABAS, needs CRYS_VECS) | "cif" (resolved by workflow.run_arpes)

    @property
    def n_points(self) -> int:
        return self.ne * self.nt * self.np_

    def validate(self) -> None:
        if self.nt < 1:
            raise ValueError(f"nt must be >= 1, got {self.nt}")
        if self.np_ < 1:
            raise ValueError(f"np_ must be >= 1, got {self.np_}")
        if self.ne < 1:
            raise ValueError(f"ne must be >= 1, got {self.ne}")
        if self.pol_p not in ("P", "S", "C+", "C-"):
            raise ValueError(f"pol_p must be one of P/S/C+/C-, got {self.pol_p!r}")
        if self.theta_e[0] > self.theta_e[1]:
            raise ValueError(f"theta_e min > max: {self.theta_e}")
        if self.phi_e[0] > self.phi_e[1]:
            raise ValueError(f"phi_e min > max: {self.phi_e}")
        if len(self.hkl) != 3 or not all(float(h).is_integer() for h in self.hkl):
            raise ValueError(f"hkl must be 3 integers, got {self.hkl}")
        if self.nl < 1:
            raise ValueError(f"nl must be >= 1, got {self.nl}")
        if self.nktab < 1:
            raise ValueError(f"nktab must be >= 1, got {self.nktab}")
        if not self.dataset:
            raise ValueError("dataset must be a non-empty string")
        if self.hkl_frame not in ("conventional", "abas", "cif"):
            raise ValueError(f"hkl_frame must be 'conventional', 'abas' or 'cif', got {self.hkl_frame!r}")

    def to_ase2sprkkr_dict(self) -> dict:
        """Nested {section: {REAL_KEYWORD: value}} for ase2sprkkr InputParameters.

        Emits CRYS_VECS (design doc §0: that keyword flips MILLER_HKL to the
        raw ABAS frame) only when ``hkl_frame == "abas"`` -- see
        ``inputs.build_arpes_inputs``, which raw-injects it into TASK since
        ase2sprkkr's own CRYS_VEC (no S) flag does not match this keyword.
        """
        spec_el = {
            "THETA": _range_value(self.theta_e),
            "PHI": _range_value(self.phi_e),
            "NT": self.nt,
            "NP": self.np_,
            "POL_E": self.pol_e,
        }
        if self.spol is not None:
            spec_el["SPOL"] = self.spol

        return {
            "CONTROL": {"DATASET": self.dataset},
            "TAU": {"NKTAB": self.nktab},
            "ENERGY": {
                "NE": self.ne,
                "EMINEV": self.e_min_eV,
                "EMAXEV": self.e_max_eV,
                "EWORK_EV": self.ework_eV,
                "IMV_INI_EV": self.imv_ini_eV,
                "IMV_FIN_EV": self.imv_fin_eV,
            },
            "SITES": {"NL": self.nl},
            "TASK": {
                "IQ_AT_SURF": self.iq_at_surf,
                "MILLER_HKL": list(self.hkl),
            },
            "SPEC_PH": {
                "THETA": self.theta_ph,
                "PHI": self.phi_ph,
                "POL_P": self.pol_p,
                "EPHOT": self.hv_eV,
            },
            "SPEC_EL": spec_el,
            "SPEC_STR": {
                "N_LAYDBL": list(self.n_laydbl),
                "NLAT_G_VEC": self.nlat_g_vec,
                "N_LAYER": self.n_layer,
                "SURF_BAR": list(self.surf_bar),
            },
        }
