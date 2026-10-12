import os
import re
import shutil
from pymatgen.core import Structure

IF_POS_FREE = (1, 1, 1)
IF_POS_FIXED = (0, 0, 0)


def build_if_pos_mask(
    structure: Structure,
    mode: str,
    *,
    ref_layer: int = 1,
    z_tol: float = 0.02,
) -> list[tuple[int, int, int]] | None:
    """Build QE selective-dynamics if_pos list (one triple per site).

    mode: ``none`` | ``fix_reference`` | ``fix_bottom``
    """
    if mode in (None, "", "none"):
        return None
    if mode == "fix_bottom":
        z_vals = [float(site.frac_coords[2]) for site in structure]
        z_min = min(z_vals)
        return [
            IF_POS_FIXED if abs(z - z_min) <= z_tol else IF_POS_FREE for z in z_vals
        ]
    if mode == "fix_reference":
        tags = structure.site_properties.get("layer_tag")
        if not tags:
            raise ValueError("Structure lacks layer_tag site properties.")
        suffix = f"_L{ref_layer}"
        return [
            IF_POS_FIXED if str(tag).endswith(suffix) else IF_POS_FREE for tag in tags
        ]
    raise ValueError(f"Unknown if_pos mode: {mode!r}")


class QEInputGenerator:
    """
    Core physics/math engine for generating Quantum Espresso (pw.x) and Wannier90 inputs.
    Zero GUI dependencies. Operates entirely independently of local executables.
    """
    def __init__(self, structure: Structure):
        self.structure = structure
        self.prefix = "tensorspec_run"
        self.app_pseudo_dir = "./pseudo"

    def apply_structure(self, structure: Structure) -> None:
        """Replace the working structure (e.g. after ionic relaxation)."""
        self.structure = structure

    def _generate_atomic_species(self, out_dir: str, use_soc: bool = False) -> str:
        """Extracts unique elements, finds their UPF files based on SOC toggle, and copies them to the run directory."""
        species = []
        os.makedirs(self.app_pseudo_dir, exist_ok=True)
        
        run_pseudo_dir = os.path.join(out_dir, "pseudo")
        os.makedirs(run_pseudo_dir, exist_ok=True)
        
        for el in self.structure.composition.elements:
            mass = el.atomic_mass
            symbol = el.symbol
            
            pseudo_name = None
            pattern = re.compile(rf"^{symbol}[._].*\.upf$", re.IGNORECASE)
            
            # 1. Find all matching elements first
            element_files = [f for f in os.listdir(self.app_pseudo_dir) if pattern.match(f)]
            
            if not element_files:
                raise FileNotFoundError(
                    f"Missing pseudopotential for {symbol}! "
                    f"Please place a {symbol} UPF file in '{self.app_pseudo_dir}'."
                )
                
            # 2. Filter based on SOC Physics
            if use_soc:
                for f in element_files:
                    if "FR" in f.upper() or "REL" in f.upper():
                        pseudo_name = f
                        break
                if pseudo_name is None:
                    raise ValueError(f"SOC is enabled, but no FR or REL pseudopotential found for {symbol}!")
            else:
                for f in element_files:
                    if "FR" not in f.upper() and "REL" not in f.upper():
                        pseudo_name = f
                        break
                if pseudo_name is None:
                    pseudo_name = element_files[0] # Fallback if only one exists
            
            src = os.path.join(self.app_pseudo_dir, pseudo_name)
            dst = os.path.join(run_pseudo_dir, pseudo_name)
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copy2(src, dst)
                
            species.append(f" {symbol}  {float(mass):.4f}  {pseudo_name}")
            
        return "\n".join(species)

    def _generate_explicit_kpoints(self, kmesh: tuple) -> list:
        """Generates the full unreduced k-point grid to 10 decimal places."""
        kpts = []
        for x in range(kmesh[0]):
            for y in range(kmesh[1]):
                for z in range(kmesh[2]):
                    kpts.append(f"  {x/kmesh[0]:.10f}  {y/kmesh[1]:.10f}  {z/kmesh[2]:.10f}")
        return kpts

    def _vdw_system_flag(self, vdw_dft_d3: bool) -> str:
        """Optional DFT-D3 correction for layered vdW systems."""
        return "\n  vdw_corr = 'dft-d3'" if vdw_dft_d3 else ""

    def write_scf_input(self, out_dir: str, ecutwfc: float = 60.0, ecutrho: float = 240.0, kmesh: tuple = (6, 6, 6), use_soc: bool = False, use_gpu: bool = False, vdw_dft_d3: bool = False):
        """Generates the main self-consistent field (SCF) input file."""
        os.makedirs(out_dir, exist_ok=True)
        scf_path = os.path.join(out_dir, "scf.in")
        abs_out = os.path.abspath(os.path.join(out_dir, "out")) + "/"
        
        ibrav = 0 
        nat = len(self.structure)
        ntyp = len(self.structure.composition.elements)

        atomic_species_str = self._generate_atomic_species(out_dir, use_soc)
        
        # Use the UI toggle to inject SOC / optional QE CUDA offload
        soc_flags = "\n  noncolin = .true.\n  lspinorb = .true." if use_soc else ""
        gpu_flags = "\n  use_gpu = .true." if use_gpu else ""
        vdw_flags = self._vdw_system_flag(vdw_dft_d3)

        scf_content = f"""&CONTROL
  calculation = 'scf'
  prefix = '{self.prefix}'
  outdir = './out/'
  pseudo_dir = './pseudo/'
  wf_collect = .true.
/
&SYSTEM
  ibrav = {ibrav}
  nat = {nat}
  ntyp = {ntyp}
  ecutwfc = {ecutwfc}
  ecutrho = {ecutrho}
  occupations = 'smearing'
  smearing = 'marzari-vanderbilt'
  degauss = 0.01{soc_flags}{gpu_flags}{vdw_flags}
/
&ELECTRONS
  conv_thr = 1.0d-8
  mixing_beta = 0.7
/
ATOMIC_SPECIES
{atomic_species_str}

{self._generate_cell_parameters()}

{self._generate_atomic_positions()}

K_POINTS {{automatic}}
  {kmesh[0]} {kmesh[1]} {kmesh[2]}  0 0 0
"""
        with open(scf_path, "w") as f:
            f.write(scf_content)
        return scf_path

    def write_relax_input(
        self,
        out_dir: str,
        *,
        ecutwfc: float = 60.0,
        ecutrho: float = 240.0,
        kmesh: tuple = (6, 6, 6),
        use_soc: bool = False,
        use_gpu: bool = False,
        vdw_dft_d3: bool = False,
        calculation: str = "relax",
        if_pos=None,
    ) -> str:
        """Generates ionic or variable-cell relaxation input (relax.in)."""
        os.makedirs(out_dir, exist_ok=True)
        relax_path = os.path.join(out_dir, "relax.in")

        ibrav = 0
        nat = len(self.structure)
        ntyp = len(self.structure.composition.elements)

        atomic_species_str = self._generate_atomic_species(out_dir, use_soc)

        soc_flags = "\n  noncolin = .true.\n  lspinorb = .true." if use_soc else ""
        gpu_flags = "\n  use_gpu = .true." if use_gpu else ""
        vdw_flags = self._vdw_system_flag(vdw_dft_d3)

        cell_block = ""
        if calculation == "vc-relax":
            cell_block = """&CELL
  cell_dynamics = 'bfgs'
  press = 0.0
/
"""

        # forc_conv_thr lives in &CONTROL (QE INPUT_PW), not &IONS.
        relax_content = f"""&CONTROL
  calculation = '{calculation}'
  prefix = '{self.prefix}'
  outdir = './out/'
  pseudo_dir = './pseudo/'
  wf_collect = .true.
  forc_conv_thr = 1.0d-3
/
&SYSTEM
  ibrav = {ibrav}
  nat = {nat}
  ntyp = {ntyp}
  ecutwfc = {ecutwfc}
  ecutrho = {ecutrho}
  occupations = 'smearing'
  smearing = 'marzari-vanderbilt'
  degauss = 0.01{soc_flags}{gpu_flags}{vdw_flags}
/
&ELECTRONS
  conv_thr = 1.0d-8
  mixing_beta = 0.7
/
&IONS
  ion_dynamics = 'bfgs'
/
{cell_block}ATOMIC_SPECIES
{atomic_species_str}

{self._generate_cell_parameters()}

{self._generate_atomic_positions(if_pos=if_pos)}

K_POINTS {{automatic}}
  {kmesh[0]} {kmesh[1]} {kmesh[2]}  0 0 0
"""
        with open(relax_path, "w") as f:
            f.write(relax_content)
        return relax_path

    def write_nscf_input(self, out_dir: str, ecutwfc: float = 60.0, ecutrho: float = 240.0, kmesh: tuple = (6, 6, 6), nbnd: int = 12, use_soc: bool = False, use_gpu: bool = False, vdw_dft_d3: bool = False):
        """Generates the non-self-consistent field (NSCF) input file with explicit k-points."""
        nscf_path = os.path.join(out_dir, "nscf.in")
        abs_out = os.path.abspath(os.path.join(out_dir, "out")) + "/"
        
        ibrav = 0  
        nat = len(self.structure)
        ntyp = len(self.structure.composition.elements)
        
        atomic_species_str = self._generate_atomic_species(out_dir, use_soc)
        
        kpts = self._generate_explicit_kpoints(kmesh)
        kpts_qe = "\n".join([f"{k}  1.0" for k in kpts])
        
        # Use the UI toggle to inject SOC / optional QE CUDA offload
        soc_flags = "\n  noncolin = .true.\n  lspinorb = .true." if use_soc else ""
        gpu_flags = "\n  use_gpu = .true." if use_gpu else ""
        vdw_flags = self._vdw_system_flag(vdw_dft_d3)

        nscf_content = f"""&CONTROL
  calculation = 'nscf'
  prefix = '{self.prefix}'
  outdir = './out/'
  pseudo_dir = './pseudo/'
  wf_collect = .true.
/
&SYSTEM
  ibrav = {ibrav}
  nat = {nat}
  ntyp = {ntyp}
  nbnd = {nbnd}
  nosym = .true.
  noinv = .true.
  ecutwfc = {ecutwfc}
  ecutrho = {ecutrho}
  occupations = 'smearing'
  smearing = 'marzari-vanderbilt'
  degauss = 0.01{soc_flags}{gpu_flags}{vdw_flags}
/
&ELECTRONS
  conv_thr = 1.0d-8
  mixing_beta = 0.7
/
ATOMIC_SPECIES
{atomic_species_str}

{self._generate_cell_parameters()}

{self._generate_atomic_positions()}

K_POINTS {{crystal}}
{len(kpts)}
{kpts_qe}
"""
        with open(nscf_path, "w") as f:
            f.write(nscf_content)
        return nscf_path

    def write_wannier90_input(self, out_dir: str, kmesh: tuple = (6, 6, 6), num_wann: int = 12, use_soc: bool = False, mlwf_mode: bool = False):
        """Generates a base wannier90.win file for extracting the tight binding Hamiltonian."""
        win_path = os.path.join(out_dir, "wannier90.win")
        
        kpts = self._generate_explicit_kpoints(kmesh)
        kpts_win = "\n".join(kpts)
        
        # Determine localization iterations based on user choice
        iterations = 100 if mlwf_mode else 0
        
        proj_lines = []
        for el in self.structure.composition.elements:
            if el.is_transition_metal or el.number > 30:
                proj_lines.append(f"{el.symbol}:s;p;d")
            else:
                proj_lines.append(f"{el.symbol}:s;p")
        proj_string = "\n".join(proj_lines)
        
        # Use the UI toggle to inject SOC
        spinor_str = "spinors = true\n" if use_soc else ""
        
        win_content = f"""num_wann = {num_wann}
num_iter = {iterations}
num_print_cycles = 10

write_hr = true
write_xyz = true
use_ws_distance = true
{spinor_str}
begin projections
{proj_string}
end projections

begin unit_cell_cart
{chr(10).join(["  " + "  ".join([f"{v:.6f}" for v in row]) for row in self.structure.lattice.matrix])}
end unit_cell_cart

begin atoms_frac
{chr(10).join([f" {site.specie.symbol}  {site.frac_coords[0]:.6f}  {site.frac_coords[1]:.6f}  {site.frac_coords[2]:.6f}" for site in self.structure])}
end atoms_frac

mp_grid = {kmesh[0]} {kmesh[1]} {kmesh[2]}

begin kpoints
{kpts_win}
end kpoints
"""
        from pymatgen.symmetry.bandstructure import HighSymmKpath
        try:
            kpath_obj = HighSymmKpath(self.structure)
            kpts_dict = kpath_obj.kpath['kpoints']
            path_segments = kpath_obj.kpath['path']
            
            path_lines = []
            for segment in path_segments:
                for i in range(len(segment) - 1):
                    l1, l2 = segment[i], segment[i+1]
                    p1, p2 = kpts_dict[l1], kpts_dict[l2]
                    l1_str = "G" if "Gamma" in l1 or l1 == "\\Gamma" else l1
                    l2_str = "G" if "Gamma" in l2 or l2 == "\\Gamma" else l2
                    path_lines.append(f"{l1_str} {p1[0]:.5f} {p1[1]:.5f} {p1[2]:.5f}  {l2_str} {p2[0]:.5f} {p2[1]:.5f} {p2[2]:.5f}")
                    
            win_content += "\nbands_plot = true\n"
            win_content += "bands_num_points = 100\n"
            win_content += "begin kpoint_path\n"
            win_content += "\n".join(path_lines)
            win_content += "\nend kpoint_path\n"
        except Exception as e:
            print(f"Could not auto-generate W90 k-path: {e}")

        with open(win_path, "w") as f:
            f.write(win_content)
        return win_path

    def write_pw2wan_input(self, out_dir: str):
        """Generates the bridge input file for pw2wannier90.x."""
        pw2wan_path = os.path.join(out_dir, "pw2wan.in")
        abs_out = os.path.abspath(os.path.join(out_dir, "out")) + "/"
        
        content = f"""&inputpp
  outdir = './out/'
  prefix = '{self.prefix}'
  seedname = 'wannier90'
  write_mmn = .true.
  write_amn = .true.
  write_unk = .false.
/
"""
        with open(pw2wan_path, "w") as f:
            f.write(content)
        return pw2wan_path

    def _generate_cell_parameters(self) -> str:
        """Extracts the lattice matrix."""
        params = ["CELL_PARAMETERS {angstrom}"]
        for row in self.structure.lattice.matrix:
            params.append("  " + "  ".join([f"{v:.6f}" for v in row]))
        return "\n".join(params)

    def _generate_atomic_positions(self, if_pos=None) -> str:
        """Converts PyMatgen fractional coordinates to QE format.

        When ``if_pos`` is provided, append QE selective-dynamics flags per site
        (0 = fixed, 1 = free along each axis).
        """
        positions = ["ATOMIC_POSITIONS {crystal}"]
        for i, site in enumerate(self.structure):
            coords = "  ".join([f"{c:.6f}" for c in site.frac_coords])
            # Use pure element symbol (e.g. 'Te') instead of string with oxidation state (e.g. 'Te2-')
            line = f" {site.specie.symbol}  {coords}"
            if if_pos is not None:
                flags = if_pos[i]
                line += f"  {flags[0]}  {flags[1]}  {flags[2]}"
            positions.append(line)
        return "\n".join(positions)

    def resolve_if_pos(self, mode: str, *, ref_layer: int = 1, z_tol: float = 0.02):
        """Convenience wrapper around :func:`build_if_pos_mask` for this structure."""
        return build_if_pos_mask(
            self.structure, mode, ref_layer=ref_layer, z_tol=z_tol
        )
    
    def _detect_soc(self) -> bool:
        """Detects if any provided pseudopotential is fully relativistic (SOC)."""
        for el in self.structure.composition.elements:
            symbol = el.symbol
            pattern = re.compile(rf"^{symbol}[._].*\.upf$", re.IGNORECASE)
            for file in os.listdir(self.app_pseudo_dir):
                if pattern.match(file):
                    if "rel" in file.lower() or "fr" in file.lower():
                        return True
        return False