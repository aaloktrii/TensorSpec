import numpy as np
import collections
import collections.abc
# Monkey-patch to fix Python 3.10+ compatibility for Chinook
collections.Iterable = collections.abc.Iterable

try:
    import chinook.build_lib as build_lib
    from chinook.ARPES_lib import experiment
    import chinook.electron_configs as econ  
    import chinook.radint_lib as radint_lib  # <-- New import to patch the cutoff bug
    
    # --- PATCH 1: CHINOOK'S MISSING ATOMIC DATABASE & NOISY PRINTS ---
    import sys
    import io
    
    original_Z_eff = econ.Z_eff
    _dummy_stream = io.StringIO() # A permanent black hole for Chinook's print statements
    
    def patched_Z_eff(Z, orb):
        # 1. Temporarily hijack stdout to trap Chinook's hardcoded error messages
        old_stdout = sys.stdout
        sys.stdout = _dummy_stream
        try:
            result = original_Z_eff(Z, orb)
        finally:
            # Restore normal printing immediately after
            sys.stdout = old_stdout
            
        if result is None:
            # Complete Slater's Rule effective charges for heavy elements (Rows 5 & 6)
            z_eff_db = {
                # --- Row 5 ---
                37: 2.20, 38: 2.85, # 5s block (Rb, Sr)
                39: 3.00, 40: 3.65, 41: 4.30, 42: 4.95, 43: 5.60, # 4d block (Y-Tc)
                44: 6.25, 45: 6.90, 46: 7.55, 47: 8.20, 48: 8.85, # 4d block (Ru-Cd)
                49: 5.00, 50: 5.65, 51: 6.30, 52: 6.95, 53: 7.60, 54: 8.25, # 5p block (In-Xe)
                
                # --- Row 6 ---
                55: 2.20, 56: 2.85, # 6s block (Cs, Ba)
                **{z: 3.00 + (z-57)*0.35 for z in range(57, 72)}, # 4f block (La-Lu)
                72: 3.65, 73: 4.30, 74: 4.95, 75: 5.60, 76: 6.25, # 5d block (Hf-Os)
                77: 6.90, 78: 7.55, 79: 8.20, 80: 8.85,           # 5d block (Ir-Hg)
                81: 5.00, 82: 5.65, 83: 6.30, 84: 6.95, 85: 7.60, 86: 8.25  # 6p block (Tl-Rn)
            }
            # Fallback to 4.5 only for extreme actinides (Z > 86)
            return z_eff_db.get(Z, 4.5)  
        return result
        
    econ.Z_eff = patched_Z_eff
    
    # --- PATCH 2: CHINOOK'S UNBOUND LOCAL ERROR BUG ---
    original_find_cutoff = radint_lib.find_cutoff
    
    def safe_find_cutoff(integrand):
        try:
            return original_find_cutoff(integrand)
        except UnboundLocalError:
            # The integrand is a mathematical function object, not an array.
            # If the integral evaluates to effectively zero, Chinook forgets to assign a cutoff.
            # We return a generous physical radial distance (30.0 Bohr radii) to safely bypass.
            return 30.0
            
    radint_lib.find_cutoff = safe_find_cutoff
    # -----------------------------------------------

    CHINOOK_AVAILABLE = True
except Exception as e:
    import traceback
    print("\n" + "="*50)
    print("CHINOOK IMPORT FAILED WITH ERROR:")
    traceback.print_exc()
    print("="*50 + "\n")
    CHINOOK_AVAILABLE = False

class ChinookWrapper:
    """
    The backend bridge to Chinook's ARPES matrix element calculator.
    Translates experimental beamline parameters into physical polarization vectors.
    """
    def __init__(self):
        self.tb_model = None
        self.fermi_shift = 0.0
        self.B_matrix = None  # <-- Add this line

    def build_model(self, tb_dict):
        """
        Safely loads the tight-binding model directly from the workspace and 
        extracts the physical on-site energy shift and reciprocal lattice.
        """
        if not CHINOOK_AVAILABLE:
            print("Chinook not installed. Running in Dummy Mode.")
            return
            
        # 1. Safely extract the pre-built model without dangerously rebuilding it
        if tb_dict.get('tb_model') is not None:
            self.tb_model = tb_dict['tb_model']
        else:
            basis = tb_dict.get('basis', tb_dict.get('Basis', tb_dict.get('chinook_basis')))
            h_dict = tb_dict.get('H_dict', tb_dict.get('hamiltonian_dict', tb_dict.get('hamiltonian')))
            if basis is None or h_dict is None:
                raise ValueError("CRITICAL: Workspace missing Tight-Binding params.")
            self.tb_model = build_lib.gen_TB(basis, h_dict)

        # 2. Extract energy reference for ARPES (avoid double-shifting Wannier H)
        self.fermi_shift = float(
            tb_dict.get(
                'arpes_e_fermi_shift',
                tb_dict.get('fermi_energy', 0.0),
            )
            or 0.0
        )

        # --- ADD THESE DEBUG LINES ---
        print("\n[LOG 4 - CHINOOK WRAPPER]")
        print(f"tb_dict['onsite_e']: {tb_dict.get('onsite_e', 'MISSING')}")
        print(f"tb_dict['fermi_energy'] (QE): {tb_dict.get('fermi_energy', 'MISSING')}")
        print(f"tb_dict['arpes_e_fermi_shift']: {tb_dict.get('arpes_e_fermi_shift', 'MISSING')}")
        print(f"Calculated self.fermi_shift applied to eigenvalues: {self.fermi_shift}")
        # -----------------------------
        
        # 3. Safely extract Reciprocal Lattice Vectors (B_matrix)
        # Chinook TB_model does not store these natively, so we pull them from the workspace dict
        if 'structure' in tb_dict:
            self.B_matrix = tb_dict['structure'].lattice.reciprocal_lattice.matrix
        elif 'recip_matrix' in tb_dict:
            self.B_matrix = tb_dict['recip_matrix']
        elif 'avec' in tb_dict:
            A_matrix = np.array(tb_dict['avec'])
            self.B_matrix = 2 * np.pi * np.linalg.inv(A_matrix).T
        elif hasattr(self.tb_model, 'Kobj') and self.tb_model.Kobj is not None and hasattr(self.tb_model.Kobj, 'avec'):
            A_matrix = np.array(self.tb_model.Kobj.avec)
            self.B_matrix = 2 * np.pi * np.linalg.inv(A_matrix).T
        else:
            print("WARNING: No lattice vectors found in Tight-Binding dictionary. Falling back to Simple Cubic.")
            self.B_matrix = 2 * np.pi * np.eye(3)

    def run_simulation(self, experiment_kwargs):
        """
        Calculates matrix elements over a transformed detector frame to strictly align
        kx with the analyzer slit and ky with the deflector scan direction.
        """
        # 1. Extract and safely copy k-space bounds (kx, ky, E) so we don't mutate the UI
        kb = {key: list(val) for key, val in experiment_kwargs['k_bounds'].items()}
        
        # Override resolutions for 1D slit measurements without deflection
        if kb['X'][0] == kb['X'][1]: kb['X'][2] = 1
        if kb['Y'][0] == kb['Y'][1]: kb['Y'][2] = 1
            
        num_x, num_y, num_e = int(kb['X'][2]), int(kb['Y'][2]), int(kb['E'][2])

        if not CHINOOK_AVAILABLE or self.tb_model is None:
            return {'intensity_broadened': np.random.rand(num_x, num_y, num_e)}

        from tensorspec.core.arpes.one_step.chinook_arpes_kmesh import (
            physics_from_experiment_kwargs,
            run_chinook_arpes,
            run_grizzly_arpes,
        )

        physics = physics_from_experiment_kwargs(experiment_kwargs)
        me_engine = str(experiment_kwargs.get("me_engine", "chinook") or "chinook").lower()
        me_device = str(experiment_kwargs.get("me_device", "cpu") or "cpu").lower()
        # Local Mac often has no CUDA — keep GrizzlyME on CPU rather than dying.
        if me_device == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    me_device = "cpu"
            except Exception:
                me_device = "cpu"

        try:
            use_grizzly = me_engine in ("grizzly", "auto")
            if use_grizzly:
                try:
                    import grizzly  # noqa: F401
                except ImportError:
                    if me_engine == "grizzly":
                        raise RuntimeError(
                            "GrizzlyME requested but grizzlyme is not installed."
                        )
                    use_grizzly = False
            if use_grizzly:
                print(
                    f"Local ARPES ME: GrizzlyME device={me_device}",
                    flush=True,
                )
                intensity_3d = run_grizzly_arpes(
                    self.tb_model,
                    kb,
                    physics,
                    self.B_matrix,
                    fermi_shift=self.fermi_shift,
                    device=me_device,
                )
            else:
                print("Local ARPES ME: Chinook CPU", flush=True)
                intensity_3d = run_chinook_arpes(
                    self.tb_model,
                    kb,
                    physics,
                    self.B_matrix,
                    fermi_shift=self.fermi_shift,
                )
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Chinook Calculation Error: {e}") from e

        return {'intensity_broadened': intensity_3d}