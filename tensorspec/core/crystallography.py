# File: tensorspec/core/crystallography.py
import os
import json
import warnings
import numpy as np
from scipy.spatial import ConvexHull
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.surface import SlabGenerator


class CrystalEngine:
    """
    Core mathematical engine for crystallographic transformations, 
    supercell generation, CDW distortion propagation, and 2D heterostructure stacking.
    """

    @staticmethod
    def get_universal_primitive_matrix(structure: Structure) -> np.ndarray:
        """Universally calculates the primitive lattice matrix without Cartesian rotation."""
        try:
            sga = SpacegroupAnalyzer(structure)
            centering = sga.get_space_group_symbol()[0].upper()
            
            # Universal Transformation Matrices for all 3D Bravais Lattices
            if centering == 'F': P = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
            elif centering == 'I': P = np.array([[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]])
            elif centering == 'A': P = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, -0.5], [0.0, 0.5, 0.5]])
            elif centering == 'B': P = np.array([[0.5, 0.0, -0.5], [0.0, 1.0, 0.0], [0.5, 0.0, 0.5]])
            elif centering == 'C': P = np.array([[0.5, 0.5, 0.0], [-0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
            elif centering == 'R': P = np.array([[2/3, 1/3, 1/3], [-1/3, 1/3, 1/3], [-1/3, -2/3, 1/3]])
            else: P = np.eye(3)
            
            return np.dot(P, structure.lattice.matrix)
        except Exception:
            return structure.lattice.matrix

    @staticmethod
    def get_symmetry_info(structure: Structure) -> dict:
        """Returns space group symbol and conventional-to-primitive volume ratio."""
        try:
            sga = SpacegroupAnalyzer(structure)
            spg_symbol = sga.get_space_group_symbol()
            conv_vol = structure.lattice.volume
            
            P = sga.get_conventional_to_primitive_transformation_matrix()
            prim_matrix = np.dot(P, structure.lattice.matrix)
            prim_vol = abs(np.linalg.det(prim_matrix))
            ratio = int(round(conv_vol / prim_vol))
            return {"spacegroup": spg_symbol, "volume_ratio": ratio}
        except Exception:
            return {"spacegroup": "N/A", "volume_ratio": 1}

    @staticmethod
    def apply_cdw_distortion(supercell: Structure, target_el: str, q_vec: tuple, amp_vec: tuple, phase_deg: float) -> Structure:
        """Propagates a Charge Density Wave (CDW) sinusoidal atomic displacement across the lattice."""
        qx, qy, qz = q_vec
        ax, ay, az = amp_vec
        phase = np.radians(phase_deg)
        
        base_frac_coords = supercell.frac_coords.copy()
        new_cart = supercell.cart_coords.copy()
        
        for i, site in enumerate(supercell):
            if target_el == "All Elements" or site.specie.symbol == target_el:
                fx, fy, fz = base_frac_coords[i]
                wave_arg = 2 * np.pi * (qx * fx + qy * fy + qz * fz) + phase
                dx = ax * np.cos(wave_arg)
                dy = ay * np.cos(wave_arg)
                dz = az * np.cos(wave_arg)
                new_cart[i] += np.array([dx, dy, dz])
                
        for i, site in enumerate(supercell):
            supercell.replace(i, site.specie, new_cart[i], coords_are_cartesian=True)
            
        return supercell

    @staticmethod
    def generate_template_structure(template_name: str, db_path: str = "user_templates.json") -> Structure:
        """Generates standard 2D monolayer and stacked structures with vacuum padding."""
        c_vac = 25.0
        
        if "Graphene" in template_name:
            lat = Lattice.hexagonal(2.46, c_vac)
            stacks = {
                "Graphene (Monolayer)": ["A"],
                "Graphene (AB Bilayer)": ["A", "B"],
                "Graphene (AA Bilayer)": ["A", "A"],
                "Graphene (ABA Trilayer)": ["A", "B", "A"],
                "Graphene (ABC 3-Layer)": ["A", "B", "C"],
                "Graphene (ABCA 4-Layer)": ["A", "B", "C", "A"],
                "Graphene (ABCAB 5-Layer)": ["A", "B", "C", "A", "B"],
                "Graphene (ABCABC 6-Layer)": ["A", "B", "C", "A", "B", "C"],
                "Graphene (ABCABCA 7-Layer)": ["A", "B", "C", "A", "B", "C", "A"]
            }
            seq = stacks.get(template_name, ["A"])
            coords = []
            z_spacing = 3.35 / c_vac
            for i, layer in enumerate(seq):
                z = 0.5 + (i * z_spacing)
                if layer == "A": coords.extend([[1/3, 2/3, z], [2/3, 1/3, z]])
                elif layer == "B": coords.extend([[0.0, 0.0, z], [1/3, 2/3, z]])
                elif layer == "C": coords.extend([[0.0, 0.0, z], [2/3, 1/3, z]])
            return Structure(lat, ["C"] * len(coords), coords)
            
        elif "(1T')" in template_name and ("TaIrTe4" in template_name or "NbIrTe4" in template_name):
            is_ta = "TaIrTe4" in template_name
            lat = Lattice.orthorhombic(3.808 if is_ta else 3.78, 12.605 if is_ta else 12.4, c_vac)
            m1 = "Ta" if is_ta else "Nb"
            species = ["Te", "Te", m1, "Te", "Te", "Ir", "Te", "Te", "Ir", "Te", "Te", m1]
            coords = [
                [0.0, 0.0657, 0.4424], [0.5, 0.1518, 0.5591], [0.0, 0.2691, 0.4963], [0.5, 0.3229, 0.4206], 
                [0.0, 0.4120, 0.5741], [0.5, 0.4655, 0.5011], [0.0, 0.5650, 0.4463], [0.5, 0.6533, 0.5526], 
                [0.0, 0.7531, 0.4981], [0.5, 0.8076, 0.4254], [0.0, 0.8933, 0.5800], [0.5, 0.9477, 0.5044], 
            ]
            return Structure(lat, species, coords)
            
        elif template_name == "h-BN":
            lat = Lattice.hexagonal(2.50, c_vac)
            return Structure(lat, ["B", "N"], [[1/3, 2/3, 0.5], [2/3, 1/3, 0.5]])
            
        elif template_name == "MoS2":
            lat = Lattice.hexagonal(3.16, c_vac)
            z_off = 1.56 / c_vac
            return Structure(lat, ["Mo", "S", "S"], [[1/3, 2/3, 0.5], [2/3, 1/3, 0.5 + z_off], [2/3, 1/3, 0.5 - z_off]])
            
        elif template_name == "WSe2":
            lat = Lattice.hexagonal(3.28, c_vac)
            z_off = 1.66 / c_vac
            return Structure(lat, ["W", "Se", "Se"], [[1/3, 2/3, 0.5], [2/3, 1/3, 0.5 + z_off], [2/3, 1/3, 0.5 - z_off]])

        if os.path.exists(db_path):
            with open(db_path, "r") as f:
                custom_templates = json.load(f)
            if template_name in custom_templates:
                return Structure.from_dict(custom_templates[template_name])
        return None

    @staticmethod
    def extract_monolayer_vdw(bulk_struct: Structure) -> tuple[Structure, float]:
        """Automatically detects maximal van der Waals gap and cleaves along Z."""
        bulk_struct.translate_sites(list(range(len(bulk_struct))), [0.01, 0.01, 0.01], to_unit_cell=True)
        sites_sorted = sorted(bulk_struct, key=lambda s: s.frac_coords[2])
        z_fracs = [s.frac_coords[2] for s in sites_sorted]
        n_sites = len(z_fracs)
        if n_sites < 2:
            raise ValueError("Structure has too few atoms to evaluate spacing differences.")

        gaps = []
        for i in range(n_sites):
            diff = z_fracs[i+1] - z_fracs[i] if i < n_sites - 1 else (z_fracs[0] + 1.0) - z_fracs[i]
            gaps.append((diff, i))

        max_gap_diff, split_index = max(gaps, key=lambda x: x[0])
        gap_angstrom = max_gap_diff * bulk_struct.lattice.c

        monolayer_sites = []
        for idx, site in enumerate(sites_sorted):
            shifted_z = site.frac_coords[2] + (1.0 if idx <= split_index else 0.0)
            monolayer_sites.append((site, shifted_z))

        monolayer_sites = sorted(monolayer_sites, key=lambda x: x[1])
        final_sites = [item[0] for item in monolayer_sites[:-1]] if n_sites > 1 else [item[0] for item in monolayer_sites]
        
        new_matrix = bulk_struct.lattice.matrix.copy()
        new_matrix[2] = [0, 0, 25.0]
        new_lat = Lattice(new_matrix)
        
        cart_coords = [s.coords for s in final_sites]
        z_vals = [c[2] for c in cart_coords]
        z_center = (max(z_vals) + min(z_vals)) / 2.0
        final_coords = [[c[0], c[1], c[2] - z_center + 12.5] for c in cart_coords]
        
        mono_struct = Structure(new_lat, [s.specie for s in final_sites], final_coords, coords_are_cartesian=True)
        return mono_struct, gap_angstrom

    @staticmethod
    def extract_monolayer_miller(bulk_struct: Structure, hkl: tuple, num_layers: int = 1) -> Structure:
        """Cleaves a bulk structure along explicit Miller indices [h k l]."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bulk_struct.translate_sites(list(range(len(bulk_struct))), [0.01, 0.01, 0.01], to_unit_cell=True)
            approx_thick = max(1.5, num_layers * 3.0)
            slabgen = SlabGenerator(bulk_struct, miller_index=hkl, min_slab_size=approx_thick, min_vacuum_size=25.0, center_slab=True)
            slabs = slabgen.get_slabs()
            if not slabs:
                raise ValueError(f"Could not generate slabs for plane {hkl}")
            raw_slab = slabs[0]
            
            new_matrix = raw_slab.lattice.matrix.copy()
            new_matrix[2] = [0, 0, 25.0]
            new_lat = Lattice(new_matrix)
            
            cart_coords = raw_slab.cart_coords
            z_vals = [c[2] for c in cart_coords]
            z_center = (max(z_vals) + min(z_vals)) / 2.0
            final_coords = [[c[0], c[1], c[2] - z_center + 12.5] for c in cart_coords]
            return Structure(new_lat, raw_slab.species, final_coords, coords_are_cartesian=True)

    @staticmethod
    def build_heterostructure_stack(layers_data: list[dict]) -> Structure:
        """
        Rotates, shifts, and combines multiple 2D structures into a unified stacked supercell.
        Each dictionary in layers_data must contain: {'struct': Structure, 'sc_x': int, 'sc_y': int, 'z_shift': float, 'twist': float}
        """
        all_species, all_coords, layer_tags = [], [], []
        
        for idx, l in enumerate(layers_data):
            supercell = l['struct'] * (l['sc_x'], l['sc_y'], 1)
            theta = np.radians(l['twist'])
            rot_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [            0,              0, 1]
            ])
            
            coords = supercell.cart_coords.copy()
            center_xy = np.mean(coords[:, :2], axis=0)
            coords[:, :2] -= center_xy
            
            rotated_coords = np.dot(coords, rot_matrix.T)
            shifted_coords = rotated_coords + np.array([0, 0, l['z_shift'] - 12.5])
            layer_tag = f"_L{idx + 1}"
            
            for i, site in enumerate(supercell):
                all_species.append(site.specie.symbol)
                all_coords.append(shifted_coords[i])
                layer_tags.append(f"{site.specie.symbol}{layer_tag}")
                
        dummy_lattice = Lattice.cubic(500.0)
        return Structure(
            dummy_lattice, 
            all_species, 
            all_coords, 
            coords_are_cartesian=True,
            site_properties={"layer_tag": layer_tags}
        )

    @staticmethod
    def calculate_moire_superlattice(layer1_struct: Structure, layer2_struct: Structure, twist1: float, twist2: float) -> dict:
        """Calculates 2D Moiré superlattice periodicity and checks commensurability."""
        m1 = layer1_struct.lattice.matrix[:2, :2]
        m2 = layer2_struct.lattice.matrix[:2, :2]
        theta = np.radians(twist2 - twist1)
        
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        m2_rot = np.dot(m2, rot_matrix.T)

        try:
            g1 = np.linalg.inv(m1).T
            g2_rot = np.linalg.inv(m2_rot).T
        except np.linalg.LinAlgError:
            return {"status": "error", "message": "Degenerate 2D lattice detected."}

        delta_g = g1 - g2_rot
        try:
            m_moire = np.linalg.inv(delta_g).T
        except np.linalg.LinAlgError:
            return {"status": "perfect_alignment", "message": "Twist is 0° with identical lattices. No Moiré pattern."}

        S_matrix = np.dot(m_moire, np.linalg.inv(m1))
        S_round = np.round(S_matrix)
        is_commensurate = np.all(np.abs(S_matrix - S_round) < 0.05)
        periodicity = float(np.linalg.norm(m_moire[0]))

        if is_commensurate:
            n_cells = int(np.round(np.sqrt(np.abs(np.linalg.det(S_round)))))
            return {"status": "commensurate", "periodicity": periodicity, "n_cells": n_cells, "matrix": m_moire}
        else:
            return {"status": "incommensurate", "periodicity": periodicity, "matrix": m_moire}

    @staticmethod
    def calculate_brillouin_zone(structure: Structure) -> dict:
        """Calculates the 3D Wigner-Seitz cell of the primitive reciprocal lattice."""
        prim_matrix = CrystalEngine.get_universal_primitive_matrix(structure)
        prim_lat = Lattice(prim_matrix)
        recip_lat = prim_lat.reciprocal_lattice
        faces = recip_lat.get_wigner_seitz_cell()
        
        all_pts = []
        export_edges = set()
        for face in faces:
            for pt in face:
                all_pts.append(pt)
            for i in range(len(face)):
                p1 = tuple(np.round(face[i], 5))
                p2 = tuple(np.round(face[(i+1)%len(face)], 5))
                export_edges.add(tuple(sorted([p1, p2])))

        if not all_pts: return {}
        all_pts = np.array(all_pts)
        hull = ConvexHull(all_pts)
        
        return {
            "points": all_pts,
            "hull_points": hull.points,
            "simplices": hull.simplices,
            "edges": export_edges
        }
    
    @staticmethod
    def calculate_surface_projection(bz_points, structure, h, k, l):
        """Projects 3D BZ vertices onto a specific 2D Miller plane and extracts the 3D silhouette."""
        from scipy.spatial import ConvexHull
        import numpy as np
        
        recip_matrix = structure.lattice.reciprocal_lattice.matrix
        normal = h * recip_matrix[0] + k * recip_matrix[1] + l * recip_matrix[2]
        dist = np.linalg.norm(normal)
        if dist == 0: return None
        normal = normal / dist
        
        # Project points onto the plane
        proj_pts = []
        for pt in bz_points:
            d = np.dot(pt, normal)
            proj_pts.append(pt - d * normal)
        proj_pts = np.array(proj_pts)
        
        # Create a local 2D basis
        u = np.cross(normal, [0, 0, 1])
        if np.linalg.norm(u) < 1e-5: u = np.cross(normal, [0, 1, 0])
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        
        pts_2d = np.array([[np.dot(p, u), np.dot(p, v)] for p in proj_pts])
        pts_2d = np.round(pts_2d, decimals=5)
        unique_pts_2d, indices = np.unique(pts_2d, axis=0, return_index=True)
        
        if len(unique_pts_2d) < 3: return None
        
        surf_hull = ConvexHull(unique_pts_2d)
        
        # Extract the exact 3D vertices that form the outer boundary (silhouette)
        hull_indices = indices[surf_hull.vertices]
        surf_vertices_3d = proj_pts[hull_indices]
        silhouette_3d = bz_points[hull_indices]
        
        # Triangulate for rendering
        center = np.mean(surf_vertices_3d, axis=0)
        surf_pts_with_center = np.vstack([surf_vertices_3d, center])
        c_idx = len(surf_pts_with_center) - 1
        
        surf_simplices = []
        n_v = len(surf_vertices_3d)
        for i in range(n_v):
            surf_simplices.append([i, (i+1)%n_v, c_idx])
            
        return {
            "normal": normal.tolist(),
            "origin_plane": surf_pts_with_center.tolist(),
            "simplices": surf_simplices,
            "silhouette_3d": silhouette_3d.tolist(),
            "projected_bounds": surf_vertices_3d.tolist()
        }

    @staticmethod
    def get_hkl_surface_frame(hkl, recip_matrix, azimuthal_ref=None):
        """
        Calculates the orthogonal surface basis for a given (h, k, l) cleavage plane.
        Returns the normal vector (z-axis) and the up vector (y-axis) for camera or simulation alignment.
        """
        import numpy as np
        h, k, l = hkl
        
        # 1. Calculate surface normal
        normal = h * recip_matrix[0] + k * recip_matrix[1] + l * recip_matrix[2]
        dist = np.linalg.norm(normal)
        if dist == 0:
            return np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])
        normal = normal / dist
        
        # 2. Calculate the azimuthal 'up' vector
        if azimuthal_ref is not None:
            up_vector = azimuthal_ref - np.dot(azimuthal_ref, normal) * normal
        else:
            # --- FIX: Lock the default azimuthal frame to the reciprocal lattice ---
            b_star = recip_matrix[1]
            a_star = recip_matrix[0]
            
            # Use b* as the default "up" direction. If the surface normal is parallel to b*, fallback to a*.
            b_star_norm = b_star / np.linalg.norm(b_star)
            default_up = b_star if abs(np.dot(b_star_norm, normal)) < 0.99 else a_star
            
            up_vector = default_up - np.dot(default_up, normal) * normal
            
        up_vector = up_vector / np.linalg.norm(up_vector)
        
        return normal, up_vector

    @staticmethod
    def classify_stack_for_dft(layers: list[dict], twist_eps: float = 1e-6) -> str:
        if not layers:
            return "empty"
        twists = [abs(float(l.get("twist", 0.0))) for l in layers]
        any_twist = any(t > twist_eps for t in twists)
        if any_twist and len(layers) != 2:
            return "reject_multitwist"
        if any_twist:
            return "twist"
        return "aligned"

    @staticmethod
    def isotropic_match_strain_percent(a_native: float, a_ref: float) -> float:
        """Percent stretch of native a to match reference: (a_ref - a_native) / a_native * 100."""
        if abs(a_native) < 1e-12:
            raise ValueError("Degenerate native lattice parameter.")
        return float((a_ref - a_native) / a_native * 100.0)

    @staticmethod
    def expected_commensurate_atom_count(moire: dict, layers: list[dict]) -> float:
        """Expected atom count when tiling each layer into a commensurate moiré cell."""
        ab = np.asarray(moire["matrix"], dtype=float)
        area_m = abs(float(np.linalg.det(ab)))
        total = 0.0
        for layer in layers:
            layer_ab = CrystalEngine._inplane_2x2(
                layer["struct"], layer["sc_x"], layer["sc_y"]
            )
            area_l = abs(float(np.linalg.det(layer_ab)))
            if area_l < 1e-12:
                raise ValueError("Degenerate layer in-plane lattice for commensurate tiling.")
            n_atoms = len(layer["struct"]) * int(layer["sc_x"]) * int(layer["sc_y"])
            total += n_atoms * (area_m / area_l)
        return total

    @staticmethod
    def inplane_strain_percent(ref_2x2: np.ndarray, other_2x2: np.ndarray) -> float:
        ref = np.asarray(ref_2x2, dtype=float)
        other = np.asarray(other_2x2, dtype=float)
        denom = np.linalg.norm(ref, ord="fro")
        if denom < 1e-12:
            raise ValueError("Degenerate reference in-plane lattice.")
        return float(100.0 * np.linalg.norm(other - ref, ord="fro") / denom)

    @staticmethod
    def _inplane_2x2(struct: Structure, sc_x: int = 1, sc_y: int = 1) -> np.ndarray:
        m = struct.lattice.matrix.copy()
        m[0] *= sc_x
        m[1] *= sc_y
        return m[:2, :2]

    @staticmethod
    def suggest_reference_layer(layers: list[dict]) -> tuple[int, list[float]]:
        n = len(layers)
        if n == 0:
            raise ValueError("No layers for reference suggestion.")
        strains = []
        for r in range(n):
            ref = CrystalEngine._inplane_2x2(layers[r]["struct"], layers[r]["sc_x"], layers[r]["sc_y"])
            total = 0.0
            for j in range(n):
                if j == r:
                    continue
                other = CrystalEngine._inplane_2x2(layers[j]["struct"], layers[j]["sc_x"], layers[j]["sc_y"])
                # relative twist: rotate other in-plane by (twist_j - twist_r)
                dtheta = np.radians(float(layers[j]["twist"]) - float(layers[r]["twist"]))
                c, s = np.cos(dtheta), np.sin(dtheta)
                R = np.array([[c, -s], [s, c]])
                other_rot = other @ R.T
                total += CrystalEngine.inplane_strain_percent(ref, other_rot)
            strains.append(total)
        best = int(np.argmin(strains))
        # tie-break within 0.1% relative -> layer 0 preference among ties
        min_s = strains[best]
        ties = [
            i for i, s in enumerate(strains)
            if min_s <= 0 or abs(s - min_s) / min_s <= 1e-3
        ]
        if 0 in ties:
            best = 0
        else:
            best = ties[0]
        return best, strains

    @staticmethod
    def aligned_needs_strain_dialog(layers: list[dict], ref_idx: int, tol_percent: float = 1.0) -> bool:
        ref = CrystalEngine._inplane_2x2(layers[ref_idx]["struct"], layers[ref_idx]["sc_x"], layers[ref_idx]["sc_y"])
        for j, layer in enumerate(layers):
            if j == ref_idx:
                continue
            other = CrystalEngine._inplane_2x2(layer["struct"], layer["sc_x"], layer["sc_y"])
            if CrystalEngine.inplane_strain_percent(ref, other) > tol_percent:
                return True
        return False

    @staticmethod
    def _finalize_slab_structure(ab_2x2, species, carts_xyz, tags, vacuum_ang: float) -> Structure:
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

    @staticmethod
    def build_dft_aligned_stack(layers: list[dict], vacuum_ang: float, ref_idx: int = 0) -> Structure:
        if not layers:
            raise ValueError("No layers to build DFT stack.")
        if ref_idx < 0 or ref_idx >= len(layers):
            raise ValueError("ref_idx out of range.")
        ab = CrystalEngine._inplane_2x2(
            layers[ref_idx]["struct"], layers[ref_idx]["sc_x"], layers[ref_idx]["sc_y"]
        )
        species, carts, tags = [], [], []
        for idx, l in enumerate(layers):
            supercell = l["struct"] * (l["sc_x"], l["sc_y"], 1)
            # Strain via fractional xy on the ref ab lattice. Do NOT center+wrap:
            # independent PBC wrap after centering tears in-plane bonds (e.g. C–C ~3 Å).
            z0 = float(l["z_shift"]) - 12.5
            z_mean = float(np.mean(supercell.cart_coords[:, 2]))
            for site in supercell:
                frac_xy = np.asarray(site.frac_coords[:2], dtype=float)
                frac_xy = frac_xy - np.floor(frac_xy + 1e-12)
                xy_in_ref = frac_xy @ ab  # ab rows = lattice vectors
                z = float(site.coords[2]) - z_mean + z0
                species.append(site.specie.symbol)
                carts.append([float(xy_in_ref[0]), float(xy_in_ref[1]), z])
                tags.append(f"{site.specie.symbol}_L{idx + 1}")
        return CrystalEngine._finalize_slab_structure(ab, species, carts, tags, vacuum_ang)

    @staticmethod
    def _place_layer_atoms(layer: dict, idx: int, apply_twist: bool) -> tuple[list, list, list]:
        """Return species, cartesian coords, tags for one layer (xy centered, z from spinbox)."""
        supercell = layer["struct"] * (layer["sc_x"], layer["sc_y"], 1)
        coords = supercell.cart_coords.copy()
        center_xy = np.mean(coords[:, :2], axis=0)
        coords[:, :2] -= center_xy
        if apply_twist:
            theta = np.radians(float(layer["twist"]))
            R = np.array([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta),  np.cos(theta), 0],
                          [0, 0, 1]])
            coords = coords @ R.T
        coords[:, 2] = coords[:, 2] - np.mean(coords[:, 2]) + (float(layer["z_shift"]) - 12.5)
        species, carts, tags = [], [], []
        for i, site in enumerate(supercell):
            species.append(site.specie.symbol)
            carts.append(coords[i].tolist())
            tags.append(f"{site.specie.symbol}_L{idx + 1}")
        return species, carts, tags

    @staticmethod
    def _supercell_shifts(moire_ab, layer_ab) -> list:
        """Integer layer-lattice shifts for one moiré cell via rounded S = moiré @ inv(layer)."""
        moire_ab = np.asarray(moire_ab, dtype=float)
        layer_ab = np.asarray(layer_ab, dtype=float)
        S = np.round(moire_ab @ np.linalg.inv(layer_ab))
        det = int(abs(round(np.linalg.det(S))))
        if det < 1:
            return []
        Sinv = np.linalg.inv(S)
        n_max = int(np.max(np.abs(S))) + 3
        shifts = []
        for n1 in range(-2 * n_max, 2 * n_max + 1):
            for n2 in range(-2 * n_max, 2 * n_max + 1):
                u = Sinv @ np.array([n1, n2], dtype=float)
                if np.all(u >= -1e-10) and np.all(u < 1.0 - 1e-10):
                    shifts.append(n1 * layer_ab[0] + n2 * layer_ab[1])
        return shifts

    @staticmethod
    def _tile_into_cell(moire_ab, layer_ab, species, carts, tags) -> tuple[list, list, list]:
        """Replicate on layer lattice using rounded-S supercell; fold into moiré [0,1)."""
        moire_ab = np.asarray(moire_ab, dtype=float)
        layer_ab = np.asarray(layer_ab, dtype=float)
        moire_inv_t = np.linalg.inv(moire_ab.T)
        shifts = CrystalEngine._supercell_shifts(moire_ab, layer_ab)
        if not shifts:
            return [], [], []

        out_s, out_c, out_t = [], [], []
        seen = set()
        for shift in shifts:
            for s, c, t in zip(species, carts, tags):
                xy = np.asarray(c[:2], dtype=float) + shift
                frac = xy @ moire_inv_t
                frac = frac - np.floor(frac + 1e-12)
                frac = np.where(frac > 1.0 - 1e-10, 0.0, frac)
                key = (round(float(frac[0]), 6), round(float(frac[1]), 6), s, t)
                if key in seen:
                    continue
                seen.add(key)
                xy_f = frac @ moire_ab.T
                out_s.append(s)
                out_c.append([float(xy_f[0]), float(xy_f[1]), float(c[2])])
                out_t.append(t)
        return out_s, out_c, out_t

    @staticmethod
    def build_dft_twist_stack(
        layers: list[dict],
        vacuum_ang: float,
        ref_idx: int,
        stacking: str = "AA",
        max_cells: int = 217,
        valley: str = "K",
    ) -> tuple[Structure, dict]:
        if len(layers) != 2:
            raise ValueError("Twist DFT rebuild requires exactly 2 layers.")
        for i, layer in enumerate(layers):
            sc_x, sc_y = int(layer["sc_x"]), int(layer["sc_y"])
            if sc_x != 1 or sc_y != 1:
                raise ValueError(
                    f"Twisted DFT cells require SC = 1×1 on every layer "
                    f"(layer {i + 1} has {sc_x}×{sc_y}). "
                    "Set supercell to 1×1 before Push; moiré tiling already expands the cell. "
                    "Aligned (zero-twist) N-layer stacks may still use SC ≠ 1."
                )
        from tensorspec.core.bernevig_twist import build_bernevig_bilayer

        theta = float(layers[1]["twist"]) - float(layers[0]["twist"])
        struct, info = build_bernevig_bilayer(
            layers[0]["struct"],
            layers[1]["struct"],
            theta_deg=theta,
            stacking=stacking,
            z_bottom=float(layers[0]["z_shift"]),
            z_top=float(layers[1]["z_shift"]),
            vacuum_ang=float(vacuum_ang),
            max_cells=int(max_cells),
            valley=valley,
        )
        return struct, info
