"""
CENTRAL MEMORY: Global dictionary/manager for all active loaded data.
Strictly pure Python logic. Zero GUI/Plotting imports.
"""
import numpy as np
from pathlib import Path
from tensorspec.core.data_models import TensorData
from tensorspec.core.data_tree import DataTreeBuilder
from tensorspec.core.arpes.photon_energy_scan import tensor_from_stacked_sim

class WorkspaceManager:
    def __init__(self):
        # The primary dictionary holding all datasets (CIFs, DataTrees, etc.)
        self._data = {}
        
        # Set up a default root directory for the project to prevent saving errors
        self.project_dir = Path.cwd() / "TensorSpec_Workspace"
        self.project_dir.mkdir(parents=True, exist_ok=True)

    def push_crystal_structure(self, name, basis_vectors):
        """
        Stores a parsed crystal structure's local basis vectors.
        This will later be called by the Crystal Suite after loading a CIF.
        """
        self._data[name] = {
            'type': 'crystal_structure',
            'basis': basis_vectors
        }

    def pull_crystal_structure(self, name):
        """
        Retrieves the basis vectors of a stored crystal structure.
        """
        item = self._data.get(name)
        if item and item.get('type') == 'crystal_structure':
            return item['basis']
        return None
    
    def list_crystal_structures(self):
        """Names of entries pushed via ``push_crystal_structure`` only."""
        return [
            key
            for key, value in self._data.items()
            if isinstance(value, dict) and value.get("type") == "crystal_structure"
        ]
    
    def push_remote_run(self, name, cluster_name, remote_path, engine, meta=None):
        """
        Stores a pointer to a remote DFT calculation output folder.
        engine should be 'SPRKKR' or 'QE'. ``meta`` (optional dict) carries
        extra provenance -- e.g. SPR-KKR's CIF lattice matrix for surface
        geometry resolution (see tensorspec.core.dft.sprkkr.workflow).
        """
        self._data[name] = {
            'type': 'remote_run',
            'cluster': cluster_name,
            'remote_path': remote_path,
            'engine': engine,
            'meta': meta or {},
        }

    def pull_remote_run(self, name):
        """Retrieves the remote run pointer dictionary."""
        item = self._data.get(name)
        if item and item.get('type') == 'remote_run':
            return item
        return None

    def list_remote_runs(self, engine=None):
        """Returns a list of remote DFT vault names, optionally filtered by engine."""
        runs = []
        for key, value in self._data.items():
            if value.get('type') == 'remote_run':
                if engine is None or value.get('engine') == engine:
                    runs.append(key)
        return runs

    def push_band_structure(self, name, k_dist, eigenvalues, eigenvectors, k_vecs, node_idx, labels, orbital_positions=None):
        """
        Stores a calculated band structure, its wavefunctions, and basis coordinates.
        """
        self._data[name] = {
            'type': 'band_structure',
            'k_dist': k_dist,               # 1D array for x-axis plotting
            'eigenvalues': eigenvalues,     # Energy bands E(k)
            'eigenvectors': eigenvectors,   # Orbital characters/wavefunctions
            'k_vecs': k_vecs,               # Actual 3D k-vectors for matrix elements
            'node_idx': node_idx,           # High symmetry point indices
            'labels': labels,               # High symmetry labels
            'orbital_positions': orbital_positions or [] # NEW: Atomic basis coords for ARPES ME
        }

    def pull_band_structure(self, name):
        """Retrieves the band structure dictionary."""
        item = self._data.get(name)
        if item and item.get('type') == 'band_structure':
            return item
        return None

    def list_band_structures(self):
        """Returns a list of all currently loaded band structure names."""
        return [k for k, v in self._data.items() if v.get('type') == 'band_structure']

    def save_simulated_arpes(
        self, name, intensity, kx, ky, E, metadata=None, hv=None
    ):
        """Saves a simulated ARPES dataset to a compressed numpy archive."""
        # Create the data directory if it doesn't exist
        arpes_dir = self.project_dir / "arpes_data" / "simulated"
        arpes_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = arpes_dir / f"{name}.npz"
        
        payload = {
            "intensity": intensity,
            "kx": kx,
            "ky": ky,
            "E": E,
            "metadata": metadata if metadata else {},
        }
        if hv is not None:
            payload["hv"] = hv
        np.savez_compressed(file_path, **payload)
        print(f"Saved simulated ARPES data to: {file_path}")
        return file_path
    
    def load_simulated_arpes_to_tensor(self, filename) -> TensorData:
        """
        Loads a simulated .npz file and packages it into the agnostic TensorData format.
        """
        file_path = self.project_dir / "arpes_data" / "simulated" / filename
        
        with np.load(file_path, allow_pickle=True) as data:
            intensity = data['intensity']  # Original shape from Chinook: (kx, ky, E)
            metadata = data['metadata'].item() if 'metadata' in data else {}

            if "hv" in data:
                return tensor_from_stacked_sim(
                    intensity,
                    data["hv"],
                    data["kx"],
                    data["ky"],
                    data["E"],
                    metadata=metadata,
                )
            
            # For the N-Dimensional viewer, we usually want Energy on the 0th axis
            # Transpose (kx, ky, E) -> (E, kx, ky)
            value_matrix = np.transpose(intensity, (2, 0, 1))
            
            # Package it into the universal format
            return TensorData(
                value=value_matrix,
                axes=[data['E'], data['kx'], data['ky']],
                labels=["Energy", "Slit Angle", "Deflection Angle"],
                units=["eV", "deg", "deg"],
                data_type="Simulated ARPES",
                metadata=metadata
            )
    def push_spectroscopy_data(self, name: str, tensor_data: TensorData):
        """
        Converts raw TensorData into a strict xarray.DataTree and stores it in memory.
        """
        tree = DataTreeBuilder.build_from_tensor(name, tensor_data)
        self._data[name] = {
            'type': 'spectroscopy_tree',
            'tree': tree
        }
        print(f"DataTree '{name}' successfully pushed to Global Workspace.")

    def push_tensor_data(self, name: str, tensor_data: TensorData):
        """Alias for push_spectroscopy_data."""
        self.push_spectroscopy_data(name, tensor_data)

    def merge_spectroscopy_raw_attrs(self, name: str, attrs: dict) -> bool:
        """Merge metadata into an existing spectroscopy tree's ``/raw`` node."""
        item = self._data.get(name)
        if not item or item.get('type') != 'spectroscopy_tree':
            return False
        item['tree'] = DataTreeBuilder.merge_raw_attrs(item['tree'], attrs)
        return True

    def write_processed_data(self, name: str, tensor_data: TensorData) -> bool:
        """Write a processed cube into ``/processed`` of an existing spectroscopy tree."""
        item = self._data.get(name)
        if not item or item.get('type') != 'spectroscopy_tree':
            return False
        item['tree'] = DataTreeBuilder.write_processed(item['tree'], tensor_data)
        return True

    def write_processed_child_data(self, name: str, child_name: str, tensor_data: TensorData) -> bool:
        item = self._data.get(name)
        if not item or item.get("type") != "spectroscopy_tree":
            return False
        item["tree"] = DataTreeBuilder.write_processed_child(
            item["tree"], child_name, tensor_data
        )
        return True

    def list_processed_children(self, name: str) -> list[str]:
        item = self._data.get(name)
        if not item or item.get("type") != "spectroscopy_tree":
            return []
        return DataTreeBuilder.list_processed_children(item["tree"])

    def write_analysis_data(self, name: str, node_name: str, dataset) -> bool:
        """Write an analysis Dataset under ``/analysis/<node_name>``."""
        item = self._data.get(name)
        if not item or item.get('type') != 'spectroscopy_tree':
            return False
        item['tree'] = DataTreeBuilder.write_analysis(item['tree'], node_name, dataset)
        return True

    def pull_analysis_data(self, name: str, node_name: str = "mdc_peakfit"):
        """Return an analysis Dataset, or None."""
        item = self._data.get(name)
        if not item or item.get('type') != 'spectroscopy_tree':
            return None
        tree = item['tree']
        leaf = node_name.strip("/")
        try:
            analysis = tree["analysis"]
            if leaf in analysis.children:
                node = analysis[leaf]
            else:
                return None
        except Exception:
            return None
        return node.to_dataset() if hasattr(node, "to_dataset") else node.ds

    def pull_tensor_data(self, name: str, node: str = "raw") -> TensorData:
        """
        Extracts a specific node from a stored DataTree and packages it back 
        into a TensorData object for the DataViewerPanel to consume.
        """
        item = self._data.get(name)
        if not item or item.get('type') != 'spectroscopy_tree':
            return None
            
        tree = item['tree']
        node = node.strip("/")
        try:
            target = tree[node]
        except KeyError:
            print(f"Error: Node '{node}' does not exist in dataset '{name}'.")
            return None

        ds = target.to_dataset() if hasattr(target, "to_dataset") else target
        if "data" not in ds:
            return None
            
        da = ds["data"]
        
        # Package it back to our universal mathematical format
        return TensorData(
            value=da.values,
            axes=[da.coords[dim].values for dim in da.dims],
            labels=list(da.dims),
            units=[da.coords[dim].attrs.get("units", "") for dim in da.dims],
            data_type=da.attrs.get("long_name", "Unknown"),
            metadata=ds.attrs
        )

    def get(self, name: str, default=None):
        """Returns the raw dictionary item for a given name."""
        return self._data.get(name, default)

    def remove(self, name: str):
        """Removes an item from the in-memory workspace."""
        if name in self._data:
            del self._data[name]
            print(f"Removed '{name}' from workspace.")

# Instantiate the global singleton to be imported across the application
global_workspace = WorkspaceManager()