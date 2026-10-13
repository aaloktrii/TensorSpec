"""
ARPES loader for 1D_Data fixed spectra format (nanoARPES files from ALS).

Handles Maestro HDF5 files where spectra are stored in 1D_Data/Fixed_Spectra*
instead of the standard 2D_Data/Process_000 location.
"""

import h5py
import numpy as np
from tensorspec.core.io.loaders.maestro.errors import MaestroSignatureError


class ARPES1DLoader:
    """Load ARPES data from 1D_Data fixed spectra format."""

    def __init__(self, filepath: str):
        self.filepath = str(filepath)

    def load(self) -> dict:
        """
        Load ARPES fixed spectra data from HDF5 file.

        Returns dict with keys:
        - data: numpy array of spectra
        - labels: axis labels
        - axes: coordinate arrays
        - metadata: extracted parameters
        - mode: 'ARPES'
        """
        with h5py.File(self.filepath, 'r') as f:
            # Check for Maestro signature
            if 'Headers' not in f:
                raise ValueError("Not a Maestro file (missing Headers)")

            # Look for fixed spectra data
            if '1D_Data' not in f:
                raise ValueError("No 1D_Data group found")

            one_d = f['1D_Data']
            spectra_key = None

            # Find Fixed_Spectra* dataset
            for key in one_d.keys():
                if key.startswith('Fixed_Spectra'):
                    spectra_key = key
                    break

            if spectra_key is None:
                raise ValueError(f"No Fixed_Spectra* dataset in 1D_Data. Found: {list(one_d.keys())}")

            spectra = np.array(one_d[spectra_key])  # Shape: (energy_bins, spatial_points)

            # Extract metadata
            metadata = self._extract_metadata(f)

            # Create coordinate arrays
            energy_bins = spectra.shape[0]
            spatial_points = spectra.shape[1]

            return {
                'data': spectra,
                'axes': [
                    np.arange(energy_bins),           # Energy/binding energy bins
                    np.arange(spatial_points),        # Spatial points (scan X, Optics Stage)
                ],
                'labels': [
                    'Energy (bins)',
                    'Spatial (scan index)',
                ],
                'units': ['bin', 'index'],
                'metadata': metadata,
                'mode': 'ARPES',
                'is_fixed': True,
                'facility': 'ALS nanoARPES',
            }

    def _extract_metadata(self, f: h5py.File) -> dict:
        """Extract experimental parameters from headers."""
        meta = {}

        # Extract from 0D_Data if available
        if '0D_Data' in f:
            meta['temperatures'] = {}
            meta['voltages'] = {}
            meta['motors'] = {}

            for key in f['0D_Data'].keys():
                dataset = f['0D_Data'][key]

                if 'nanoARPES' in key:
                    # Temperature sensors
                    meta['temperatures'][key] = float(np.mean(dataset[()]))
                elif 'SMU' in key or 'V_' in key:
                    # Voltages
                    meta['voltages'][key] = float(np.mean(dataset[()]))
                elif any(m in key for m in ['Scan', 'Optics', 'Motor']):
                    # Motor positions
                    meta['motors'][key] = {
                        'mean': float(np.mean(dataset[()])),
                        'min': float(np.min(dataset[()])),
                        'max': float(np.max(dataset[()])),
                    }

        # Extract from Headers
        if 'Headers' in f:
            headers = f['Headers']

            # Beamline parameters
            if 'Beamline' in headers:
                beamline = headers['Beamline'][()]
                for entry in beamline:
                    name = entry[1].decode('utf-8') if isinstance(entry[1], bytes) else str(entry[1])
                    value = entry[2].decode('utf-8') if isinstance(entry[2], bytes) else str(entry[2])

                    if 'PHOTON' in name or 'ENERGY' in name or 'MONOEV' in name:
                        meta['photon_energy'] = value
                    elif 'TEMP' in name:
                        meta['temperature'] = value

            # Low level scan mode
            if 'Low_Level_Scan' in headers:
                scan = headers['Low_Level_Scan'][()]
                for entry in scan:
                    name = entry[1].decode('utf-8') if isinstance(entry[1], bytes) else str(entry[1])
                    value = entry[2].decode('utf-8') if isinstance(entry[2], bytes) else str(entry[2])
                    if 'LWLVNM' in name:
                        meta['scan_mode'] = value

            # Sample/project info
            if 'Notebook' in headers:
                notebook = headers['Notebook'][()]
                for entry in notebook:
                    name = entry[1].decode('utf-8') if isinstance(entry[1], bytes) else str(entry[1])
                    value = entry[2].decode('utf-8') if isinstance(entry[2], bytes) else str(entry[2])
                    meta[f'notebook_{name.lower()}'] = value

        return meta


__all__ = ['ARPES1DLoader']
