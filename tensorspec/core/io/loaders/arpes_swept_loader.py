"""
ARPES loader for Swept Spectra format (nanoARPES files from ALS).

Handles Maestro HDF5 files with ARPES spectroscopy data in 2D_Data/Swept_Spectra*
along with 0D metadata (temperatures, voltages) and Headers.
"""

import h5py
import numpy as np


class ARPESSweptLoader:
    """Load ARPES data from 2D_Data Swept Spectra format (WTe2 M28 dataset)."""

    def __init__(self, filepath: str):
        self.filepath = str(filepath)

    def load(self) -> dict:
        """
        Load ARPES swept spectra data from HDF5 file.

        WTe2 M28 files contain:
        - 2D_Data/Swept_Spectra*: (energy_bins, spatial_points, sweep_dimension)
        - 0D_Data/*: Temperature sensors and voltage measurements
        - Headers/*: Experimental parameters

        Returns dict with keys:
        - data: numpy array of shape (E, X, param) where param is gate voltage or other sweep
        - labels: axis labels ['Energy (eV)', 'Position (spatial)', 'Gate Voltage (V)']
        - axes: coordinate arrays
        - metadata: extracted parameters
        - mode: 'ARPES'
        """
        with h5py.File(self.filepath, 'r') as f:
            # Check for Maestro signature
            if 'Headers' not in f:
                raise ValueError("Not a Maestro file (missing Headers)")

            # Look for swept spectra data
            if '2D_Data' not in f:
                raise ValueError("No 2D_Data group found")

            two_d = f['2D_Data']
            spectra_key = None

            # Find Swept_Spectra* dataset
            for key in two_d.keys():
                if key.startswith('Swept_Spectra'):
                    spectra_key = key
                    break

            if spectra_key is None:
                raise ValueError(f"No Swept_Spectra* dataset in 2D_Data. Found: {list(two_d.keys())}")

            spectra = np.array(two_d[spectra_key])  # Shape: (energy_bins, spatial_points, sweep_dim)

            # Extract metadata
            metadata = self._extract_metadata(f, spectra_key)

            # Create coordinate arrays
            energy_bins = spectra.shape[0]
            spatial_points = spectra.shape[1]
            sweep_points = spectra.shape[2] if len(spectra.shape) > 2 else 1

            return {
                'data': spectra,
                'axes': [
                    np.arange(energy_bins),           # Energy/binding energy bins
                    np.arange(spatial_points),        # Spatial points (detector X)
                    np.arange(sweep_points),          # Sweep dimension (gate voltage or parameter)
                ],
                'labels': [
                    'Energy (bins)',
                    'Position (detector X)',
                    'Gate Voltage (sweep index)',
                ],
                'units': ['bin', 'index', 'index'],
                'metadata': metadata,
                'mode': 'ARPES',
                'is_fixed': False,  # Swept data
                'facility': 'ALS nanoARPES',
            }

    def _extract_metadata(self, f: h5py.File, spectra_key: str) -> dict:
        """Extract experimental parameters from headers and 0D data."""
        meta = {}

        # Extract from 0D_Data (temperatures, voltages, sweep info)
        if '0D_Data' in f:
            meta['temperatures'] = {}
            meta['voltages'] = {}
            meta['other_0d'] = {}

            for key in f['0D_Data'].keys():
                dataset = f['0D_Data'][key]
                vals = np.array(dataset[()])

                if 'nanoARPES' in key:
                    # Temperature sensors
                    if len(vals) > 1:
                        meta['temperatures'][key] = {
                            'mean': float(np.mean(vals)),
                            'min': float(np.min(vals)),
                            'max': float(np.max(vals)),
                        }
                    else:
                        meta['temperatures'][key] = float(vals[0]) if len(vals) > 0 else 0.0

                elif 'SMU' in key and '_Measure_' in key:
                    # Source-Measure Unit current
                    if len(vals) > 1:
                        meta['voltages'][key] = {
                            'mean': float(np.mean(vals)),
                            'min': float(np.min(vals)),
                            'max': float(np.max(vals)),
                        }
                    else:
                        meta['voltages'][key] = float(vals[0]) if len(vals) > 0 else 0.0

                elif 'SMU' in key or 'Out_V' in key:
                    # Gate/drain voltages
                    if len(vals) > 1:
                        meta['voltages'][key] = {
                            'mean': float(np.mean(vals)),
                            'min': float(np.min(vals)),
                            'max': float(np.max(vals)),
                        }
                    else:
                        meta['voltages'][key] = float(vals[0]) if len(vals) > 0 else 0.0

                elif 'Slit' in key or 'Theta' in key or 'Motor' in key:
                    # Scan parameters
                    if len(vals) > 1:
                        meta['other_0d'][key] = {
                            'mean': float(np.mean(vals)),
                            'min': float(np.min(vals)),
                            'max': float(np.max(vals)),
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
                    elif 'MONO' in name or 'GPD' in name:
                        meta[f'beamline_{name.lower()}'] = value

            # DAQ/Sweep parameters
            if 'DAQ_Swept' in headers:
                daq = headers['DAQ_Swept'][()]
                for entry in daq:
                    name = entry[1].decode('utf-8') if isinstance(entry[1], bytes) else str(entry[1])
                    value = entry[2].decode('utf-8') if isinstance(entry[2], bytes) else str(entry[2])

                    if 'ENERGY' in name or 'PASS' in name:
                        meta['daq_sweep'] = value

            # Sample info
            if 'Notebook' in headers:
                notebook = headers['Notebook'][()]
                for entry in notebook:
                    name = entry[1].decode('utf-8') if isinstance(entry[1], bytes) else str(entry[1])
                    value = entry[2].decode('utf-8') if isinstance(entry[2], bytes) else str(entry[2])
                    meta[f'notebook_{name.lower()}'] = value

        # Extract sweep dimension info from spectra key
        meta['spectra_key'] = spectra_key

        return meta


__all__ = ['ARPESSweptLoader']
