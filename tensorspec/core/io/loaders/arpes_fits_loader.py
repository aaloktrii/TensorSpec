"""
ARPES loader for FITS format (ALS nanoARPES files).

Reads temperature data and experimental parameters from FITS binary tables.
"""

import numpy as np
from astropy.io import fits


class ARPESFitsLoader:
    """Load ARPES metadata from FITS files."""

    def __init__(self, filepath: str):
        self.filepath = str(filepath)

    def load(self) -> dict:
        """
        Load ARPES metadata from FITS file.

        FITS files from nanoARPES contain:
        - PRIMARY HDU: Header with beamline parameters
        - data HDU: Binary table with temperatures, voltages, detector data

        Returns dict with metadata (FITS files provide parameters, not spectra).
        """
        with fits.open(self.filepath) as hdul:
            # Get primary header
            primary_header = hdul[0].header

            # Get data table
            if len(hdul) < 2:
                raise ValueError("FITS file missing data HDU")

            data_hdu = hdul[1]
            if data_hdu.data is None:
                raise ValueError("FITS data table is empty")

            # Extract metadata
            metadata = self._extract_metadata(primary_header, data_hdu)

            # Create a simple metadata-only tensor
            # (FITS files don't contain spectra in the expected format)
            num_points = len(data_hdu.data)

            return {
                'data': np.arange(num_points),  # Index array as placeholder
                'axes': [np.arange(num_points)],
                'labels': ['Measurement Index'],
                'units': ['index'],
                'metadata': metadata,
                'mode': 'ARPES-Metadata',  # Metadata-only mode
                'is_fixed': True,
                'facility': 'ALS nanoARPES',
            }

    def _extract_metadata(self, header, data_hdu) -> dict:
        """Extract experimental parameters from FITS header and table."""
        meta = {
            'temperatures': {},
            'voltages': {},
            'other_0d': {},
        }

        # From primary header
        if 'MONOEV' in header:
            meta['photon_energy_eV'] = float(header['MONOEV'])

        if 'UNDGAP' in header:
            meta['undulator_gap'] = float(header['UNDGAP'])

        # From data table columns
        try:
            for col_name in data_hdu.columns.names:
                col_data = data_hdu.data[col_name]

                if 'nanoARPES' in col_name or 'TEMP' in col_name:
                    # Temperature sensors
                    meta['temperatures'][col_name] = {
                        'mean': float(np.mean(col_data)),
                        'min': float(np.min(col_data)),
                        'max': float(np.max(col_data)),
                    }
                elif 'SMU' in col_name or 'Volt' in col_name or '_V' in col_name:
                    # Voltage/current measurements
                    meta['voltages'][col_name] = {
                        'mean': float(np.mean(col_data)),
                        'min': float(np.min(col_data)),
                        'max': float(np.max(col_data)),
                    }
                else:
                    # Other 0D data
                    if np.isscalar(col_data[0]):
                        meta['other_0d'][col_name] = {
                            'mean': float(np.mean(col_data)),
                            'shape': col_data.shape,
                        }

        except Exception as e:
            print(f"Warning: Could not extract all metadata from FITS: {e}")

        return meta


__all__ = ['ARPESFitsLoader']
