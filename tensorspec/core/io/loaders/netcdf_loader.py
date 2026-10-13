"""
NetCDF ARPES Data Loader
Loads standard NetCDF (.nc) files with 2D/3D ARPES data
"""

import numpy as np
import os
from pathlib import Path

try:
    import netCDF4
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False
    try:
        import scipy.io.netcdf as netcdf_scipy
        HAS_NETCDF_SCIPY = True
    except ImportError:
        HAS_NETCDF_SCIPY = False


class NetCDFLoader:
    """Load NetCDF format ARPES data files (.nc)"""

    def __init__(self, filepath):
        self.filepath = str(filepath)
        self.filename = os.path.basename(self.filepath)

        if not HAS_NETCDF4 and not HAS_NETCDF_SCIPY:
            raise ImportError("NetCDF support requires netCDF4 or scipy. Install with: pip install netCDF4")

    def load(self):
        """Load NetCDF file and return data dict for TensorSpec"""

        # Signature check
        if not self.filename.lower().endswith('.nc'):
            raise ValueError("Not a NetCDF file (.nc)")

        # Open with netCDF4 (preferred) or scipy fallback
        if HAS_NETCDF4:
            return self._load_netcdf4()
        else:
            return self._load_scipy()

    def _load_netcdf4(self):
        """Load using netCDF4 library"""
        import netCDF4

        with netCDF4.Dataset(self.filepath, 'r') as ds:
            # Auto-detect dimensions and variables
            dims = list(ds.dimensions.keys())
            vars = list(ds.variables.keys())

            print(f"[{self.filename}] NetCDF Dimensions: {dims}")
            print(f"[{self.filename}] NetCDF Variables: {vars}")

            # Find main data variable (largest 2D/3D array)
            data_var = self._find_data_variable(ds, vars)
            raw_data = ds[data_var][()]

            # Extract axes
            axes = self._extract_axes(ds, dims, raw_data.shape)

            # Extract attributes
            mode_string = ds.getncattr('technique') if 'technique' in ds.ncattrs() else 'ARPES'
            metadata = {key: ds.getncattr(key) for key in ds.ncattrs()}

            print(f"[{self.filename}] Successfully loaded via NetCDFLoader")
            print(f"[{self.filename}] Data shape: {raw_data.shape}, Mode: {mode_string}")

            return {
                "name": Path(self.filename).stem,
                "data": raw_data,
                "axes": axes,
                "mode": mode_string,
                "metadata": metadata,
                "is_fixed": False
            }

    def _load_scipy(self):
        """Fallback loader using scipy.io.netcdf"""
        from scipy.io.netcdf import netcdf_file

        with netcdf_file(self.filepath, 'r') as f:
            dims = list(f.dimensions.keys())
            vars = list(f.variables.keys())

            # Find main data variable
            data_var = self._find_data_variable(f, vars)
            raw_data = f.variables[data_var][:].copy()

            # Extract axes
            axes = self._extract_axes(f, dims, raw_data.shape)

            # Extract attributes
            mode_string = getattr(f, 'technique', 'ARPES')

            print(f"[{self.filename}] Successfully loaded via scipy NetCDF")

            return {
                "name": Path(self.filename).stem,
                "data": raw_data,
                "axes": axes,
                "mode": mode_string,
                "is_fixed": False
            }

    def _find_data_variable(self, ds, var_list, preferred_shape=None):
        """Identify main data variable (largest array)"""

        # Priority: look for 'data', 'intensity', 'counts', 'signal'
        priority_names = ['data', 'intensity', 'counts', 'signal', 'ARPES', 'arpes']

        for name in priority_names:
            if name in var_list:
                return name

        # Find largest variable
        max_size = 0
        max_var = None

        for var_name in var_list:
            try:
                var_size = np.prod(ds[var_name].shape)
                if var_size > max_size and var_size > 100:  # Skip tiny arrays
                    max_size = var_size
                    max_var = var_name
            except:
                pass

        if max_var is None:
            raise ValueError(f"Could not find data variable in {var_list}")

        return max_var

    def _extract_axes(self, ds, dims, data_shape):
        """Extract coordinate axes from NetCDF file"""

        axes = {}

        # Try to match dimensions to data shape
        for i, dim_name in enumerate(dims):
            if i < len(data_shape):
                if dim_name in ds.variables:
                    axis_data = ds[dim_name][()]
                    if len(axis_data) == data_shape[i]:
                        # Determine axis label
                        if 'energy' in dim_name.lower() or 'e' in dim_name.lower():
                            label = f"Energy ({dim_name})"
                        elif 'k' in dim_name.lower() or 'momentum' in dim_name.lower():
                            label = f"Momentum ({dim_name})"
                        elif 'angle' in dim_name.lower() or 'theta' in dim_name.lower():
                            label = f"Angle ({dim_name})"
                        else:
                            label = dim_name

                        axes[label] = axis_data
                else:
                    # Create default axis
                    axes[dim_name] = np.arange(data_shape[i])

        # If not enough axes, create defaults
        if len(axes) < len(data_shape):
            for i in range(len(axes), len(data_shape)):
                axes[f"Axis_{i}"] = np.arange(data_shape[i])

        return axes


# Support function for ARPESLoader
def is_netcdf_file(filepath):
    """Check if file is NetCDF format"""
    return str(filepath).lower().endswith('.nc')
