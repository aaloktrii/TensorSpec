import os
from .loaders.mock_data_loader import MockDataLoader
from .loaders.maestro_loader import MaestroLoader
from .loaders.arpes_swept_loader import ARPESSweptLoader
from .loaders.arpes_1d_loader import ARPES1DLoader
from .loaders.arpes_fits_loader import ARPESFitsLoader
from .loaders.netcdf_loader import NetCDFLoader
from .loaders.matlab_loader import MatlabLoader
from .loaders.maestro import MaestroSignatureError
from tensorspec.core.data_models import TensorData


def _units_from_labels(labels):
    return [
        "eV" if "eV" in label else "deg" if "deg" in label else "a.u."
        for label in labels
    ]


def _pack_tensor_data(raw_dict):
    if "labels" in raw_dict:
        labels = raw_dict["labels"]
        axes = raw_dict["axes"]
        units = raw_dict.get("units") or _units_from_labels(labels)
    else:
        axes_dict = raw_dict["axes"]
        labels = list(axes_dict.keys())
        axes = list(axes_dict.values())
        units = _units_from_labels(labels)

    metadata = dict(raw_dict.get("metadata") or {})
    metadata.setdefault("facility", raw_dict.get("facility"))
    metadata.setdefault("is_fixed", raw_dict.get("is_fixed"))

    return TensorData(
        value=raw_dict["data"],
        axes=axes,
        labels=labels,
        units=units,
        data_type=raw_dict.get("mode", "ARPES"),
        metadata=metadata,
    )


class ARPESLoader:
    """
    Main agnostic IO Manager.
    Passes the file through loaders, then packages it into the universal TensorData format.
    """

    @classmethod
    def load(cls, filepath):
        filepath_str = str(filepath)
        filename = os.path.basename(filepath_str)

        available_loaders = [
            MockDataLoader,
            MaestroLoader,
            ARPESSweptLoader,     # ARPES swept spectra (2D_Data HDF5 format) — WTe2 M28
            ARPES1DLoader,        # ARPES fixed spectra (1D_Data HDF5 format)
            ARPESFitsLoader,      # ARPES metadata (FITS format)
            MatlabLoader,         # MATLAB .mat files (v4–v7.3)
            NetCDFLoader,         # NetCDF format (scientific data)
        ]

        for LoaderClass in available_loaders:
            try:
                loader = LoaderClass(filepath_str)
                raw_dict = loader.load()
                return _pack_tensor_data(raw_dict)

            except MaestroSignatureError:
                continue
            except ValueError as e:
                # Allow specific loaders to fail and try next loader
                if LoaderClass is MockDataLoader or LoaderClass.__name__ in ('MaestroLoader', 'ARPESSweptLoader', 'ARPES1DLoader'):
                    continue
                raise
            except Exception as e:
                print(f"Warning: {LoaderClass.__name__} encountered an error: {e}")
                continue

        raise ValueError(f"Could not load '{filename}': No matching facility format found.")
