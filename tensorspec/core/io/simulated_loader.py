import numpy as np
from pathlib import Path
from tensorspec.core.data_models import TensorData
from tensorspec.core.arpes.photon_energy_scan import tensor_from_stacked_sim


def inplane_axes(data, metadata, n0: int, n1: int):
    """Axes the file already stores. No angle-to-k conversion.

    ``n0``, ``n1`` are the in-plane sizes of ``intensity``. A momentum file has
    ``kx``/``ky`` (1/A). An angle file has ``theta``/``phi`` (deg). If both are
    present, ``stack_axes`` / ``axes_are_angles`` wins; otherwise the pair whose
    lengths match the intensity.
    """
    meta = metadata or {}
    files = set(data.files)
    has_k = {"kx", "ky"} <= files
    has_a = {"theta", "phi"} <= files
    kx = ky = th = ph = None
    k_fit = a_fit = False
    if has_k:
        kx, ky = np.asarray(data["kx"]), np.asarray(data["ky"])
        k_fit = kx.size == n0 and ky.size == n1
    if has_a:
        th, ph = np.asarray(data["theta"]), np.asarray(data["phi"])
        a_fit = th.size == n0 and ph.size == n1
    asked = str(meta.get("stack_axes") or "").lower()
    want_angle = asked == "angle" or bool(meta.get("axes_are_angles"))
    if asked == "k" and k_fit:
        return kx, ky, "k"
    if want_angle and a_fit:
        return th, ph, "angle"
    if k_fit and not a_fit:
        return kx, ky, "k"
    if a_fit and not k_fit:
        return th, ph, "angle"
    if k_fit:
        return kx, ky, "k"
    if a_fit:
        return th, ph, "angle"
    raise KeyError(
        "simulated ARPES file needs kx/ky or theta/phi whose lengths match the intensity"
    )


def kz_along_hv(data, metadata, n_hv: int):
    """kz already stored for each photon energy. No conversion.

    Angle stacks keep ``kz_normal``. Momentum stacks keep the same numbers in
    ``metadata['kz_normal_emission_1perA']``. Length must match ``hv``.
    """
    if "kz_normal" in getattr(data, "files", ()):
        kz = np.asarray(data["kz_normal"], float).ravel()
        if kz.size == n_hv:
            return kz
    raw = (metadata or {}).get("kz_normal_emission_1perA")
    if raw is None:
        return None
    kz = np.asarray(raw, float).ravel()
    if kz.size == n_hv:
        return kz
    return None


class SimulatedARPESLoader:
    """Loader for TensorSpec's native .npz simulated ARPES datasets.

    Contract: intensity already in momentum (``kx``/``ky``, 1/A) or already in angle
    (``theta``/``phi``, deg). A photon-energy stack also carries ``hv``. Nothing is converted.
    """

    @staticmethod
    def load(filepath: str) -> TensorData:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Cannot find simulated data at {filepath}")

        with np.load(path, allow_pickle=True) as data:
            intensity = data['intensity']  # (axis0, axis1, E), or (hv, axis0, axis1, E)
            metadata = data['metadata'].item() if 'metadata' in data else {}

            if "hv" in data:
                if intensity.ndim != 4:
                    raise ValueError(
                        f"photon-energy scan intensity must be (hv, axis0, axis1, E), got {intensity.shape}"
                    )
                ax0, ax1, space = inplane_axes(data, metadata, intensity.shape[1], intensity.shape[2])
                metadata = {**metadata, "stack_axes": space}
                kz = kz_along_hv(data, metadata, intensity.shape[0])
                if kz is not None:
                    metadata = {
                        **metadata,
                        "axis_alternates": {
                            "kz (1/A)": {
                                "dim": 0,
                                "values": kz,
                                "label": "kz",
                                "unit": "1/A",
                            }
                        },
                    }
                return tensor_from_stacked_sim(
                    intensity,
                    data["hv"],
                    ax0,
                    ax1,
                    data["E"],
                    metadata=metadata,
                )

            if intensity.ndim != 3:
                raise ValueError(
                    f"intensity must be (axis0, axis1, E), got {intensity.shape}"
                )
            ax0, ax1, space = inplane_axes(data, metadata, intensity.shape[0], intensity.shape[1])
            if space == "angle":
                labels = ["Energy", "theta (Slit angle)", "phi (Deflection angle)"]
                units = ["eV", "deg", "deg"]
                data_type = "Simulated ARPES Matrix Elements (angle space)"
            else:
                labels = ["Energy", "kx (Slit)", "ky (Deflection)"]
                units = ["eV", "1/A", "1/A"]
                data_type = "Simulated ARPES Matrix Elements"

            # Transpose (axis0, axis1, E) -> (E, axis0, axis1) to match the Data Viewer's expected axis order
            value_matrix = np.transpose(intensity, (2, 0, 1))

            return TensorData(
                value=value_matrix,
                axes=[data['E'], ax0, ax1],
                labels=labels,
                units=units,
                data_type=data_type,
                metadata=metadata
            )
