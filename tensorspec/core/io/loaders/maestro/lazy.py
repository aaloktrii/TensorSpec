from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral

import h5py
import numpy as np

from tensorspec.core.io.loaders.maestro.detect import (
    assert_maestro_signature,
    is_fixed_mode,
    select_spectra_dataset,
)
from tensorspec.core.io.loaders.maestro.low_level_scan import parse_low_level_scan
from tensorspec.core.io.loaders.maestro.registry import match_kind
from tensorspec.core.io.loaders.maestro.reshape import (
    detector_dims_for_buffer,
    recover_partial_grid,
    scan_cycles,
)
from tensorspec.core.io.loaders.maestro.types import ScanMotor, ScanPlan


@dataclass
class MaestroDescriptor:
    """Lazy Maestro metadata plus detector-frame hyperslab access.

    The descriptor owns one read-only HDF5 handle. Call :meth:`close` when
    finished, or use the descriptor as a context manager.
    """

    path: str
    kind: str
    labels: list[str]
    axes: list[np.ndarray]
    units: list[str]
    shape: tuple[int, ...]
    metadata: dict
    _file: h5py.File = field(repr=False)
    _dataset: h5py.Dataset = field(repr=False)
    _points_axis: int = field(repr=False)
    _n_energy: int = field(repr=False)
    _n_angle: int = field(repr=False)
    _readable_points: int = field(repr=False)
    _canonical_scan_shape: tuple[int, ...] = field(repr=False)
    _acquisition_scan_shape: tuple[int, ...] = field(repr=False)
    _acquisition_from_canonical: tuple[int, ...] = field(repr=False)

    def read_block(self, index: int | slice) -> np.ndarray:
        """Read one point or contiguous flat C-order span as ``(E, A)`` frames."""
        if not self._file.id.valid:
            raise ValueError("Maestro descriptor is closed.")

        canonical_selection = _normalize_selection(
            index, self._readable_points
        )
        if isinstance(canonical_selection, int):
            selection = self._to_acquisition_index(canonical_selection)
            return self._read_selection(selection)

        canonical_indices = range(
            canonical_selection.start, canonical_selection.stop
        )
        acquisition_indices = [
            self._to_acquisition_index(point)
            for point in canonical_indices
        ]
        if not acquisition_indices:
            return np.empty(
                (0, self._n_energy, self._n_angle),
                dtype=self._dataset.dtype,
            )
        first = acquisition_indices[0]
        if acquisition_indices == list(
            range(first, first + len(acquisition_indices))
        ):
            return self._read_selection(
                slice(first, first + len(acquisition_indices))
            )
        return np.stack(
            [self._read_selection(point) for point in acquisition_indices]
        )

    def _to_acquisition_index(self, canonical_index: int) -> int:
        canonical_coords = np.unravel_index(
            canonical_index, self._canonical_scan_shape
        )
        acquisition_coords = tuple(
            canonical_coords[canonical_axis]
            for canonical_axis in self._acquisition_from_canonical
        )
        return int(
            np.ravel_multi_index(
                acquisition_coords, self._acquisition_scan_shape
            )
        )

    def _read_selection(self, selection: int | slice) -> np.ndarray:
        if self._points_axis == 0:
            slab = self._dataset[selection, :, :]
        else:
            slab = self._dataset[:, :, selection]
        return detector_frame_from_slab(
            slab,
            self._n_energy,
            self._n_angle,
            self._points_axis,
        )

    def close(self) -> None:
        if self._file.id.valid:
            self._file.close()

    def __enter__(self) -> MaestroDescriptor:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def open_maestro(path: str) -> MaestroDescriptor:
    """Open supported Maestro scan metadata without loading its spectra buffer."""
    source_path = str(path)
    file = h5py.File(source_path, "r")
    try:
        assert_maestro_signature(file)
        headers = file.get("Headers")
        if headers is None or "Low_Level_Scan" not in headers:
            raise ValueError("Missing Headers/Low_Level_Scan.")

        fixed = is_fixed_mode(file)
        plan = parse_low_level_scan(headers["Low_Level_Scan"][()])
        module = match_kind(plan, fixed)
        if module is None:
            raise ValueError(
                f"No Maestro kind matched mode={plan.mode_name!r}, "
                f"is_fixed={fixed}."
            )

        dataset = select_spectra_dataset(file)
        dataset_shape = tuple(int(dim) for dim in dataset.shape)
        points_axis, actual, num_cycles = scan_cycles(
            file, dataset_shape, plan.expected_cycles
        )
        n_e, n_a, energy, angle, energy_unit, angle_unit = (
            detector_dims_for_buffer(
                dataset,
                dataset_shape,
                points_axis,
                is_fixed=fixed,
            )
        )

        labels, axes, units, canonical_motors = _kind_layout(
            module.KIND_ID,
            plan,
            energy,
            angle,
            energy_unit,
            angle_unit,
        )
        acquisition_motors = _acquisition_motors(
            module.KIND_ID, plan, canonical_motors
        )
        acquisition_from_canonical = _axis_permutation(
            acquisition_motors, canonical_motors
        )
        canonical_scan_shape = tuple(
            motor.n for motor in canonical_motors
        )
        acquisition_scan_shape = tuple(
            motor.n for motor in acquisition_motors
        )
        metadata: dict = {
            "kind": module.KIND_ID,
            "source_path": source_path,
            "scan_plan": {
                "mode_name": plan.mode_name,
                "expected_cycles": plan.expected_cycles,
                "actual_cycles": actual,
            },
        }
        if num_cycles is not None:
            metadata["scan_plan"]["num_cycles"] = num_cycles

        readable_points = actual
        if actual < plan.expected_cycles:
            partial = recover_partial_grid(actual, acquisition_scan_shape)
            if partial is not None:
                acquisition_scan_shape = partial.kept_shape
                canonical_shape = list(canonical_scan_shape)
                canonical_axis = acquisition_from_canonical[
                    partial.truncated_axis
                ]
                canonical_shape[canonical_axis] = partial.kept_shape[
                    partial.truncated_axis
                ]
                canonical_scan_shape = tuple(canonical_shape)
                axes[canonical_axis] = axes[canonical_axis][
                    : canonical_scan_shape[canonical_axis]
                ]
                readable_points = partial.kept_points
                metadata["partial_scan"] = {
                    "expected": plan.expected_cycles,
                    "actual": actual,
                    "kept_rows": partial.kept_rows,
                }
            else:
                labels = ["Point", "Energy", "Angle"]
                axes = [np.arange(actual), energy, angle]
                units = ["index", energy_unit, angle_unit]
                canonical_scan_shape = (actual,)
                acquisition_scan_shape = (actual,)
                acquisition_from_canonical = (0,)
                metadata["truncate_warning"] = (
                    f"Aborted scan: truncated from {plan.expected_cycles} "
                    f"to {actual} points."
                )

        return MaestroDescriptor(
            path=source_path,
            kind=module.KIND_ID,
            labels=labels,
            axes=axes,
            units=units,
            shape=(*canonical_scan_shape, n_e, n_a),
            metadata=metadata,
            _file=file,
            _dataset=dataset,
            _points_axis=points_axis,
            _n_energy=n_e,
            _n_angle=n_a,
            _readable_points=readable_points,
            _canonical_scan_shape=canonical_scan_shape,
            _acquisition_scan_shape=acquisition_scan_shape,
            _acquisition_from_canonical=acquisition_from_canonical,
        )
    except BaseException:
        file.close()
        raise


def detector_frame_from_slab(
    slab: np.ndarray,
    n_e: int,
    n_a: int,
    points_axis: int,
) -> np.ndarray:
    """Convert a fixed-spectra ``(angle, energy)`` slab to ``(..., E, A)``."""
    array = np.asarray(slab)
    if points_axis == 0:
        detector_shape = array.shape[-2:]
        if detector_shape != (n_a, n_e):
            raise ValueError(
                f"Detector slab has shape {detector_shape}, "
                f"expected ({n_a}, {n_e})."
            )
        result = np.swapaxes(array, -2, -1)
    elif points_axis == 2:
        detector_shape = array.shape[:2]
        if detector_shape != (n_a, n_e):
            raise ValueError(
                f"Detector slab has shape {detector_shape}, "
                f"expected ({n_a}, {n_e})."
            )
        if array.ndim == 2:
            result = array.T
        else:
            result = np.transpose(array, (2, 1, 0))
    else:
        raise ValueError(f"Unsupported points axis {points_axis}.")
    return np.ascontiguousarray(result)


def _normalize_selection(
    index: int | slice,
    n_points: int,
) -> int | slice:
    if isinstance(index, bool):
        raise TypeError("Point index must be an integer or slice.")
    if isinstance(index, Integral):
        normalized = int(index)
        if normalized < 0 or normalized >= n_points:
            raise IndexError(
                f"Point index {normalized} outside valid range "
                f"0..{n_points - 1}."
            )
        return normalized
    if not isinstance(index, slice):
        raise TypeError("Point index must be an integer or slice.")
    if index.step not in (None, 1):
        raise ValueError("read_block only accepts contiguous slices.")
    start, stop, _ = index.indices(n_points)
    return slice(start, stop)


def _kind_layout(
    kind: str,
    plan: ScanPlan,
    energy: np.ndarray,
    angle: np.ndarray,
    energy_unit: str,
    angle_unit: str,
) -> tuple[list[str], list[np.ndarray], list[str], list[ScanMotor]]:
    defl_motors = plan.angle_motors()
    if kind == "xy_fine_4d":
        x_motor, y_motor = _require_xy(plan, kind)
        scan_motors = [y_motor, x_motor]
        labels = ["Y", "X"]
    elif kind == "focus_xy_fine_5d":
        x_motor, y_motor = _require_xy(plan, kind)
        scan_motors = [y_motor, x_motor, defl_motors[0]]
        labels = ["Y", "X", defl_motors[0].name]
    elif kind == "defl_x_line_4d":
        x_motor = next(
            (
                motor
                for motor in plan.loops[0].motors
                if motor.name.casefold() in {"scan x", "sample x"}
            ),
            None,
        )
        if x_motor is None:
            raise ValueError(f"{kind}: scan plan has no Scan X motor.")
        scan_motors = [x_motor, defl_motors[0]]
        labels = ["X", defl_motors[0].name]
    elif kind == "fermi_defl_3d":
        scan_motors = [defl_motors[0]]
        labels = [defl_motors[0].name]
    else:
        raise ValueError(f"Lazy Maestro access does not support kind {kind!r}.")

    axes = [_motor_axis(motor) for motor in scan_motors]
    units = [motor.units for motor in scan_motors]
    return (
        labels + ["Energy", "Angle"],
        axes + [energy, angle],
        units + [energy_unit, angle_unit],
        scan_motors,
    )


def _acquisition_motors(
    kind: str,
    plan: ScanPlan,
    canonical_motors: list[ScanMotor],
) -> list[ScanMotor]:
    """Return acquisition (file) motor order matching each kind's eager reshape.

    Lazy ``read_block(i)`` must hit the same spectrum as
    ``eager.reshape(-1, ...)[i]``. That means acquisition order here must
    match how the eager loader reshapes the points axis before transpose
    to canonical axes.

    - ``xy_fine_4d``: canonical (Y, X); eager asserts that order.
    - ``focus_xy_fine_5d``: ``plan.motors_slow_to_fast()`` (Y, X, Defl).
    - ``defl_x_line_4d`` (and default): flattened loop-motor list, same as
      the eager loader's ``motors_in_order``.
    """
    if kind == "xy_fine_4d":
        # XY mesh points are stored with X varying fastest.
        return canonical_motors
    if kind == "focus_xy_fine_5d":
        return list(plan.motors_slow_to_fast())
    return [
        motor
        for loop in plan.loops
        for motor in loop.motors
    ]


def _axis_permutation(
    acquisition_motors: list[ScanMotor],
    canonical_motors: list[ScanMotor],
) -> tuple[int, ...]:
    if len(acquisition_motors) != len(canonical_motors):
        raise ValueError("Acquisition and canonical scan ranks differ.")
    try:
        return tuple(
            canonical_motors.index(motor)
            for motor in acquisition_motors
        )
    except ValueError as exc:
        raise ValueError(
            "Acquisition motors do not match canonical kind motors."
        ) from exc


def _require_xy(plan: ScanPlan, kind: str) -> tuple[ScanMotor, ScanMotor]:
    motors = plan.xy_motors()
    if motors is None:
        raise ValueError(f"{kind}: scan plan has no XY mesh.")
    return motors


def _motor_axis(motor: ScanMotor) -> np.ndarray:
    return np.linspace(motor.start, motor.end, motor.n)


__all__ = ["MaestroDescriptor", "open_maestro"]
