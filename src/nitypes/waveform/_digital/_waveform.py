from __future__ import annotations

import datetime as dt
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, SupportsIndex, overload

import hightime as ht
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from nitypes._arguments import arg_to_uint, validate_dtype, validate_unsupported_arg
from nitypes._exceptions import invalid_arg_type, invalid_array_ndim
from nitypes._numpy import asarray as _np_asarray
from nitypes.waveform._digital._port import bit_mask, get_port_dtype, port_to_line_data
from nitypes.waveform._digital._signal_collection import DigitalWaveformSignalCollection
from nitypes.waveform._digital._state import DigitalState
from nitypes.waveform._digital._types import (
    _DIGITAL_PORT_DTYPES,
    _DIGITAL_STATE_DTYPES,
    _TOtherState,
    _TState,
)
from nitypes.waveform._exceptions import (
    capacity_mismatch,
    capacity_too_small,
    data_type_mismatch,
    irregular_timestamp_count_mismatch,
    signal_count_mismatch,
    start_index_or_sample_count_too_large,
    start_index_too_large,
)
from nitypes.waveform._extended_properties import (
    CHANNEL_NAME,
    LINE_NAMES,
    ExtendedPropertyDictionary,
    ExtendedPropertyValue,
)
from nitypes.waveform._timing import Timing, _AnyDateTime, _AnyTimeDelta

if sys.version_info < (3, 10):
    import array as std_array


@dataclass(frozen=True)
class DigitalWaveformFailure:
    """A test failure, indicating where the actual waveform did not match the expected waveform."""

    sample_index: int
    """The sample index into the compared waveform where the test failure occurred."""

    expected_sample_index: int
    """The sample index into the expected waveform where the test failure occurred."""

    signal_index: int
    """The signal index where the test failure occurred."""

    actual_state: DigitalState
    """The state from the compared waveform where the test failure occurred."""

    expected_state: DigitalState
    """The state from the expected waveform where the test failure occurred."""


@dataclass(frozen=True)
class DigitalWaveformTestResult:
    """A test result from comparing a digital waveform against an expected digital waveform."""

    @property
    def success(self) -> bool:
        """True if the test is successful, False if the test failed."""
        return len(self.failures) == 0

    failures: Sequence[DigitalWaveformFailure]
    """A collection of test failure information."""


class DigitalWaveform(Generic[_TState]):
    """A digital waveform, which encapsulates digital data and timing information."""

    @overload
    @classmethod
    def from_lines(
        cls,
        array: npt.NDArray[_TOtherState],
        dtype: None = ...,
        *,
        copy: bool = ...,
        start_index: SupportsIndex | None = ...,
        sample_count: SupportsIndex | None = ...,
        signal_count: SupportsIndex | None = ...,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = ...,
        timing: Timing[_AnyDateTime, _AnyTimeDelta, _AnyTimeDelta] | None = ...,
    ) -> DigitalWaveform[_TOtherState]: ...

    @overload
    @classmethod
    def from_lines(
        cls,
        array: npt.NDArray[Any] | Sequence[Any],
        dtype: type[_TOtherState] | np.dtype[_TOtherState],
        *,
        copy: bool = ...,
        start_index: SupportsIndex | None = ...,
        sample_count: SupportsIndex | None = ...,
        signal_count: SupportsIndex | None = ...,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = ...,
        timing: Timing[_AnyDateTime, _AnyTimeDelta, _AnyTimeDelta] | None = ...,
    ) -> DigitalWaveform[_TOtherState]: ...

    @overload
    @classmethod
    def from_lines(
        cls,
        array: npt.NDArray[Any] | Sequence[Any],
        dtype: npt.DTypeLike = ...,
        *,
        copy: bool = ...,
        start_index: SupportsIndex | None = ...,
        sample_count: SupportsIndex | None = ...,
        signal_count: SupportsIndex | None = ...,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = ...,
        timing: Timing[_AnyDateTime, _AnyTimeDelta, _AnyTimeDelta] | None = ...,
    ) -> DigitalWaveform[Any]: ...

    @classmethod
    def from_lines(
        cls,
        array: npt.NDArray[Any] | Sequence[Any],
        dtype: npt.DTypeLike = None,
        *,
        copy: bool = True,
        start_index: SupportsIndex | None = 0,
        sample_count: SupportsIndex | None = None,
        signal_count: SupportsIndex | None = None,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = None,
        timing: Timing[_AnyDateTime, _AnyTimeDelta, _AnyTimeDelta] | None = None,
    ) -> DigitalWaveform[Any]:
        """Construct a waveform from a one or two-dimensional array or sequence of line data.

        Each element of the line data array represents a digital state, such as 1 for "on" or 0
        for "off". The line data should be in a 1D array indexed by sample or a 2D array indexed
        by (sample, signal). The line data may also use digital state values from the
        :any:`DigitalState` enum.

        Args:
            array: The line data as a one or two-dimensional array or a sequence.
            dtype: The NumPy data type for the waveform data.
            copy: Specifies whether to copy the array or save a reference to it.
            start_index: The sample index at which the waveform data begins.
            sample_count: The number of samples in the waveform.
            signal_count: The number of signals in the waveform.
            extended_properties: The extended properties of the waveform.
            timing: The timing information of the waveform.

        Returns:
            A waveform containing the specified data.
        """
        if isinstance(array, np.ndarray):
            if array.ndim not in (1, 2):
                raise invalid_array_ndim(
                    "input array", "one or two-dimensional array or sequence", array.ndim
                )
            if dtype is not None and array.dtype != dtype:
                raise data_type_mismatch("input array", array.dtype, "requested", dtype)
        elif isinstance(array, Sequence) or (
            sys.version_info < (3, 10) and isinstance(array, std_array.array)
        ):
            if dtype is None:
                dtype = np.uint8
        else:
            raise invalid_arg_type("input array", "one or two-dimensional array or sequence", array)

        return cls(
            data=_np_asarray(array, dtype, copy=copy),
            start_index=start_index,
            sample_count=sample_count,
            signal_count=signal_count,
            extended_properties=extended_properties,
            timing=timing,
        )

    @overload
    @classmethod
    def from_port(
        cls,
        array: npt.NDArray[Any] | Sequence[Any],
        mask: SupportsIndex | None = ...,
        dtype: None = ...,
        *,
        start_index: SupportsIndex | None = ...,
        sample_count: SupportsIndex | None = ...,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = ...,
        timing: Timing[_AnyDateTime, _AnyTimeDelta, _AnyTimeDelta] | None = ...,
    ) -> DigitalWaveform[np.uint8]: ...

    @overload
    @classmethod
    def from_port(
        cls,
        array: npt.NDArray[Any] | Sequence[Any],
        mask: SupportsIndex | None = ...,
        dtype: (
            type[_TOtherState] | np.dtype[_TOtherState]  # pyright: ignore[reportInvalidTypeVarUse]
        ) = ...,
        *,
        start_index: SupportsIndex | None = ...,
        sample_count: SupportsIndex | None = ...,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = ...,
        timing: Timing[_AnyDateTime, _AnyTimeDelta, _AnyTimeDelta] | None = ...,
    ) -> DigitalWaveform[_TOtherState]: ...

    @overload
    @classmethod
    def from_port(
        cls,
        array: npt.NDArray[Any] | Sequence[Any],
        mask: SupportsIndex | None = ...,
        dtype: npt.DTypeLike = ...,
        *,
        start_index: SupportsIndex | None = ...,
        sample_count: SupportsIndex | None = ...,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = ...,
        timing: Timing[_AnyDateTime, _AnyTimeDelta, _AnyTimeDelta] | None = ...,
    ) -> DigitalWaveform[Any]: ...

    @classmethod
    def from_port(
        cls,
        array: npt.NDArray[Any] | Sequence[Any],
        mask: SupportsIndex | None = None,
        dtype: npt.DTypeLike = None,
        *,
        start_index: SupportsIndex | None = 0,
        sample_count: SupportsIndex | None = None,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = None,
        timing: Timing[_AnyDateTime, _AnyTimeDelta, _AnyTimeDelta] | None = None,
    ) -> DigitalWaveform[Any]:
        """Construct a waveform from a one-dimensional array or sequence of port data.

        This method allocates a new array in order to convert the port data to line data.

        Each element of the port data array represents a digital sample taken over a port of
        signals. Each bit in the sample represents a digital state, either 1 for "on" or 0 for
        "off". The least significant bit of the sample is placed at signal index 0 of the
        DigitalWaveform.

        If the input array is not a NumPy array, you must specify the mask.

        Args:
            array: The port data as a one-dimensional array or a sequence.
            mask: A bitmask specifying which lines to include in the waveform.
            dtype: The NumPy data type for the waveform (line) data.
            start_index: The sample index at which the waveform data begins.
            sample_count: The number of samples in the waveform.
            extended_properties: The extended properties of the waveform.
            timing: The timing information of the waveform.

        Returns:
            A waveform containing the specified data.
        """
        if isinstance(array, np.ndarray):
            if array.ndim != 1:
                raise invalid_array_ndim(
                    "input array", "one-dimensional array or sequence", array.ndim
                )
            validate_dtype(array.dtype, _DIGITAL_PORT_DTYPES)
            default_mask = bit_mask(array.dtype.itemsize * 8)
        elif isinstance(array, Sequence) or (
            sys.version_info < (3, 10) and isinstance(array, std_array.array)
        ):
            # np.array([0,1]).dtype is np.int64 by default, so the user must specify the port size.
            if mask is None:
                raise ValueError(
                    "You must specify a mask when the input array is not a NumPy array."
                )
            default_mask = 0
        else:
            raise invalid_arg_type("input array", "one or two-dimensional array or sequence", array)

        if dtype is None:
            dtype = np.uint8
        validate_dtype(dtype, _DIGITAL_STATE_DTYPES)

        mask = arg_to_uint("mask", mask, default_mask)

        if isinstance(array, np.ndarray):
            port_dtype = array.dtype
        else:
            port_dtype = get_port_dtype(mask)

        port_data = _np_asarray(array, port_dtype)
        line_data = port_to_line_data(port_data, mask)
        if line_data.dtype != dtype:
            line_data = line_data.view(dtype)

        return cls(
            data=line_data,
            dtype=dtype,
            start_index=start_index,
            sample_count=sample_count,
            extended_properties=extended_properties,
            timing=timing,
        )

    @overload
    @classmethod
    def from_ports(
        cls,
        array: npt.NDArray[Any] | Sequence[Any],
        masks: Sequence[SupportsIndex] | None = ...,
        dtype: None = ...,
        *,
        start_index: SupportsIndex | None = ...,
        sample_count: SupportsIndex | None = ...,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = ...,
        timing: Timing[_AnyDateTime, _AnyTimeDelta, _AnyTimeDelta] | None = ...,
    ) -> Sequence[DigitalWaveform[np.uint8]]: ...

    @overload
    @classmethod
    def from_ports(
        cls,
        array: npt.NDArray[Any] | Sequence[Any],
        masks: Sequence[SupportsIndex] | None = ...,
        dtype: (
            type[_TOtherState] | np.dtype[_TOtherState]  # pyright: ignore[reportInvalidTypeVarUse]
        ) = ...,
        *,
        start_index: SupportsIndex | None = ...,
        sample_count: SupportsIndex | None = ...,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = ...,
        timing: Timing[_AnyDateTime, _AnyTimeDelta, _AnyTimeDelta] | None = ...,
    ) -> Sequence[DigitalWaveform[_TOtherState]]: ...

    @overload
    @classmethod
    def from_ports(
        cls,
        array: npt.NDArray[Any] | Sequence[Any],
        masks: Sequence[SupportsIndex] | None = ...,
        dtype: npt.DTypeLike = ...,
        *,
        start_index: SupportsIndex | None = ...,
        sample_count: SupportsIndex | None = ...,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = ...,
        timing: Timing[_AnyDateTime, _AnyTimeDelta, _AnyTimeDelta] | None = ...,
    ) -> Sequence[DigitalWaveform[Any]]: ...

    @classmethod
    def from_ports(
        cls,
        array: npt.NDArray[Any] | Sequence[Any],
        masks: Sequence[SupportsIndex] | None = None,
        dtype: npt.DTypeLike = None,
        *,
        start_index: SupportsIndex | None = 0,
        sample_count: SupportsIndex | None = None,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = None,
        timing: Timing[_AnyDateTime, _AnyTimeDelta, _AnyTimeDelta] | None = None,
    ) -> Sequence[DigitalWaveform[Any]]:
        """Construct a waveform from a two-dimensional array or sequence of port data.

        This method allocates a new array in order to convert the port data to line data.

        Each row of the port data array corresponds to a resulting DigitalWaveform. Each element of
        the port data array represents a digital sample taken over a port of signals. Each bit in
        the sample is represents a digital state, either 1 for "on" or 0 for "off". The least
        significant bit of the sample is placed at signal index 0 of the corresponding
        DigitalWaveform.

        If the input array is not a NumPy array, you must specify the masks.

        Args:
            array: The port data as a two-dimensional array or a sequence.
            masks: A sequence of bitmasks specifying which lines from each port to include in the
                corresponding waveform.
            dtype: The NumPy data type for the waveform (line) data.
            start_index: The sample index at which the waveform data begins.
            sample_count: The number of samples in the waveform.
            extended_properties: The extended properties of the waveform.
            timing: The timing information of the waveform.

        Returns:
            A waveform containing the specified data.
        """
        if isinstance(array, np.ndarray):
            if array.ndim != 2:
                raise invalid_array_ndim(
                    "input array", "two-dimensional array or sequence", array.ndim
                )
            validate_dtype(array.dtype, _DIGITAL_PORT_DTYPES)
            default_masks = [bit_mask(array.dtype.itemsize * 8)] * array.shape[0]
        elif isinstance(array, Sequence) or (
            sys.version_info < (3, 10) and isinstance(array, std_array.array)
        ):
            # np.array([0,1]).dtype is np.int64 by default, so the user must specify the port size.
            if masks is None:
                raise ValueError(
                    "You must specify the masks when the input array is not a NumPy array."
                )
            default_masks = []
        else:
            raise invalid_arg_type("input array", "one or two-dimensional array or sequence", array)

        if dtype is None:
            dtype = np.uint8
        validate_dtype(dtype, _DIGITAL_STATE_DTYPES)

        if not isinstance(masks, (type(None), Sequence)):
            raise invalid_arg_type("masks", "sequence of ints")
        if masks is not None:
            validated_masks = [arg_to_uint("mask", mask) for mask in masks]
        else:
            validated_masks = default_masks

        if isinstance(array, np.ndarray):
            port_dtype = array.dtype
        else:
            port_dtype = get_port_dtype(validated_masks)

        port_data = _np_asarray(array, port_dtype)
        waveforms = []
        for port_index in range(port_data.shape[0]):
            line_data = port_to_line_data(port_data[port_index, :], validated_masks[port_index])
            if line_data.dtype != dtype:
                line_data = line_data.view(dtype)

            waveforms.append(
                cls(
                    data=line_data,
                    dtype=dtype,
                    start_index=start_index,
                    sample_count=sample_count,
                    extended_properties=extended_properties,
                    timing=timing,
                )
            )
        return waveforms

    __slots__ = [
        "_data",
        "_data_1d",
        "_start_index",
        "_sample_count",
        "_extended_properties",
        "_timing",
        "_signals",
        "_signal_names",
        "__weakref__",
    ]

    _data: npt.NDArray[_TState]
    _data_1d: npt.NDArray[_TState] | None
    _start_index: int
    _sample_count: int
    _extended_properties: ExtendedPropertyDictionary
    _timing: Timing[_AnyDateTime, _AnyTimeDelta, _AnyTimeDelta]
    _signals: DigitalWaveformSignalCollection[_TState] | None
    _signal_names: list[str] | None

    # If neither dtype nor data is specified, _TData defaults to np.uint8.
    @overload
    def __init__(  # noqa: D107 - Missing docstring in __init__ (auto-generated noqa)
        self: DigitalWaveform[np.uint8],
        sample_count: SupportsIndex | None = ...,
        signal_count: SupportsIndex | None = ...,
        dtype: None = ...,
        default_value: bool | int | DigitalState | None = ...,
        *,
        data: None = ...,
        start_index: SupportsIndex | None = ...,
        capacity: SupportsIndex | None = ...,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = ...,
        copy_extended_properties: bool = ...,
        timing: Timing[_AnyDateTime, _AnyTimeDelta, _AnyTimeDelta] | None = ...,
    ) -> None: ...

    @overload
    def __init__(  # noqa: D107 - Missing docstring in __init__ (auto-generated noqa)
        self: DigitalWaveform[_TOtherState],
        sample_count: SupportsIndex | None = ...,
        signal_count: SupportsIndex | None = ...,
        dtype: type[_TOtherState] | np.dtype[_TOtherState] = ...,
        default_value: bool | int | DigitalState | None = ...,
        *,
        data: None = ...,
        start_index: SupportsIndex | None = ...,
        capacity: SupportsIndex | None = ...,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = ...,
        copy_extended_properties: bool = ...,
        timing: Timing[_AnyDateTime, _AnyTimeDelta, _AnyTimeDelta] | None = ...,
    ) -> None: ...

    @overload
    def __init__(  # noqa: D107 - Missing docstring in __init__ (auto-generated noqa)
        self: DigitalWaveform[_TOtherState],
        sample_count: SupportsIndex | None = ...,
        signal_count: SupportsIndex | None = ...,
        dtype: None = ...,
        default_value: bool | int | DigitalState | None = ...,
        *,
        data: npt.NDArray[_TOtherState] = ...,
        start_index: SupportsIndex | None = ...,
        capacity: SupportsIndex | None = ...,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = ...,
        copy_extended_properties: bool = ...,
        timing: Timing[_AnyDateTime, _AnyTimeDelta, _AnyTimeDelta] | None = ...,
    ) -> None: ...

    @overload
    def __init__(  # noqa: D107 - Missing docstring in __init__ (auto-generated noqa)
        self: DigitalWaveform[Any],
        sample_count: SupportsIndex | None = ...,
        signal_count: SupportsIndex | None = ...,
        dtype: npt.DTypeLike = ...,
        default_value: bool | int | DigitalState | None = ...,
        *,
        data: npt.NDArray[Any] | None = ...,
        start_index: SupportsIndex | None = ...,
        capacity: SupportsIndex | None = ...,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = ...,
        copy_extended_properties: bool = ...,
        timing: Timing[_AnyDateTime, _AnyTimeDelta, _AnyTimeDelta] | None = ...,
    ) -> None: ...

    def __init__(
        self,
        sample_count: SupportsIndex | None = None,
        signal_count: SupportsIndex | None = None,
        dtype: npt.DTypeLike = None,
        default_value: bool | int | DigitalState | None = None,
        *,
        data: npt.NDArray[Any] | None = None,
        start_index: SupportsIndex | None = None,
        capacity: SupportsIndex | None = None,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = None,
        copy_extended_properties: bool = True,
        timing: Timing[_AnyDateTime, _AnyTimeDelta, _AnyTimeDelta] | None = None,
    ) -> None:
        """Initialize a new digital waveform.

        Args:
            sample_count: The number of samples in the waveform.
            signal_count: The number of signals in the waveform.
            dtype: The NumPy data type for the waveform data.
            default_value: The :any:`DigitalState` to initialize the waveform with.
            data: A NumPy ndarray to use for sample storage. The waveform takes ownership
                of this array. If not specified, an ndarray is created based on the specified dtype,
                start index, sample count, and capacity.
            start_index: The sample index at which the waveform data begins.
            sample_count: The number of samples in the waveform.
            capacity: The number of samples to allocate. Pre-allocating a larger buffer optimizes
                appending samples to the waveform.
            extended_properties: The extended properties of the waveform.
            copy_extended_properties: Specifies whether to copy the extended properties or take
                ownership.
            timing: The timing information of the waveform.

        Returns:
            A digital waveform.
        """
        if data is None:
            self._init_with_new_array(
                sample_count,
                signal_count,
                dtype,
                default_value,
                start_index=start_index,
                capacity=capacity,
            )
        elif isinstance(data, np.ndarray):
            self._init_with_provided_array(
                data,
                dtype,
                start_index=start_index,
                sample_count=sample_count,
                signal_count=signal_count,
                capacity=capacity,
            )
        else:
            raise invalid_arg_type("raw data", "NumPy ndarray", data)

        if copy_extended_properties or not isinstance(
            extended_properties, ExtendedPropertyDictionary
        ):
            extended_properties = ExtendedPropertyDictionary(extended_properties)
        self._extended_properties = extended_properties

        if timing is None:
            timing = Timing.empty
        self._timing = timing

        self._signals = None
        self._signal_names = None

    def _init_with_new_array(
        self,
        sample_count: SupportsIndex | None = None,
        signal_count: SupportsIndex | None = None,
        dtype: npt.DTypeLike = None,
        default_value: bool | int | DigitalState | None = None,
        *,
        start_index: SupportsIndex | None = None,
        capacity: SupportsIndex | None = None,
    ) -> None:
        start_index = arg_to_uint("start index", start_index, 0)
        sample_count = arg_to_uint("sample count", sample_count, 0)
        signal_count = arg_to_uint("signal count", signal_count, 1)
        capacity = arg_to_uint("capacity", capacity, sample_count)

        if dtype is None:
            dtype = np.uint8
        validate_dtype(dtype, _DIGITAL_STATE_DTYPES)

        if start_index > capacity:
            raise start_index_too_large(start_index, "capacity", capacity)
        if start_index + sample_count > capacity:
            raise start_index_or_sample_count_too_large(
                start_index, sample_count, "capacity", capacity
            )

        if default_value is None:
            default_value = 0
        elif not isinstance(default_value, (bool, int, DigitalState)):
            raise invalid_arg_type("default value", "bool, int, or DigitalState", default_value)

        self._data = np.full((capacity, signal_count), default_value, dtype)
        self._data_1d = None
        self._start_index = start_index
        self._sample_count = sample_count

    def _init_with_provided_array(
        self,
        data: npt.NDArray[_TState],
        dtype: npt.DTypeLike = None,
        *,
        start_index: SupportsIndex | None = None,
        sample_count: SupportsIndex | None = None,
        signal_count: SupportsIndex | None = None,
        capacity: SupportsIndex | None = None,
    ) -> None:
        if not isinstance(data, np.ndarray):
            raise invalid_arg_type("input array", "one or two-dimensional array", data)

        if dtype is None:
            dtype = data.dtype
        if dtype != data.dtype:
            raise data_type_mismatch("input array", data.dtype, "requested", np.dtype(dtype))
        validate_dtype(dtype, _DIGITAL_STATE_DTYPES)

        if data.ndim == 1:
            data_signal_count = 1
            data_1d = data
            data = data.reshape(len(data), 1)
        elif data.ndim == 2:
            data_signal_count = data.shape[1]
            data_1d = None
        else:
            raise invalid_array_ndim("input array", "one or two-dimensional array", data.ndim)

        capacity = arg_to_uint("capacity", capacity, len(data))
        if capacity != len(data):
            raise capacity_mismatch(capacity, len(data))

        start_index = arg_to_uint("start index", start_index, 0)
        if start_index > capacity:
            raise start_index_too_large(start_index, "input array length", capacity)

        sample_count = arg_to_uint("sample count", sample_count, len(data) - start_index)
        if start_index + sample_count > len(data):
            raise start_index_or_sample_count_too_large(
                start_index, sample_count, "input array length", len(data)
            )

        signal_count = arg_to_uint("signal count", signal_count, data_signal_count)
        if signal_count != data_signal_count:
            raise signal_count_mismatch("provided", signal_count, "array", data_signal_count)

        self._data = data
        self._data_1d = data_1d
        self._start_index = start_index
        self._sample_count = sample_count

    @property
    def signals(self) -> DigitalWaveformSignalCollection[_TState]:
        """A collection of objects representing waveform signals."""
        # Lazily allocate self._signals if the application needs it.
        #
        # https://github.com/ni/nitypes-python/issues/131 - DigitalWaveform.signals introduces a
        # reference cycle, which affects GC overhead.
        value = self._signals
        if value is None:
            value = self._signals = DigitalWaveformSignalCollection(self)
        return value

    @property
    def data(self) -> npt.NDArray[_TState]:
        """The waveform data, indexed by (sample, signal)."""
        return self._data[self._start_index : self._start_index + self._sample_count]

    def get_data(
        self, start_index: SupportsIndex | None = 0, sample_count: SupportsIndex | None = None
    ) -> npt.NDArray[_TState]:
        """Get a subset of the waveform data.

        Args:
            start_index: The sample index at which the data begins.
            sample_count: The number of samples to return.

        Returns:
            A subset of the raw waveform data.
        """
        start_index = arg_to_uint("start index", start_index, 0)
        if start_index > self.sample_count:
            raise start_index_too_large(
                start_index, "number of samples in the waveform", self.sample_count
            )

        sample_count = arg_to_uint("sample count", sample_count, self.sample_count - start_index)
        if start_index + sample_count > self.sample_count:
            raise start_index_or_sample_count_too_large(
                start_index, sample_count, "number of samples in the waveform", self.sample_count
            )

        return self.data[start_index : start_index + sample_count]

    @property
    def sample_count(self) -> int:
        """The number of samples in the waveform."""
        return self._sample_count

    @property
    def signal_count(self) -> int:
        """The number of signals in the waveform."""
        # npt.NDArray[_ScalarT] currently has a shape type of _AnyShape, which is tuple[Any, ...]
        shape: tuple[int, ...] = self._data.shape
        return shape[1]

    @property
    def capacity(self) -> int:
        """The total capacity available for waveform data.

        Setting the capacity resizes the underlying NumPy array in-place.

        * Other Python objects with references to the array will see the array size change.
        * If the array has a reference to an external buffer (such as an array.array), attempting
          to resize it raises ValueError.
        """
        return len(self._data)

    @capacity.setter
    def capacity(self, value: int) -> None:
        value = arg_to_uint("capacity", value)
        min_capacity = self._start_index + self._sample_count
        if value < min_capacity:
            raise capacity_too_small(value, min_capacity, "waveform")
        if value != len(self._data):
            if self._data_1d is not None:
                # If _data is a 2D view of a 1D array, resize the base array and recreate the view.
                self._data_1d.resize(value, refcheck=False)
                self._data = self._data_1d.reshape(len(self._data_1d), 1)
            else:
                self._data.resize((value, self.signal_count), refcheck=False)

    @property
    def dtype(self) -> np.dtype[_TState]:
        """The NumPy dtype for the waveform data."""
        return self._data.dtype

    @property
    def extended_properties(self) -> ExtendedPropertyDictionary:
        """The extended properties for the waveform."""
        return self._extended_properties

    @property
    def channel_name(self) -> str:
        """The name of the device channel from which the waveform was acquired."""
        value = self._extended_properties.get(CHANNEL_NAME, "")
        assert isinstance(value, str)
        return value

    @channel_name.setter
    def channel_name(self, value: str) -> None:
        if not isinstance(value, str):
            raise invalid_arg_type("channel name", "str", value)
        self._extended_properties[CHANNEL_NAME] = value

    def _get_signal_names(self) -> list[str]:
        # Lazily allocate self._signal_names if the application needs it.
        signal_names = self._signal_names
        if signal_names is None:
            signal_names_str = self._extended_properties.get(LINE_NAMES, "")
            assert isinstance(signal_names_str, str)
            signal_names = self._signal_names = [
                name.strip() for name in signal_names_str.split(",")
            ]
            if len(signal_names) < self.signal_count:
                signal_names.extend([""] * (self.signal_count - len(signal_names)))
        return signal_names

    def _get_signal_name(self, signal_index: int) -> str:
        return self._get_signal_names()[signal_index]

    def _set_signal_name(self, signal_index: int, value: str) -> None:
        signal_names = self._get_signal_names()
        signal_names[signal_index] = value
        self._extended_properties[LINE_NAMES] = ", ".join(signal_names)

    def _set_timing(self, value: Timing[_AnyDateTime, _AnyTimeDelta, _AnyTimeDelta]) -> None:
        if self._timing is not value:
            self._timing = value

    def _validate_timing(self, value: Timing[_AnyDateTime, _AnyTimeDelta, _AnyTimeDelta]) -> None:
        if value._timestamps is not None and len(value._timestamps) != self._sample_count:
            raise irregular_timestamp_count_mismatch(
                len(value._timestamps), "number of samples in the waveform", self._sample_count
            )

    @property
    def timing(self) -> Timing[_AnyDateTime, _AnyTimeDelta, _AnyTimeDelta]:
        """The timing information of the waveform.

        The default value is Timing.empty.
        """
        return self._timing

    @timing.setter
    def timing(self, value: Timing[_AnyDateTime, _AnyTimeDelta, _AnyTimeDelta]) -> None:
        if not isinstance(value, Timing):
            raise invalid_arg_type("timing information", "Timing object", value)
        self._validate_timing(value)
        self._set_timing(value)

    def append(
        self,
        other: npt.NDArray[_TState] | DigitalWaveform[_TState] | Sequence[DigitalWaveform[_TState]],
        /,
        timestamps: Sequence[dt.datetime] | Sequence[ht.datetime] | None = None,
    ) -> None:
        """Append data to the waveform.

        Args:
            other: The array or waveform(s) to append.
            timestamps: A sequence of timestamps. When the current waveform has
                SampleIntervalMode.IRREGULAR, you must provide a sequence of timestamps with the
                same length as the array.

        Raises:
            TimingMismatchError: The current and other waveforms have incompatible timing.
            TimingMismatchWarning: The sample intervals of the waveform(s) do not match.
            ValueError: The other array has the wrong number of dimensions or the length of the
                timestamps argument does not match the length of the other array.
            TypeError: The data types of the current waveform and other array or waveform(s) do not
                match, or an argument has the wrong data type.

        When appending waveforms:

        * Timing information is merged based on the sample interval mode of the current
          waveform:

          * SampleIntervalMode.NONE or SampleIntervalMode.REGULAR: The other waveform(s) must also
            have SampleIntervalMode.NONE or SampleIntervalMode.REGULAR. If the sample interval does
            not match, a TimingMismatchWarning is generated. Otherwise, the timing information of
            the other waveform(s) is discarded.

          * SampleIntervalMode.IRREGULAR: The other waveforms(s) must also have
            SampleIntervalMode.IRREGULAR. The timestamps of the other waveforms(s) are appended to
            the current waveform's timing information.

        * Extended properties of the other waveform(s) are merged into the current waveform if they
          are not already set in the current waveform.
        """
        if isinstance(other, np.ndarray):
            self._append_array(other, timestamps)
        elif isinstance(other, DigitalWaveform):
            validate_unsupported_arg("timestamps", timestamps)
            self._append_waveform(other)  # type: ignore[arg-type]  # https://github.com/python/mypy/issues/19221
        elif isinstance(other, Sequence) and all(isinstance(x, DigitalWaveform) for x in other):
            validate_unsupported_arg("timestamps", timestamps)
            self._append_waveforms(other)
        else:
            raise invalid_arg_type("input", "array or waveform(s)", other)

    def _append_array(
        self,
        array: npt.NDArray[_TState],
        timestamps: Sequence[dt.datetime] | Sequence[ht.datetime] | None = None,
    ) -> None:
        if array.dtype != self.dtype:
            raise data_type_mismatch("input array", array.dtype, "waveform", self.dtype)

        if array.ndim == 1:
            array_signal_count = 1
            array = array.reshape(len(array), 1)
        elif array.ndim == 2:
            array_signal_count = array.shape[1]
        else:
            raise invalid_array_ndim("input array", "one or two-dimensional array", array.ndim)

        if array_signal_count != self.signal_count:
            raise signal_count_mismatch(
                "input array", array_signal_count, "waveform", self.signal_count
            )

        if timestamps is not None and len(array) != len(timestamps):
            raise irregular_timestamp_count_mismatch(
                len(timestamps), "input array length", len(array)
            )

        new_timing = self._timing._append_timestamps(timestamps)

        self._increase_capacity(len(array))
        self._set_timing(new_timing)

        offset = self._start_index + self._sample_count
        self._data[offset : offset + len(array)] = array
        self._sample_count += len(array)

    def _append_waveform(self, waveform: DigitalWaveform[_TState]) -> None:
        self._append_waveforms([waveform])

    def _append_waveforms(self, waveforms: Sequence[DigitalWaveform[_TState]]) -> None:
        for waveform in waveforms:
            if waveform.dtype != self.dtype:
                raise data_type_mismatch("input waveform", waveform.dtype, "waveform", self.dtype)

        new_timing = self._timing
        for waveform in waveforms:
            new_timing = new_timing._append_timing(waveform._timing)

        self._increase_capacity(sum(waveform.sample_count for waveform in waveforms))
        self._set_timing(new_timing)

        offset = self._start_index + self._sample_count
        for waveform in waveforms:
            self._data[offset : offset + waveform.sample_count] = waveform.data
            offset += waveform.sample_count
            self._sample_count += waveform.sample_count
            self._extended_properties._merge(waveform._extended_properties)

    def _increase_capacity(self, amount: int) -> None:
        new_capacity = self._start_index + self._sample_count + amount
        if new_capacity > self.capacity:
            self.capacity = new_capacity

    def load_data(
        self,
        array: npt.NDArray[_TState],
        *,
        copy: bool = True,
        start_index: SupportsIndex | None = 0,
        sample_count: SupportsIndex | None = None,
    ) -> None:
        """Load new data into an existing waveform.

        Args:
            array: A NumPy array containing the data to load.
            copy: Specifies whether to copy the array or save a reference to it.
            start_index: The sample index at which the waveform data begins.
            sample_count: The number of samples in the waveform.
        """
        if isinstance(array, np.ndarray):
            self._load_array(array, copy=copy, start_index=start_index, sample_count=sample_count)
        else:
            raise invalid_arg_type("input array", "array", array)

    def _load_array(
        self,
        array: npt.NDArray[_TState],
        *,
        copy: bool = True,
        start_index: SupportsIndex | None = 0,
        sample_count: SupportsIndex | None = None,
        signal_count: SupportsIndex | None = None,
    ) -> None:
        if array.dtype != self.dtype:
            raise data_type_mismatch("input array", array.dtype, "waveform", self.dtype)

        if array.ndim == 1:
            array_signal_count = 1
            array = array.reshape(len(array), 1)
        elif array.ndim == 2:
            array_signal_count = array.shape[1]
        else:
            raise invalid_array_ndim("input array", "one or two-dimensional array", array.ndim)

        if self._timing._timestamps is not None and len(array) != len(self._timing._timestamps):
            raise irregular_timestamp_count_mismatch(
                len(self._timing._timestamps), "input array length", len(array), reversed=True
            )

        start_index = arg_to_uint("start index", start_index, 0)
        sample_count = arg_to_uint("sample count", sample_count, len(array) - start_index)
        signal_count = arg_to_uint("signal count", signal_count, array_signal_count)

        if signal_count != array_signal_count:
            raise signal_count_mismatch("input array", signal_count, "waveform", array_signal_count)

        if copy:
            if sample_count > len(self._data):
                self.capacity = sample_count
            self._data[0:sample_count] = array[start_index : start_index + sample_count]
            self._start_index = 0
            self._sample_count = sample_count
        else:
            self._data = array
            self._start_index = start_index
            self._sample_count = sample_count

    def test(
        self,
        expected_waveform: DigitalWaveform[_TState],
        *,
        start_sample: SupportsIndex | None = 0,
        expected_start_sample: SupportsIndex | None = 0,
        sample_count: SupportsIndex | None = None,
    ) -> DigitalWaveformTestResult:
        """Test the digital waveform against an expected digital waveform.

        Args:
            expected_waveform: The expected digital waveform to compare against.
            start_sample: The beginning sample of ``self`` to compare.
            expected_start_sample: The beginning sample of ``expected_waveform`` to compare.
            sample_count: The number of samples to compare.

        Returns:
            The test result.
        """
        start_sample = arg_to_uint("start sample", start_sample, 0)
        expected_start_sample = arg_to_uint("expected start sample", expected_start_sample, 0)
        sample_count = arg_to_uint("sample count", sample_count, self.sample_count - start_sample)

        if self.signal_count != expected_waveform.signal_count:
            raise signal_count_mismatch(
                "expected waveform", expected_waveform.signal_count, "waveform", self.signal_count
            )
        if start_sample + sample_count > self.sample_count:
            raise start_index_or_sample_count_too_large(
                start_sample, sample_count, "number of samples in the waveform", self.sample_count
            )
        if expected_start_sample + sample_count > expected_waveform.sample_count:
            raise start_index_or_sample_count_too_large(
                expected_start_sample,
                sample_count,
                "number of samples in the expected waveform",
                expected_waveform.sample_count,
            )

        failures = []
        for _ in range(sample_count):
            for signal_index in range(self.signal_count):
                actual_state = DigitalState(self.data[start_sample, signal_index])
                expected_state = DigitalState(
                    expected_waveform.data[expected_start_sample, signal_index]
                )
                if DigitalState.test(actual_state, expected_state):
                    failures.append(
                        DigitalWaveformFailure(
                            start_sample,
                            expected_start_sample,
                            signal_index,
                            actual_state,
                            expected_state,
                        )
                    )
            start_sample += 1
            expected_start_sample += 1

        return DigitalWaveformTestResult(failures)

    def __eq__(self, value: object, /) -> bool:
        """Return self==value."""
        if not isinstance(value, self.__class__):
            return NotImplemented
        return (
            self.dtype == value.dtype
            and np.array_equal(self.data, value.data)
            and self._extended_properties == value._extended_properties
            and self._timing == value._timing
        )

    def __reduce__(self) -> tuple[Any, ...]:
        """Return object state for pickling."""
        ctor_args = (self._sample_count, self.signal_count, self.dtype)
        ctor_kwargs: dict[str, Any] = {
            "data": self.data,
            "extended_properties": self._extended_properties,
            "copy_extended_properties": False,
            "timing": self._timing,
        }
        return (self.__class__._unpickle, (ctor_args, ctor_kwargs))

    @classmethod
    def _unpickle(cls, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Self:
        return cls(*args, **kwargs)

    def __repr__(self) -> str:
        """Return repr(self)."""
        args = [f"{self._sample_count}, {self.signal_count}"]
        if self.dtype != np.uint8:
            args.append(f"{self.dtype.name}")
        # start_index and capacity are not shown because they are allocation details. data hides
        # the unused data before start_index and after start_index+sample_count.
        if self._sample_count > 0:
            # Hack: undo NumPy's line wrapping
            args.append(f"data={self.data!r}".replace("\n      ", ""))
        if self._extended_properties:
            args.append(f"extended_properties={self._extended_properties._properties!r}")
        if self._timing is not Timing.empty:
            args.append(f"timing={self._timing!r}")
        return f"{self.__class__.__module__}.{self.__class__.__name__}({', '.join(args)})"
