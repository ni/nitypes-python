from __future__ import annotations

import datetime as dt
import sys
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Generic, SupportsIndex, TypeVar, Union, cast, overload

import hightime as ht
import numpy as np
import numpy.typing as npt

from nitypes._arguments import arg_to_uint, validate_dtype, validate_unsupported_arg
from nitypes._exceptions import invalid_arg_type, invalid_array_ndim
from nitypes.waveform._exceptions import (
    input_array_data_type_mismatch,
    input_waveform_data_type_mismatch,
)
from nitypes.waveform._extended_properties import (
    CHANNEL_NAME,
    UNIT_DESCRIPTION,
    ExtendedPropertyDictionary,
    ExtendedPropertyValue,
)
from nitypes.waveform._scaling import NO_SCALING, ScaleMode
from nitypes.waveform._timing import BaseTiming, PrecisionTiming, Timing, convert_timing
from nitypes.waveform._warnings import scale_mode_mismatch

try:
    from typing import Self, TypeAlias
except ImportError:
    from nitypes._typing import Self, TypeAlias

if sys.version_info < (3, 10):
    import array as std_array


_ScalarType = TypeVar("_ScalarType", bound=np.generic)
_ScalarType_co = TypeVar("_ScalarType_co", bound=np.generic, covariant=True)

_AnyTiming: TypeAlias = Union[BaseTiming[Any, Any], Timing, PrecisionTiming]
_TTiming = TypeVar("_TTiming", bound=BaseTiming[Any, Any])

# Use the C types here because np.isdtype() considers some of them to be distinct types, even when
# they have the same size (e.g. np.intc vs. np.int_ vs. np.long).
_ANALOG_DTYPES = (
    # Floating point
    np.single,
    np.double,
    # Signed integers
    np.byte,
    np.short,
    np.intc,
    np.int_,
    np.long,
    np.longlong,
    # Unsigned integers
    np.ubyte,
    np.ushort,
    np.uintc,
    np.uint,
    np.ulong,
    np.ulonglong,
)

_SCALED_DTYPES = (
    # Floating point
    np.single,
    np.double,
)

# Note about NumPy type hints:
# - At time of writing (April 2025), shape typing is still under development, so we do not
#   distinguish between 1D and 2D arrays in type hints.
# - npt.ArrayLike accepts some types that np.asarray() does not, such as buffers, so we are
#   explicitly using npt.NDArray | Sequence instead of npt.ArrayLike.
# - _ScalarType is bound to np.generic, so Sequence[_ScalarType] will not match list[int].
# - We are not using PEP 696 – Type Defaults for Type Parameters because it makes the type parameter
#   default to np.float64 in some cases where it should be inferred as Any, such as when dtype is
#   specified as a str.


class AnalogWaveform(Generic[_ScalarType_co]):
    """An analog waveform, which encapsulates analog data and timing information."""

    @overload
    @staticmethod
    def from_array_1d(
        array: npt.NDArray[_ScalarType],
        dtype: None = ...,
        *,
        copy: bool = ...,
        start_index: SupportsIndex | None = ...,
        sample_count: SupportsIndex | None = ...,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = ...,
        timing: Timing | PrecisionTiming | None = ...,
        scale_mode: ScaleMode | None = ...,
    ) -> AnalogWaveform[_ScalarType]: ...

    @overload
    @staticmethod
    def from_array_1d(
        array: npt.NDArray[Any] | Sequence[Any],
        dtype: type[_ScalarType] | np.dtype[_ScalarType] = ...,
        *,
        copy: bool = ...,
        start_index: SupportsIndex | None = ...,
        sample_count: SupportsIndex | None = ...,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = ...,
        timing: Timing | PrecisionTiming | None = ...,
        scale_mode: ScaleMode | None = ...,
    ) -> AnalogWaveform[_ScalarType]: ...

    @overload
    @staticmethod
    def from_array_1d(
        array: npt.NDArray[Any] | Sequence[Any],
        dtype: npt.DTypeLike = ...,
        *,
        copy: bool = ...,
        start_index: SupportsIndex | None = ...,
        sample_count: SupportsIndex | None = ...,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = ...,
        timing: Timing | PrecisionTiming | None = ...,
        scale_mode: ScaleMode | None = ...,
    ) -> AnalogWaveform[Any]: ...

    @staticmethod
    def from_array_1d(
        array: npt.NDArray[Any] | Sequence[Any],
        dtype: npt.DTypeLike = None,
        *,
        copy: bool = True,
        start_index: SupportsIndex | None = 0,
        sample_count: SupportsIndex | None = None,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = None,
        timing: Timing | PrecisionTiming | None = None,
        scale_mode: ScaleMode | None = None,
    ) -> AnalogWaveform[_ScalarType]:
        """Construct an analog waveform from a one-dimensional array or sequence.

        Args:
            array: The analog waveform data as a one-dimensional array or a sequence.
            dtype: The NumPy data type for the analog waveform data. This argument is required
                when array is a sequence.
            copy: Specifies whether to copy the array or save a reference to it.
            start_index: The sample index at which the analog waveform data begins.
            sample_count: The number of samples in the analog waveform.
            extended_properties: The extended properties of the analog waveform.
            timing: The timing information of the analog waveform.
            scale_mode: The scale mode of the analog waveform.

        Returns:
            An analog waveform containing the specified data.
        """
        if isinstance(array, np.ndarray):
            if array.ndim != 1:
                raise invalid_array_ndim(
                    "input array", "one-dimensional array or sequence", array.ndim
                )
        elif isinstance(array, Sequence) or (
            sys.version_info < (3, 10) and isinstance(array, std_array.array)
        ):
            if dtype is None:
                raise ValueError("You must specify a dtype when the input array is a sequence.")
        else:
            raise invalid_arg_type("input array", "one-dimensional array or sequence", array)

        return AnalogWaveform(
            raw_data=np.asarray(array, dtype, copy=copy),
            start_index=start_index,
            sample_count=sample_count,
            extended_properties=extended_properties,
            timing=timing,
            scale_mode=scale_mode,
        )

    @overload
    @staticmethod
    def from_array_2d(
        array: npt.NDArray[_ScalarType],
        dtype: None = ...,
        *,
        copy: bool = ...,
        start_index: SupportsIndex | None = ...,
        sample_count: SupportsIndex | None = ...,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = ...,
        timing: Timing | PrecisionTiming | None = ...,
        scale_mode: ScaleMode | None = ...,
    ) -> list[AnalogWaveform[_ScalarType]]: ...

    @overload
    @staticmethod
    def from_array_2d(
        array: npt.NDArray[Any] | Sequence[Sequence[Any]],
        dtype: type[_ScalarType] | np.dtype[_ScalarType] = ...,
        *,
        copy: bool = ...,
        start_index: SupportsIndex | None = ...,
        sample_count: SupportsIndex | None = ...,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = ...,
        timing: Timing | PrecisionTiming | None = ...,
        scale_mode: ScaleMode | None = ...,
    ) -> list[AnalogWaveform[_ScalarType]]: ...

    @overload
    @staticmethod
    def from_array_2d(
        array: npt.NDArray[Any] | Sequence[Sequence[Any]],
        dtype: npt.DTypeLike = ...,
        *,
        copy: bool = ...,
        start_index: SupportsIndex | None = ...,
        sample_count: SupportsIndex | None = ...,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = ...,
        timing: Timing | PrecisionTiming | None = ...,
        scale_mode: ScaleMode | None = ...,
    ) -> list[AnalogWaveform[Any]]: ...

    @staticmethod
    def from_array_2d(
        array: npt.NDArray[Any] | Sequence[Sequence[Any]],
        dtype: npt.DTypeLike = None,
        *,
        copy: bool = True,
        start_index: SupportsIndex | None = 0,
        sample_count: SupportsIndex | None = None,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = None,
        timing: Timing | PrecisionTiming | None = None,
        scale_mode: ScaleMode | None = None,
    ) -> list[AnalogWaveform[_ScalarType]]:
        """Construct a list of analog waveforms from a two-dimensional array or nested sequence.

        Args:
            array: The analog waveform data as a two-dimensional array or a nested sequence.
            dtype: The NumPy data type for the analog waveform data. This argument is required
                when array is a sequence.
            copy: Specifies whether to copy the array or save a reference to it.
            start_index: The sample index at which the analog waveform data begins.
            sample_count: The number of samples in the analog waveform.
            extended_properties: The extended properties of the analog waveform.
            timing: The timing information of the analog waveform.
            scale_mode: The scale mode of the analog waveform.

        Returns:
            A list containing an analog waveform for each row of the specified data.

        When constructing multiple analog waveforms, the same extended properties, timing
        information, and scale mode are applied to all analog waveforms. Consider assigning
        these properties after construction.
        """
        if isinstance(array, np.ndarray):
            if array.ndim != 2:
                raise invalid_array_ndim(
                    "input array", "two-dimensional array or nested sequence", array.ndim
                )
        elif isinstance(array, Sequence) or (
            sys.version_info < (3, 10) and isinstance(array, std_array.array)
        ):
            if dtype is None:
                raise ValueError("You must specify a dtype when the input array is a sequence.")
        else:
            raise invalid_arg_type("input array", "two-dimensional array or nested sequence", array)

        return [
            AnalogWaveform(
                raw_data=np.asarray(array[i], dtype, copy=copy),
                start_index=start_index,
                sample_count=sample_count,
                extended_properties=extended_properties,
                timing=timing,
                scale_mode=scale_mode,
            )
            for i in range(len(array))
        ]

    __slots__ = [
        "_data",
        "_start_index",
        "_sample_count",
        "_extended_properties",
        "_timing",
        "_converted_timing_cache",
        "_scale_mode",
        "__weakref__",
    ]

    _data: npt.NDArray[_ScalarType_co]
    _start_index: int
    _sample_count: int
    _extended_properties: ExtendedPropertyDictionary
    _timing: BaseTiming[Any, Any]
    _converted_timing_cache: dict[type[_AnyTiming], _AnyTiming]
    _scale_mode: ScaleMode

    # If neither dtype nor _data is specified, the type parameter defaults to np.float64.
    @overload
    def __init__(  # noqa: D107 - Missing docstring in __init__ (auto-generated noqa)
        self: AnalogWaveform[np.float64],
        sample_count: SupportsIndex | None = ...,
        dtype: None = ...,
        *,
        raw_data: None = ...,
        start_index: SupportsIndex | None = ...,
        capacity: SupportsIndex | None = ...,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = ...,
        timing: Timing | PrecisionTiming | None = ...,
        scale_mode: ScaleMode | None = ...,
    ) -> None: ...

    @overload
    def __init__(  # noqa: D107 - Missing docstring in __init__ (auto-generated noqa)
        self: AnalogWaveform[_ScalarType_co],
        sample_count: SupportsIndex | None = ...,
        dtype: type[_ScalarType_co] | np.dtype[_ScalarType_co] = ...,
        *,
        raw_data: None = ...,
        start_index: SupportsIndex | None = ...,
        capacity: SupportsIndex | None = ...,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = ...,
        timing: Timing | PrecisionTiming | None = ...,
        scale_mode: ScaleMode | None = ...,
    ) -> None: ...

    @overload
    def __init__(  # noqa: D107 - Missing docstring in __init__ (auto-generated noqa)
        self: AnalogWaveform[_ScalarType_co],
        sample_count: SupportsIndex | None = ...,
        dtype: None = ...,
        *,
        raw_data: npt.NDArray[_ScalarType_co] | None = ...,
        start_index: SupportsIndex | None = ...,
        capacity: SupportsIndex | None = ...,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = ...,
        timing: Timing | PrecisionTiming | None = ...,
        scale_mode: ScaleMode | None = ...,
    ) -> None: ...

    @overload
    def __init__(  # noqa: D107 - Missing docstring in __init__ (auto-generated noqa)
        self: AnalogWaveform[Any],
        sample_count: SupportsIndex | None = ...,
        dtype: npt.DTypeLike = ...,
        *,
        raw_data: npt.NDArray[Any] | None = ...,
        start_index: SupportsIndex | None = ...,
        capacity: SupportsIndex | None = ...,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = ...,
        timing: Timing | PrecisionTiming | None = ...,
        scale_mode: ScaleMode | None = ...,
    ) -> None: ...

    def __init__(
        self,
        sample_count: SupportsIndex | None = None,
        dtype: npt.DTypeLike = None,
        *,
        raw_data: npt.NDArray[_ScalarType_co] | None = None,
        start_index: SupportsIndex | None = None,
        capacity: SupportsIndex | None = None,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = None,
        copy_extended_properties: bool = True,
        timing: Timing | PrecisionTiming | None = None,
        scale_mode: ScaleMode | None = None,
    ) -> None:
        """Construct an analog waveform.

        Args:
            sample_count: The number of samples in the analog waveform.
            dtype: The NumPy data type for the analog waveform data. If not specified, the data
                type defaults to np.float64.
            raw_data: A NumPy ndarray to use for sample storage. The analog waveform takes ownership
                of this array. If not specified, an ndarray is created based on the specified dtype,
                start index, sample count, and capacity.
            start_index: The sample index at which the analog waveform data begins.
            sample_count: The number of samples in the analog waveform.
            capacity: The number of samples to allocate. Pre-allocating a larger buffer optimizes
                appending samples to the waveform.
            extended_properties: The extended properties of the analog waveform.
            copy_extended_properties: Specifies whether to copy the extended properties or take
                ownership.
            timing: The timing information of the analog waveform.
            scale_mode: The scale mode of the analog waveform.

        Returns:
            An analog waveform.
        """
        if raw_data is None:
            self._init_with_new_array(
                sample_count, dtype, start_index=start_index, capacity=capacity
            )
        elif isinstance(raw_data, np.ndarray):
            self._init_with_provided_array(
                raw_data,
                dtype,
                start_index=start_index,
                sample_count=sample_count,
                capacity=capacity,
            )
        else:
            raise invalid_arg_type("raw data", "NumPy ndarray", raw_data)

        if copy_extended_properties or not isinstance(
            extended_properties, ExtendedPropertyDictionary
        ):
            extended_properties = ExtendedPropertyDictionary(extended_properties)
        self._extended_properties = extended_properties

        if timing is None:
            timing = Timing.empty
        self._timing = timing
        self._converted_timing_cache = {}

        if scale_mode is None:
            scale_mode = NO_SCALING
        self._scale_mode = scale_mode

    def _init_with_new_array(
        self,
        sample_count: SupportsIndex | None = None,
        dtype: npt.DTypeLike = None,
        *,
        start_index: SupportsIndex | None = None,
        capacity: SupportsIndex | None = None,
    ) -> None:
        start_index = arg_to_uint("start index", start_index, 0)
        sample_count = arg_to_uint("sample count", sample_count, 0)
        capacity = arg_to_uint("capacity", capacity, sample_count)

        if dtype is None:
            dtype = np.float64
        validate_dtype(dtype, _ANALOG_DTYPES)

        if start_index > capacity:
            raise ValueError(
                "The start index must be less than or equal to the capacity.\n\n"
                f"Start index: {start_index}\n"
                f"Capacity: {capacity}"
            )
        if start_index + sample_count > capacity:
            raise ValueError(
                "The sum of the start index and sample count must be less than or equal to the capacity.\n\n"
                f"Start index: {start_index}\n"
                f"Sample count: {sample_count}\n"
                f"Capacity: {capacity}"
            )

        self._data = np.zeros(capacity, dtype)
        self._start_index = start_index
        self._sample_count = sample_count

    def _init_with_provided_array(
        self,
        data: npt.NDArray[_ScalarType_co],
        dtype: npt.DTypeLike = None,
        *,
        start_index: SupportsIndex | None = None,
        sample_count: SupportsIndex | None = None,
        capacity: SupportsIndex | None = None,
    ) -> None:
        if not isinstance(data, np.ndarray):
            raise invalid_arg_type("input array", "one-dimensional array", data)
        if data.ndim != 1:
            raise invalid_array_ndim("input array", "one-dimensional array", data.ndim)

        if dtype is None:
            dtype = data.dtype
        if dtype != data.dtype:
            raise TypeError(
                "The data type of the input array must match the requested data type.\n\n"
                f"Array data type: {data.dtype}\n"
                f"Requested data type: {np.dtype(dtype)}"
            )
        validate_dtype(dtype, _ANALOG_DTYPES)

        capacity = arg_to_uint("capacity", capacity, len(data))
        if capacity != len(data):
            raise ValueError(
                "The capacity must match the input array length.\n\n"
                f"Capacity: {capacity}\n"
                f"Array length: {len(data)}"
            )

        start_index = arg_to_uint("start index", start_index, 0)
        if start_index > capacity:
            raise ValueError(
                "The start index must be less than or equal to the input array length.\n\n"
                f"Start index: {start_index}\n"
                f"Capacity: {capacity}"
            )

        sample_count = arg_to_uint("sample count", sample_count, len(data) - start_index)
        if start_index + sample_count > len(data):
            raise ValueError(
                "The sum of the start index and sample count must be less than or equal to the input array length.\n\n"
                f"Start index: {start_index}\n"
                f"Sample count: {sample_count}\n"
                f"Array length: {len(data)}"
            )

        self._data = data
        self._start_index = start_index
        self._sample_count = sample_count

    @property
    def raw_data(self) -> npt.NDArray[_ScalarType_co]:
        """The raw analog waveform data."""
        return self._data[self._start_index : self._start_index + self._sample_count]

    def get_raw_data(
        self, start_index: SupportsIndex | None = 0, sample_count: SupportsIndex | None = None
    ) -> npt.NDArray[_ScalarType_co]:
        """Get a subset of the raw analog waveform data.

        Args:
            start_index: The sample index at which the data begins.
            sample_count: The number of samples to return.

        Returns:
            A subset of the raw analog waveform data.
        """
        start_index = arg_to_uint("sample index", start_index, 0)
        if start_index > self.sample_count:
            raise ValueError(
                "The start index must be less than or equal to the number of samples in the waveform.\n\n"
                f"Start index: {start_index}\n"
                f"Number of samples: {self.sample_count}"
            )

        sample_count = arg_to_uint("sample count", sample_count, self.sample_count - start_index)
        if start_index + sample_count > self.sample_count:
            raise ValueError(
                "The sum of the start index and sample count must be less than or equal to the number of samples in the waveform.\n\n"
                f"Start index: {start_index}\n"
                f"Sample count: {sample_count}\n"
                f"Number of samples: {self.sample_count}"
            )

        return self.raw_data[start_index : start_index + sample_count]

    @property
    def scaled_data(self) -> npt.NDArray[np.float64]:
        """The scaled analog waveform data.

        This property converts all of the waveform samples to float64 and scales them. To scale a
        subset of the waveform or convert to float32, use the get_scaled_data() method instead.
        """
        return self.get_scaled_data()

    # If dtype is not specified, _ScaledDataType defaults to np.float64.
    @overload
    def get_scaled_data(  # noqa: D107 - Missing docstring in __init__ (auto-generated noqa)
        self,
        dtype: None = ...,
        *,
        start_index: SupportsIndex | None = ...,
        sample_count: SupportsIndex | None = ...,
    ) -> npt.NDArray[np.float64]: ...

    @overload
    def get_scaled_data(  # noqa: D107 - Missing docstring in __init__ (auto-generated noqa)
        self,
        dtype: type[_ScalarType] | np.dtype[_ScalarType] = ...,
        *,
        start_index: SupportsIndex | None = ...,
        sample_count: SupportsIndex | None = ...,
    ) -> npt.NDArray[_ScalarType]: ...

    @overload
    def get_scaled_data(  # noqa: D107 - Missing docstring in __init__ (auto-generated noqa)
        self,
        dtype: npt.DTypeLike = ...,
        *,
        start_index: SupportsIndex | None = ...,
        sample_count: SupportsIndex | None = ...,
    ) -> npt.NDArray[Any]: ...

    def get_scaled_data(
        self,
        dtype: npt.DTypeLike = None,
        *,
        start_index: SupportsIndex | None = 0,
        sample_count: SupportsIndex | None = None,
    ) -> npt.NDArray[Any]:
        """Get a subset of the scaled analog waveform data with the specified dtype.

        Args:
            dtype: The NumPy data type to use for scaled data.
            start_index: The sample index at which to start scaling.
            sample_count: The number of samples to scale.

        Returns:
            A subset of the scaled analog waveform data.
        """
        if dtype is None:
            dtype = np.float64
        validate_dtype(dtype, _SCALED_DTYPES)

        raw_data = self.get_raw_data(start_index, sample_count)
        converted_data = raw_data.astype(dtype)
        return self._scale_mode._transform_data(converted_data)

    @property
    def sample_count(self) -> int:
        """The number of samples in the analog waveform."""
        return self._sample_count

    @property
    def capacity(self) -> int:
        """The total capacity available for analog waveform data.

        Setting the capacity resizes the underlying NumPy array in-place.

        * Other Python objects with references to the array will see the array size change.
        * If the array has a reference to an external buffer (such as an array.array), attempting
          to resize it raises ValueError.
        """
        return len(self._data)

    @capacity.setter
    def capacity(self, value: int) -> None:
        value = arg_to_uint("capacity", value)
        if value < self._start_index + self._sample_count:
            raise ValueError(
                "The capacity must be equal to or greater than the number of samples in the waveform.\n\n"
                f"Capacity: {value}\n"
                f"Number of samples: {self._start_index + self._sample_count}"
            )
        if value != len(self._data):
            self._data.resize(value, refcheck=False)

    @property
    def dtype(self) -> np.dtype[_ScalarType_co]:
        """The NumPy dtype for the analog waveform data."""
        return self._data.dtype

    @property
    def extended_properties(self) -> ExtendedPropertyDictionary:
        """The extended properties for the analog waveform."""
        return self._extended_properties

    @property
    def channel_name(self) -> str:
        """The name of the device channel from which the analog waveform was acquired."""
        value = self._extended_properties.get(CHANNEL_NAME, "")
        assert isinstance(value, str)
        return value

    @channel_name.setter
    def channel_name(self, value: str) -> None:
        if not isinstance(value, str):
            raise invalid_arg_type("channel name", "str", value)
        self._extended_properties[CHANNEL_NAME] = value

    @property
    def unit_description(self) -> str:
        """The unit of measurement, such as volts, of the analog waveform."""
        value = self._extended_properties.get(UNIT_DESCRIPTION, "")
        assert isinstance(value, str)
        return value

    @unit_description.setter
    def unit_description(self, value: str) -> None:
        if not isinstance(value, str):
            raise invalid_arg_type("unit description", "str", value)
        self._extended_properties[UNIT_DESCRIPTION] = value

    def _get_timing(self, requested_type: type[_TTiming]) -> _TTiming:
        if isinstance(self._timing, requested_type):
            return self._timing
        value = cast(_TTiming, self._converted_timing_cache.get(requested_type))
        if value is None:
            value = convert_timing(requested_type, self._timing)
            self._converted_timing_cache[requested_type] = value
        return value

    def _set_timing(self, value: _TTiming) -> None:
        if self._timing is not value:
            self._timing = value
            self._converted_timing_cache.clear()

    def _validate_timing(self, value: _TTiming) -> None:
        if value._timestamps is not None and len(value._timestamps) != self._sample_count:
            raise ValueError(
                "The number of irregular timestamps is not equal to the number of samples in the waveform.\n\n"
                f"Number of timestamps: {len(value._timestamps)}\n"
                f"Number of samples in the waveform: {self._sample_count}"
            )

    @property
    def timing(self) -> Timing:
        """The timing information of the analog waveform.

        The default value is Timing.empty.
        """
        return self._get_timing(Timing)

    @timing.setter
    def timing(self, value: Timing) -> None:
        if not isinstance(value, Timing):
            raise invalid_arg_type("timing information", "Timing object", value)
        self._validate_timing(value)
        self._set_timing(value)

    @property
    def is_precision_timing_initialized(self) -> bool:
        """Indicates whether the waveform's timing information was set using precision timing."""
        return isinstance(self._timing, PrecisionTiming)

    @property
    def precision_timing(self) -> PrecisionTiming:
        """The precision timing information of the analog waveform.

        The default value is PrecisionTiming.empty.

        Use AnalogWaveform.precision_timing instead of AnalogWaveform.timing to obtain timing
        information with higher precision than AnalogWaveform.timing. If the timing information is
        set using AnalogWaveform.precision_timing, then this property returns timing information
        with up to yoctosecond precision. If the timing information is set using
        AnalogWaveform.timing, then the timing information returned has up to microsecond precision.

        Accessing this property can potentially decrease performance if the timing information is
        set using AnalogWaveform.timing. Use AnalogWaveform.is_precision_timing_initialized to
        determine if AnalogWaveform.precision_timing has been initialized.
        """
        return self._get_timing(PrecisionTiming)

    @precision_timing.setter
    def precision_timing(self, value: PrecisionTiming) -> None:
        if not isinstance(value, PrecisionTiming):
            raise invalid_arg_type("precision timing information", "PrecisionTiming object", value)
        self._validate_timing(value)
        self._set_timing(value)

    @property
    def scale_mode(self) -> ScaleMode:
        """The scale mode of the analog waveform."""
        return self._scale_mode

    @scale_mode.setter
    def scale_mode(self, value: ScaleMode) -> None:
        if not isinstance(value, ScaleMode):
            raise invalid_arg_type("scale mode", "ScaleMode object", value)
        self._scale_mode = value

    def append(
        self,
        other: (
            npt.NDArray[_ScalarType_co]
            | AnalogWaveform[_ScalarType_co]
            | Sequence[AnalogWaveform[_ScalarType_co]]
        ),
        /,
        timestamps: Sequence[dt.datetime] | Sequence[ht.datetime] | None = None,
    ) -> None:
        """Append data to the analog waveform.

        Args:
            other: The array or waveform(s) to append.
            timestamps: A sequence of timestamps. When the current waveform has
                SampleIntervalMode.IRREGULAR, you must provide a sequence of timestamps with the
                same length as the array.

        Raises:
            TimingMismatchError: The current and other waveforms have incompatible timing.
            TimingMismatchWarning: The sample intervals of the waveform(s) do not match.
            ScalingMismatchWarning: The scale modes of the waveform(s) do not match.
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

        * If the scale mode of other waveform(s) does not match the scale mode of the current
          waveform, a ScalingMismatchWarning is generated. Otherwise, the scaling information of the
          other waveform(s) is discarded.
        """
        if isinstance(other, np.ndarray):
            self._append_array(other, timestamps)
        elif isinstance(other, AnalogWaveform):
            validate_unsupported_arg("timestamps", timestamps)
            self._append_waveform(other)
        elif isinstance(other, Sequence) and all(isinstance(x, AnalogWaveform) for x in other):
            validate_unsupported_arg("timestamps", timestamps)
            self._append_waveforms(other)
        else:
            raise invalid_arg_type("input", "array or waveform(s)", other)

    def _append_array(
        self,
        array: npt.NDArray[_ScalarType_co],
        timestamps: Sequence[dt.datetime] | Sequence[ht.datetime] | None = None,
    ) -> None:
        if array.dtype != self.dtype:
            raise input_array_data_type_mismatch(array.dtype, self.dtype)
        if array.ndim != 1:
            raise invalid_array_ndim("input array", "one-dimensional array", array.ndim)
        if timestamps is not None and len(array) != len(timestamps):
            raise ValueError(
                "The number of irregular timestamps must be equal to the input array length.\n\n"
                f"Number of timestamps: {len(timestamps)}\n"
                f"Array length: {len(array)}"
            )

        new_timing = self._timing._append_timestamps(timestamps)

        self._increase_capacity(len(array))
        self._set_timing(new_timing)

        offset = self._start_index + self._sample_count
        self._data[offset : offset + len(array)] = array
        self._sample_count += len(array)

    def _append_waveform(self, waveform: AnalogWaveform[_ScalarType_co]) -> None:
        self._append_waveforms([waveform])

    def _append_waveforms(self, waveforms: Sequence[AnalogWaveform[_ScalarType_co]]) -> None:
        for waveform in waveforms:
            if waveform.dtype != self.dtype:
                raise input_waveform_data_type_mismatch(waveform.dtype, self.dtype)
            if waveform._scale_mode != self._scale_mode:
                warnings.warn(scale_mode_mismatch())

        new_timing = self._timing
        for waveform in waveforms:
            new_timing = new_timing._append_timing(waveform._timing)

        self._increase_capacity(sum(waveform.sample_count for waveform in waveforms))
        self._set_timing(new_timing)

        offset = self._start_index + self._sample_count
        for waveform in waveforms:
            self._data[offset : offset + waveform.sample_count] = waveform.raw_data
            offset += waveform.sample_count
            self._sample_count += waveform.sample_count
            self._extended_properties._merge(waveform._extended_properties)

    def _increase_capacity(self, amount: int) -> None:
        new_capacity = self._start_index + self._sample_count + amount
        if new_capacity > self.capacity:
            self.capacity = new_capacity

    def load_data(
        self,
        array: npt.NDArray[_ScalarType_co],
        *,
        copy: bool = True,
        start_index: SupportsIndex | None = 0,
        sample_count: SupportsIndex | None = None,
    ) -> None:
        """Load new data into an existing waveform.

        Args:
            array: A NumPy array containing the data to load.
            copy: Specifies whether to copy the array or save a reference to it.
            start_index: The sample index at which the analog waveform data begins.
            sample_count: The number of samples in the analog waveform.
        """
        if isinstance(array, np.ndarray):
            self._load_array(array, copy=copy, start_index=start_index, sample_count=sample_count)
        else:
            raise invalid_arg_type("input array", "array", array)

    def _load_array(
        self,
        array: npt.NDArray[_ScalarType_co],
        *,
        copy: bool = True,
        start_index: SupportsIndex | None = 0,
        sample_count: SupportsIndex | None = None,
    ) -> None:
        if array.dtype != self.dtype:
            raise input_array_data_type_mismatch(array.dtype, self.dtype)
        if array.ndim != 1:
            raise invalid_array_ndim("input array", "one-dimensional array", array.ndim)
        if self._timing._timestamps is not None and len(array) != len(self._timing._timestamps):
            raise ValueError(
                "The input array length must be equal to the number of irregular timestamps.\n\n"
                f"Array length: {len(array)}\n"
                f"Number of timestamps: {len(self._timing._timestamps)}"
            )

        start_index = arg_to_uint("start index", start_index, 0)
        sample_count = arg_to_uint("sample count", sample_count, len(array) - start_index)

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

    def __eq__(self, value: object, /) -> bool:
        """Return self==value."""
        if not isinstance(value, self.__class__):
            return NotImplemented
        return (
            self.dtype == value.dtype
            and np.array_equal(self.raw_data, value.raw_data)
            and self._extended_properties == value._extended_properties
            and self._timing == value._timing
            and self._scale_mode == value._scale_mode
        )

    def __reduce__(self) -> tuple[Any, ...]:
        """Return object state for pickling."""
        ctor_args = (self._sample_count, self.dtype)
        ctor_kwargs: dict[str, Any] = {
            "raw_data": self.raw_data,
            "extended_properties": self._extended_properties,
            "copy_extended_properties": False,
            "timing": self._timing,
            "scale_mode": self._scale_mode,
        }
        return (self.__class__._unpickle, (ctor_args, ctor_kwargs))

    @classmethod
    def _unpickle(cls, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Self:
        return cls(*args, **kwargs)

    def __repr__(self) -> str:
        """Return repr(self)."""
        args = [f"{self._sample_count}"]
        if self.dtype != np.float64:
            args.append(f"{self.dtype.name}")
        # start_index and capacity are not shown because they are allocation details. raw_data hides
        # the unused data before start_index and after start_index+sample_count.
        if self._sample_count > 0:
            args.append(f"raw_data={self.raw_data!r}")
        if self._extended_properties:
            args.append(f"extended_properties={self._extended_properties._properties!r}")
        if self._timing is not Timing.empty and self._timing is not PrecisionTiming.empty:
            args.append(f"timing={self._timing!r}")
        if self._scale_mode is not NO_SCALING:
            args.append(f"scale_mode={self._scale_mode}")
        return f"{self.__class__.__module__}.{self.__class__.__name__}({', '.join(args)})"
