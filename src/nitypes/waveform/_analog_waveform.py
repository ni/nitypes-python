from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any, Generic, SupportsIndex, TypeVar, overload

import numpy as np
import numpy.typing as npt

from nitypes._arguments import arg_to_uint, validate_dtype
from nitypes._exceptions import invalid_arg_type, invalid_array_ndim
from nitypes.waveform._extended_properties import (
    CHANNEL_NAME,
    UNIT_DESCRIPTION,
    ExtendedPropertyDictionary,
)
from nitypes.waveform._scaling import NO_SCALING, ScaleMode
from nitypes.waveform._timing._conversion import convert_timing
from nitypes.waveform._timing._precision import PrecisionTiming
from nitypes.waveform._timing._standard import Timing

if sys.version_info < (3, 10):
    import array as std_array

_ScalarType = TypeVar("_ScalarType", bound=np.generic)
_ScalarType_co = TypeVar("_ScalarType_co", bound=np.generic, covariant=True)

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
    ) -> AnalogWaveform[Any]: ...

    @staticmethod
    def from_array_1d(
        array: npt.NDArray[Any] | Sequence[Any],
        dtype: npt.DTypeLike = None,
        *,
        copy: bool = True,
        start_index: SupportsIndex | None = 0,
        sample_count: SupportsIndex | None = None,
    ) -> AnalogWaveform[_ScalarType]:
        """Construct an analog waveform from a one-dimensional array or sequence.

        Args:
            array: The analog waveform data as a one-dimensional array or a sequence.
            dtype: The NumPy data type for the analog waveform data. This argument is required
                when array is a sequence.
            copy: Specifies whether to copy the array or save a reference to it.
            start_index: The sample index at which the analog waveform data begins.
            sample_count: The number of samples in the analog waveform.

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
            _data=np.asarray(array, dtype, copy=copy),
            start_index=start_index,
            sample_count=sample_count,
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
    ) -> list[AnalogWaveform[Any]]: ...

    @staticmethod
    def from_array_2d(
        array: npt.NDArray[Any] | Sequence[Sequence[Any]],
        dtype: npt.DTypeLike = None,
        *,
        copy: bool = True,
        start_index: SupportsIndex | None = 0,
        sample_count: SupportsIndex | None = None,
    ) -> list[AnalogWaveform[_ScalarType]]:
        """Construct a list of analog waveforms from a two-dimensional array or nested sequence.

        Args:
            array: The analog waveform data as a two-dimensional array or a nested sequence.
            dtype: The NumPy data type for the analog waveform data. This argument is required
                when array is a sequence.
            copy: Specifies whether to copy the array or save a reference to it.
            start_index: The sample index at which the analog waveform data begins.
            sample_count: The number of samples in the analog waveform.

        Returns:
            A list containing an analog waveform for each row of the specified data.
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
                _data=np.asarray(array[i], dtype, copy=copy),
                start_index=start_index,
                sample_count=sample_count,
            )
            for i in range(len(array))
        ]

    __slots__ = [
        "_data",
        "_start_index",
        "_sample_count",
        "_extended_properties",
        "_timing",
        "_precision_timing",
        "_scale_mode",
        "__weakref__",
    ]

    _data: npt.NDArray[_ScalarType_co]
    _start_index: int
    _sample_count: int
    _extended_properties: ExtendedPropertyDictionary
    _timing: Timing | None
    _precision_timing: PrecisionTiming | None
    _scale_mode: ScaleMode

    # If neither dtype nor _data is specified, the type parameter defaults to np.float64.
    @overload
    def __init__(  # noqa: D107 - Missing docstring in __init__ (auto-generated noqa)
        self: AnalogWaveform[np.float64],
        sample_count: SupportsIndex | None = ...,
        dtype: None = ...,
        *,
        start_index: SupportsIndex | None = ...,
        capacity: SupportsIndex | None = ...,
        _data: None = ...,
    ) -> None: ...

    @overload
    def __init__(  # noqa: D107 - Missing docstring in __init__ (auto-generated noqa)
        self: AnalogWaveform[_ScalarType_co],
        sample_count: SupportsIndex | None = ...,
        dtype: type[_ScalarType_co] | np.dtype[_ScalarType_co] = ...,
        *,
        start_index: SupportsIndex | None = ...,
        capacity: SupportsIndex | None = ...,
        _data: None = ...,
    ) -> None: ...

    @overload
    def __init__(  # noqa: D107 - Missing docstring in __init__ (auto-generated noqa)
        self: AnalogWaveform[_ScalarType_co],
        sample_count: SupportsIndex | None = ...,
        dtype: None = ...,
        *,
        start_index: SupportsIndex | None = ...,
        capacity: SupportsIndex | None = ...,
        _data: npt.NDArray[_ScalarType_co] | None = ...,
    ) -> None: ...

    @overload
    def __init__(  # noqa: D107 - Missing docstring in __init__ (auto-generated noqa)
        self: AnalogWaveform[Any],
        sample_count: SupportsIndex | None = ...,
        dtype: npt.DTypeLike = ...,
        *,
        start_index: SupportsIndex | None = ...,
        capacity: SupportsIndex | None = ...,
        _data: npt.NDArray[Any] | None = ...,
    ) -> None: ...

    def __init__(
        self,
        sample_count: SupportsIndex | None = None,
        dtype: npt.DTypeLike = None,
        *,
        start_index: SupportsIndex | None = None,
        capacity: SupportsIndex | None = None,
        _data: npt.NDArray[_ScalarType_co] | None = None,
    ) -> None:
        """Construct an analog waveform.

        Args:
            sample_count: The number of samples in the analog waveform.
            dtype: The NumPy data type for the analog waveform data. If not specified, the data
                type defaults to np.float64.
            start_index: The sample index at which the analog waveform data begins.
            sample_count: The number of samples in the analog waveform.
            capacity: The number of samples to allocate. Pre-allocating a larger buffer optimizes
                appending samples to the waveform.

        Returns:
            An analog waveform.

        Arguments that are prefixed with an underscore are internal implementation details and are
        subject to change.
        """
        if _data is None:
            self._init_with_new_array(
                sample_count, dtype, start_index=start_index, capacity=capacity
            )
        else:
            self._init_with_provided_array(
                _data, dtype, start_index=start_index, sample_count=sample_count, capacity=capacity
            )

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
        self._extended_properties = ExtendedPropertyDictionary()
        self._timing = Timing.empty
        self._precision_timing = None
        self._scale_mode = NO_SCALING

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
        self._extended_properties = ExtendedPropertyDictionary()
        self._timing = Timing.empty
        self._precision_timing = None
        self._scale_mode = NO_SCALING

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
        - Other Python objects with references to the array will see the array size change.
        - If the array has a reference to an external buffer (such as an array.array), attempting
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

    @property
    def timing(self) -> Timing:
        """The timing information of the analog waveform.

        The default value is Timing.empty.
        """
        if self._timing is None:
            if self._precision_timing is PrecisionTiming.empty:
                self._timing = Timing.empty
            elif self._precision_timing is not None:
                self._timing = convert_timing(Timing, self._precision_timing)
            else:
                raise RuntimeError("The waveform has no timing information.")
        return self._timing

    @timing.setter
    def timing(self, value: Timing) -> None:
        if not isinstance(value, Timing):
            raise invalid_arg_type("timing information", "Timing object", value)
        self._timing = value
        self._precision_timing = None

    @property
    def is_precision_timing_initialized(self) -> bool:
        """Indicates whether the waveform's precision timing information is initialized."""
        return self._precision_timing is not None

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
        if self._precision_timing is None:
            if self._timing is Timing.empty:
                self._precision_timing = PrecisionTiming.empty
            elif self._timing is not None:
                self._precision_timing = convert_timing(PrecisionTiming, self._timing)
            else:
                raise RuntimeError("The waveform has no timing information.")
        return self._precision_timing

    @precision_timing.setter
    def precision_timing(self, value: PrecisionTiming) -> None:
        if not isinstance(value, PrecisionTiming):
            raise invalid_arg_type("precision timing information", "PrecisionTiming object", value)
        self._precision_timing = value
        self._timing = None

    @property
    def scale_mode(self) -> ScaleMode:
        """The scale mode of the analog waveform."""
        return self._scale_mode

    @scale_mode.setter
    def scale_mode(self, value: ScaleMode) -> None:
        if not isinstance(value, ScaleMode):
            raise invalid_arg_type("scale mode", "ScaleMode object", value)
        self._scale_mode = value
