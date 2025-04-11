from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Generic, SupportsIndex, TypeVar, overload

import numpy as np
import numpy.typing as npt

from nitypes.waveform._extended_properties import (
    CHANNEL_NAME,
    UNIT_DESCRIPTION,
    ExtendedPropertyDictionary,
)
from nitypes.waveform._utils import arg_to_uint

_ScalarType = TypeVar("_ScalarType", bound=np.generic)
_ScalarType_co = TypeVar("_ScalarType_co", bound=np.generic, covariant=True)

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
    def create(
        sample_count: SupportsIndex | None = ...,
        dtype: None = ...,
        *,
        capacity: SupportsIndex | None = ...,
    ) -> AnalogWaveform[np.float64]: ...

    @overload
    @staticmethod
    def create(
        sample_count: SupportsIndex | None = ...,
        dtype: type[_ScalarType] | np.dtype[_ScalarType] = ...,
        *,
        capacity: SupportsIndex | None = ...,
    ) -> AnalogWaveform[_ScalarType]: ...

    @overload
    @staticmethod
    def create(
        sample_count: SupportsIndex | None = ...,
        dtype: npt.DTypeLike = ...,
        *,
        capacity: SupportsIndex | None = ...,
    ) -> AnalogWaveform[Any]: ...

    @staticmethod
    def create(
        sample_count: SupportsIndex = 0,
        dtype: npt.DTypeLike = np.float64,
        *,
        capacity: SupportsIndex | None = None,
    ) -> AnalogWaveform[_ScalarType]:
        """Construct an analog waveform.

        Args:
            sample_count: The number of samples in the analog waveform.
            dtype: The NumPy data type for the analog waveform data. If not specified, this
                argument defaults to np.float64.
            capacity: The total capacity of the analog waveform.

        Returns:
            An analog waveform with the specified sample count, data type, and capacity.
        """
        return AnalogWaveform(dtype=dtype, sample_count=sample_count, capacity=capacity)

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
                raise ValueError(
                    f"The input array must be a one-dimensional array or sequence.\n\nNumber of dimensions: {array.ndim}"
                )
        elif isinstance(array, Sequence):
            if dtype is None:
                raise ValueError("You must specify a dtype when the input array is a sequence.")
        else:
            raise TypeError(
                f"The input array must be a one-dimensional array or sequence.\n\nType: {type(array)}"
            )

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
                raise ValueError(
                    f"The input array must be a two-dimensional array or nested sequence.\n\nNumber of dimensions: {array.ndim}"
                )
        elif isinstance(array, Sequence):
            if dtype is None:
                raise ValueError("You must specify a dtype when the input array is a sequence.")
        else:
            raise TypeError(
                f"The input array must be a two-dimensional array or nested sequence.\n\nType: {type(array)}"
            )

        return [
            AnalogWaveform(
                _data=np.asarray(array[i], dtype, copy=copy),
                start_index=start_index,
                sample_count=sample_count,
            )
            for i in range(len(array))
        ]

    __slots__ = ["_data", "_start_index", "_sample_count", "_extended_properties", "__weakref__"]

    _data: npt.NDArray[_ScalarType_co]
    _start_index: int
    _sample_count: int
    _extended_properties: ExtendedPropertyDictionary

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
        start_index = arg_to_uint("start index", start_index)
        sample_count = arg_to_uint("sample count", sample_count)
        capacity = arg_to_uint("capacity", capacity, sample_count)

        if dtype is None:
            dtype = np.float64

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
            raise TypeError("The input array must be a one-dimensional array.")
        if data.ndim != 1:
            raise ValueError("The input array must be a one-dimensional array.")

        if dtype is None:
            dtype = data.dtype
        elif dtype != data.dtype:
            raise ValueError(
                "The dtype of the input array must match the specified dtype.\n\n"
                f"Array dtype: {data.dtype}\n"
                f"Specified dtype: {dtype}"
            )

        capacity = arg_to_uint("capacity", capacity, len(data))
        if capacity != len(data):
            raise ValueError(
                "The capacity must match the input array length.\n\n"
                f"Capacity: {capacity}\n"
                f"Array length: {len(data)}"
            )

        start_index = arg_to_uint("start index", start_index)
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

    @property
    def raw_data(self) -> npt.NDArray[_ScalarType_co]:
        """The raw analog waveform data."""
        return self._data[self._start_index : self._start_index + self._sample_count]

    @property
    def scaled_data(self) -> npt.NDArray[np.float64]:
        """The scaled analog waveform data."""
        # TODO: implement scaling
        return self.raw_data.astype(np.float64)

    @property
    def sample_count(self) -> int:
        """The number of samples in the analog waveform."""
        return self._sample_count

    @property
    def capacity(self) -> int:
        """The total capacity available for analog waveform data."""
        return len(self._data)

    @capacity.setter
    def capacity(self, value: int) -> None:
        if value < 0:
            raise ValueError(
                "The capacity must be a non-negative integer.\n\n" f"Capacity: {value}"
            )
        if value < self._start_index + self._sample_count:
            raise ValueError(
                "The capacity must be equal to or greater than the number of samples in the waveform.\n\n"
                f"Capacity: {value}\n"
                f"Number of samples: {self._start_index + self._sample_count}"
            )
        if value != len(self._data):
            self._data.resize(value)

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
            raise TypeError("The channel name must be a str.\n\n" f"Channel name: {value!r}")
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
            raise TypeError(
                "The unit description must be a str.\n\n" f"Unit description: {value!r}"
            )
        self._extended_properties[UNIT_DESCRIPTION] = value
