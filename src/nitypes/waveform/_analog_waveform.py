from __future__ import annotations

import operator
import sys
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Generic, SupportsIndex, overload

import numpy as np
import numpy.typing as npt

from nitypes.waveform._extended_properties import (
    CHANNEL_NAME,
    UNIT_DESCRIPTION,
    ExtendedPropertyDictionary,
)

# Don't use typing_extensions at run time.
if not TYPE_CHECKING or sys.version_info >= (3, 13):
    from typing import TypeVar
else:
    from typing_extensions import TypeVar

_DType = TypeVar("_DType", bound=np.dtype[Any])
if TYPE_CHECKING or sys.version_info >= (3, 13):
    # PEP 696 – Type Defaults for Type Parameters
    _ScalarType = TypeVar("_ScalarType", bound=np.generic, default=np.float64)
    _ScalarType_co = TypeVar("_ScalarType_co", bound=np.generic, covariant=True, default=np.float64)
else:
    _ScalarType = TypeVar("_ScalarType", bound=np.generic)
    _ScalarType_co = TypeVar("_ScalarType_co", bound=np.generic, covariant=True)


class AnalogWaveform(Generic[_ScalarType_co]):
    """An analog waveform, which encapsulates analog data and timing information.

    To construct an analog waveform, use one of the static methods:
    - create
    - from_iter
    - from_ndarray
    """

    @overload
    @staticmethod
    def create(
        sample_count: SupportsIndex = ...,
        dtype: None = ...,
        capacity: SupportsIndex = ...,
    ) -> AnalogWaveform[np.float64]: ...

    @overload
    @staticmethod
    def create(
        sample_count: SupportsIndex = ...,
        dtype: type[_ScalarType] | np.dtype[_ScalarType] = ...,
        capacity: SupportsIndex = ...,
    ) -> AnalogWaveform[_ScalarType]: ...

    @overload
    @staticmethod
    def create(
        sample_count: SupportsIndex = ...,
        dtype: npt.DTypeLike = ...,
        capacity: SupportsIndex = ...,
    ) -> AnalogWaveform[Any]: ...

    @staticmethod
    def create(
        sample_count: SupportsIndex = 0,
        dtype: npt.DTypeLike = np.float64,
        capacity: SupportsIndex = -1,
    ) -> AnalogWaveform[_ScalarType]:
        """Construct an analog waveform.

        Args:
            sample_count: The number of samples in the analog waveform.
            dtype: The NumPy data type for the analog waveform data.
            capacity: The total capacity of the analog waveform.

        Returns:
            An analog waveform with the specified sample count, data type, and capacity.
        """
        if capacity == -1:
            capacity = sample_count
        return AnalogWaveform(np.zeros(capacity, dtype), sample_count=sample_count)

    @overload
    @staticmethod
    def from_iter(
        iter: Iterable[Any],
        dtype: None = ...,
        sample_count: SupportsIndex = ...,
    ) -> AnalogWaveform[np.float64]: ...

    @overload
    @staticmethod
    def from_iter(
        iter: Iterable[Any],
        dtype: type[_ScalarType] | np.dtype[_ScalarType] = ...,
        sample_count: SupportsIndex = ...,
    ) -> AnalogWaveform[_ScalarType]: ...

    @overload
    @staticmethod
    def from_iter(
        iter: Iterable[Any],
        dtype: npt.DTypeLike = ...,
        sample_count: SupportsIndex = ...,
    ) -> AnalogWaveform[Any]: ...

    @staticmethod
    def from_iter(
        iter: Iterable[Any],
        dtype: npt.DTypeLike = np.float64,
        sample_count: SupportsIndex = -1,
    ) -> AnalogWaveform[_ScalarType]:
        """Construct an analog waveform from an iterable object.

        Args:
            iter: The iterable object containing the analog waveform data.
            dtype: The NumPy data type for the analog waveform data.
            sample_count: The number of items to read from the iterable. By default, all items
                are read.

        Returns:
            An analog waveform containing a copy of the data from the iterable object.
        """
        return AnalogWaveform(np.fromiter(iter, dtype, sample_count))

    @staticmethod
    def from_ndarray(
        array: npt.NDArray[_ScalarType],
        *,
        copy: bool = True,
        start_index: SupportsIndex = 0,
        sample_count: SupportsIndex = -1,
    ) -> AnalogWaveform[_ScalarType]:
        """Construct an analog waveform from a NumPy array.

        Args:
            array: The Numpy array containing the analog waveform data.
            copy: Specifies whether to copy the array or reference the passed-in array.
            start_index: The array index at which the analog waveform data begins.
            sample_count: The number of samples in the analog waveform.

        Returns:
            An analog waveform containing a copy or reference to the NumPy array.
        """
        if copy:
            array = array.copy()
        return AnalogWaveform(array, start_index=start_index, sample_count=sample_count)

    __slots__ = ["_data", "_start_index", "_sample_count", "_extended_properties", "__weakref__"]

    def __init__(  # noqa: D107 - Missing docstring in __init__ (auto-generated noqa)
        self,
        data: npt.NDArray[_ScalarType_co],
        *,
        start_index: SupportsIndex = 0,
        sample_count: SupportsIndex = -1,
    ) -> None:
        start_index = operator.index(start_index)
        sample_count = operator.index(sample_count)

        # TODO: support negative index?
        if start_index < 0:
            raise ValueError(f"Start index {start_index} is less than zero.")
        elif start_index > len(data):
            raise ValueError(f"Start index {start_index} is greater than array length {len(data)}.")

        if sample_count == -1:
            sample_count = len(data) - start_index
        elif sample_count < 0:
            raise ValueError(f"Sample count {sample_count} is less than zero.")
        elif sample_count > len(data):
            raise ValueError(
                f"Sample count {sample_count} is greater than array length {len(data)}."
            )

        if start_index + sample_count > len(data):
            raise ValueError(
                "The capacity must be equal to or greater than the number of samples in the waveform."
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
            raise ValueError(f"Capacity {value} is less than zero.")
        if value < self._start_index + self._sample_count:
            raise ValueError(
                "The capacity must be equal to or greater than the number of samples in the waveform."
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
            raise TypeError(f"Channel name {value} is not a str.")
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
            raise TypeError(f"Unit description {value} is not a str.")
        self._extended_properties[UNIT_DESCRIPTION] = value
