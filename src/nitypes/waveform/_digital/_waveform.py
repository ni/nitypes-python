from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from typing import Any, Generic, SupportsIndex, TypeVar, Union, overload

import hightime as ht
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from nitypes._arguments import arg_to_uint, validate_dtype, validate_unsupported_arg
from nitypes._exceptions import invalid_arg_type, invalid_array_ndim
from nitypes.waveform._digital._state import DigitalState
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
    ExtendedPropertyDictionary,
    ExtendedPropertyValue,
)
from nitypes.waveform._timing import Timing, _AnyDateTime, _AnyTimeDelta

_TData = TypeVar("_TData", bound=Union[np.bool, np.uint8])
_TOtherData = TypeVar("_TOtherData", bound=Union[np.bool, np.uint8])

_DIGITAL_DTYPES = (np.bool, np.uint8)


class DigitalWaveform(Generic[_TData]):
    """A digital waveform, which encapsulates digital data and timing information."""

    __slots__ = [
        "_data",
        "_start_index",
        "_sample_count",
        "_extended_properties",
        "_timing",
        "__weakref__",
    ]

    _data: np.ndarray[tuple[int, int], np.dtype[_TData]]
    _start_index: int
    _sample_count: int
    _extended_properties: ExtendedPropertyDictionary
    _timing: Timing[_AnyDateTime, _AnyTimeDelta, _AnyTimeDelta]

    # If neither dtype nor data is specified, _TData defaults to np.bool.
    @overload
    def __init__(  # noqa: D107 - Missing docstring in __init__ (auto-generated noqa)
        self: DigitalWaveform[np.bool],
        sample_count: SupportsIndex | None = ...,
        signal_count: SupportsIndex | None = ...,
        dtype: None = ...,
        default_value: DigitalState | None = ...,
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
        self: DigitalWaveform[_TOtherData],
        sample_count: SupportsIndex | None = ...,
        signal_count: SupportsIndex | None = ...,
        dtype: type[_TOtherData] | np.dtype[_TOtherData] = ...,
        default_value: DigitalState | None = ...,
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
        self: DigitalWaveform[_TOtherData],
        sample_count: SupportsIndex | None = ...,
        signal_count: SupportsIndex | None = ...,
        dtype: None = ...,
        default_value: DigitalState | None = ...,
        *,
        data: npt.NDArray[_TOtherData] = ...,
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
        default_value: DigitalState | None = ...,
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
        default_value: DigitalState | None = None,
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

    def _init_with_new_array(
        self,
        sample_count: SupportsIndex | None = None,
        signal_count: SupportsIndex | None = None,
        dtype: npt.DTypeLike = None,
        default_value: DigitalState | None = None,
        *,
        start_index: SupportsIndex | None = None,
        capacity: SupportsIndex | None = None,
    ) -> None:
        start_index = arg_to_uint("start index", start_index, 0)
        sample_count = arg_to_uint("sample count", sample_count, 0)
        signal_count = arg_to_uint("signal count", signal_count, 1)
        capacity = arg_to_uint("capacity", capacity, sample_count)

        if dtype is None:
            dtype = np.bool
        validate_dtype(dtype, _DIGITAL_DTYPES)

        if start_index > capacity:
            raise start_index_too_large(start_index, "capacity", capacity)
        if start_index + sample_count > capacity:
            raise start_index_or_sample_count_too_large(
                start_index, sample_count, "capacity", capacity
            )

        if default_value is None:
            default_value = DigitalState.FORCE_DOWN

        self._data = np.full((capacity, signal_count), default_value, dtype)
        self._start_index = start_index
        self._sample_count = sample_count

    def _init_with_provided_array(
        self,
        data: npt.NDArray[_TData],
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
        validate_dtype(dtype, _DIGITAL_DTYPES)

        if data.ndim == 1:
            data_signal_count = 1
            data = data.reshape(len(data), 1)
        elif data.ndim == 2:
            data_signal_count = data.shape[1]
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

        signal_count = arg_to_uint("signal count", signal_count, 1)
        if signal_count != data_signal_count:
            raise signal_count_mismatch("provided", signal_count, "array", data_signal_count)

        self._data = data
        self._start_index = start_index
        self._sample_count = sample_count

    @property
    def data(self) -> np.ndarray[tuple[int, int], np.dtype[_TData]]:
        """The waveform data, indexed by (sample, signal)."""
        return self._data[self._start_index : self._start_index + self._sample_count]

    def get_data(
        self, start_index: SupportsIndex | None = 0, sample_count: SupportsIndex | None = None
    ) -> np.ndarray[tuple[int, int], np.dtype[_TData]]:
        """Get a subset of the waveform data.

        Args:
            start_index: The sample index at which the data begins.
            sample_count: The number of samples to return.

        Returns:
            A subset of the raw waveform data.
        """
        start_index = arg_to_uint("sample index", start_index, 0)
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
        return self._data.shape[1]

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
            self._data.resize(value, refcheck=False)

    @property
    def dtype(self) -> np.dtype[_TData]:
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
        other: npt.NDArray[_TData] | DigitalWaveform[_TData] | Sequence[DigitalWaveform[_TData]],
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
        array: npt.NDArray[_TData],
        timestamps: Sequence[dt.datetime] | Sequence[ht.datetime] | None = None,
    ) -> None:
        if array.dtype != self.dtype:
            raise data_type_mismatch("input array", array.dtype, "spectrum", self.dtype)

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

    def _append_waveform(self, waveform: DigitalWaveform[_TData]) -> None:
        self._append_waveforms([waveform])

    def _append_waveforms(self, waveforms: Sequence[DigitalWaveform[_TData]]) -> None:
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
        array: npt.NDArray[_TData],
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
        array: npt.NDArray[_TData],
        *,
        copy: bool = True,
        start_index: SupportsIndex | None = 0,
        sample_count: SupportsIndex | None = None,
        signal_count: SupportsIndex | None = None,
    ) -> None:
        if array.dtype != self.dtype:
            raise data_type_mismatch("input array", array.dtype, "spectrum", self.dtype)

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
        ctor_args = (self._sample_count, self.dtype)
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
        args = [f"{self._sample_count}"]
        # start_index and capacity are not shown because they are allocation details. data hides
        # the unused data before start_index and after start_index+sample_count.
        if self._sample_count > 0:
            args.append(f"data={self.data!r}")
        if self._extended_properties:
            args.append(f"extended_properties={self._extended_properties._properties!r}")
        if self._timing is not Timing.empty:
            args.append(f"timing={self._timing!r}")
        return f"{self.__class__.__module__}.{self.__class__.__name__}({', '.join(args)})"
