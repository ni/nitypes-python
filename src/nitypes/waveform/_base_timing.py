from __future__ import annotations

import datetime as dt
import operator
from abc import ABC
from collections.abc import Generator, Iterable
from enum import Enum
from typing import Generic, SupportsIndex, TypeVar

# Note about NumPy type hints:
# - ht.datetime and ht.timedelta are subclasses of dt.datetime and dt.timedelta.
# - BaseWaveformTiming[ht.datetime, ht.timedelta] is a BaseWaveformTiming[dt.datetime, dt.timedelta]
#   due to covariance.
# - PrecisionWaveformTiming is not a subclass of WaveformTiming.


class WaveformSampleIntervalMode(Enum):
    """The sample interval mode that specifies how the waveform is sampled."""

    NONE = 0
    """No sample interval."""

    REGULAR = 1
    """Regular sample interval."""

    IRREGULAR = 2
    """Irregular sample interval."""


_TDateTime = TypeVar("_TDateTime", bound=dt.datetime)
_TDateTime_co = TypeVar("_TDateTime_co", bound=dt.datetime, covariant=True)

_TTimeDelta = TypeVar("_TTimeDelta", bound=dt.timedelta)
_TTimeDelta_co = TypeVar("_TTimeDelta_co", bound=dt.timedelta, covariant=True)


class BaseWaveformTiming(ABC, Generic[_TDateTime_co, _TTimeDelta_co]):
    """Base class for waveform timing information."""

    __slots__ = [
        "_timestamp",
        "_time_offset",
        "_sample_interval",
        "_sample_interval_mode",
        "_timestamps",
    ]

    _timestamp: _TDateTime_co | None
    _time_offset: _TTimeDelta_co
    _sample_interval: _TTimeDelta_co
    _sample_interval_mode: WaveformSampleIntervalMode
    _timestamps: list[_TDateTime_co] | None

    def __init__(
        self,
        timestamp: _TDateTime_co | None,
        time_offset: _TTimeDelta_co,
        sample_interval: _TTimeDelta_co,
        sample_interval_mode: WaveformSampleIntervalMode,
        timestamps: list[_TDateTime_co] | None,
    ) -> None:
        """Construct a base waveform timing object.

        This constructor is a private implementation detail.
        """
        self._timestamp = timestamp
        self._time_offset = time_offset
        self._sample_interval = sample_interval
        self._sample_interval_mode = sample_interval_mode
        self._timestamps = timestamps

    @property
    def has_timestamp(self) -> bool:
        """Indicates whether the waveform timing has a timestamp."""
        return self._timestamp is not None

    @property
    def timestamp(self) -> _TDateTime_co:
        """A timestamp representing the start of an acquisition or a related occurrence."""
        value = self._timestamp
        if value is None:
            raise RuntimeError("The waveform timing does not have a timestamp.")
        return value

    @property
    def start_time(self) -> _TDateTime_co:
        """The time that the first sample in the waveform was acquired."""
        return self.timestamp + self.time_offset

    @property
    def time_offset(self) -> _TTimeDelta_co:
        """The time difference between the timestamp and the first sample."""
        return self._time_offset

    @property
    def _has_sample_interval(self) -> bool:
        return self._sample_interval_mode == WaveformSampleIntervalMode.REGULAR

    @property
    def sample_interval(self) -> _TTimeDelta_co:
        """The time interval between samples."""
        if self._sample_interval_mode != WaveformSampleIntervalMode.REGULAR:
            raise RuntimeError("The waveform timing does not have a sample interval.")
        return self._sample_interval

    @property
    def sample_interval_mode(self) -> WaveformSampleIntervalMode:
        """The sample interval mode that specifies how the waveform is sampled."""
        return self._sample_interval_mode

    def get_timestamps(
        self, start_index: SupportsIndex, count: SupportsIndex
    ) -> Iterable[_TDateTime_co]:
        """Retrieve the timestamps of the waveform samples.

        Args:
            start_index: The sample index of the first timestamp to retrieve.
            count: The number of timestamps to retrieve.

        Returns:
            An iterable containing the requested timestamps.
        """
        start_index = operator.index(start_index)
        count = operator.index(count)

        if start_index < 0:
            raise ValueError("The sample index must be a non-negative integer.")
        if count < 0:
            raise ValueError("The count must be a non-negative integer.")

        if self._sample_interval_mode == WaveformSampleIntervalMode.REGULAR and self.has_timestamp:
            return self._generate_regular_timestamps(start_index, count)
        elif self._sample_interval_mode == WaveformSampleIntervalMode.IRREGULAR:
            assert self._timestamps is not None
            if count > len(self._timestamps):
                raise ValueError("The count must be less or equal to the number of timestamps.")
            return self._timestamps[start_index : start_index + count]
        else:
            raise RuntimeError(
                "The waveform timing does not have valid timestamp information. To obtain timestamps, the waveform must be irregular or must be initialized with a valid time stamp and sample interval."
            )

    def _generate_regular_timestamps(
        self, start_index: int, count: int
    ) -> Generator[_TDateTime_co]:
        sample_interval = self.sample_interval
        timestamp = self.start_time + start_index * sample_interval
        for i in range(count):
            if i != 0:
                timestamp += sample_interval
            yield timestamp

    def __eq__(self, value: object) -> bool:  # noqa: D105 - Missing docstring in magic method
        if not isinstance(value, BaseWaveformTiming):
            return NotImplemented
        return (
            self._timestamp == value._timestamp
            and self._time_offset == value._time_offset
            and self._sample_interval == value._sample_interval
            and self._sample_interval_mode == value._sample_interval_mode
            and self._timestamps == value._timestamps
        )
