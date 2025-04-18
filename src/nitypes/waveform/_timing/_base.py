from __future__ import annotations

import datetime as dt
import operator
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable, Sequence
from enum import Enum
from typing import Generic, SupportsIndex, TypeVar

from nitypes.waveform._utils import add_note


class SampleIntervalMode(Enum):
    """The sample interval mode that specifies how the waveform is sampled."""

    NONE = 0
    """No sample interval."""

    REGULAR = 1
    """Regular sample interval."""

    IRREGULAR = 2
    """Irregular sample interval."""


# TODO: should these be constrained types? I guess we'll find out when we add NI-BTF types.
_TDateTime_co = TypeVar("_TDateTime_co", bound=dt.datetime)
_TTimeDelta_co = TypeVar("_TTimeDelta_co", bound=dt.timedelta)


class BaseTiming(ABC, Generic[_TDateTime_co, _TTimeDelta_co]):
    """Base class for waveform timing information."""

    @staticmethod
    @abstractmethod
    def _get_datetime_type() -> type[_TDateTime_co]:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def _get_timedelta_type() -> type[_TTimeDelta_co]:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def _get_default_time_offset() -> _TTimeDelta_co:
        raise NotImplementedError()

    __slots__ = [
        "_sample_interval_mode",
        "_timestamp",
        "_time_offset",
        "_sample_interval",
        "_timestamps",
    ]

    _sample_interval_mode: SampleIntervalMode
    _timestamp: _TDateTime_co | None
    _time_offset: _TTimeDelta_co
    _sample_interval: _TTimeDelta_co | None
    _timestamps: list[_TDateTime_co] | None

    def __init__(
        self,
        sample_interval_mode: SampleIntervalMode,
        timestamp: _TDateTime_co | None,
        time_offset: _TTimeDelta_co | None,
        sample_interval: _TTimeDelta_co | None,
        timestamps: Sequence[_TDateTime_co] | None,
    ) -> None:
        """Construct a base waveform timing object."""
        datetime_type = self.__class__._get_datetime_type()
        timedelta_type = self.__class__._get_timedelta_type()

        try:
            if sample_interval_mode == SampleIntervalMode.NONE:
                if not isinstance(timestamp, (datetime_type, type(None))):
                    raise TypeError(
                        "The timestamp must be a datetime or None.\n" f"Provided value: {timestamp}"
                    )
                if not isinstance(time_offset, (timedelta_type, type(None))):
                    raise TypeError(
                        f"The time offset must be a timedelta or None."
                        f"Provided value: {time_offset}"
                    )
                if sample_interval is not None:
                    raise ValueError(
                        "The sample interval argument is not supported."
                        f"Provided value: {sample_interval}"
                    )
                if timestamps is not None:
                    raise ValueError(
                        "The timestamps argument is not supported." f"Provided value: {timestamps}"
                    )
            elif sample_interval_mode == SampleIntervalMode.REGULAR:
                if not isinstance(timestamp, (datetime_type, type(None))):
                    raise TypeError(
                        "The timestamp must be a datetime or None.\n" f"Provided value: {timestamp}"
                    )
                if not isinstance(time_offset, (timedelta_type, type(None))):
                    raise TypeError(
                        f"The time offset must be a timedelta or None."
                        f"Provided value: {time_offset}"
                    )
                if not isinstance(sample_interval, timedelta_type):
                    raise ValueError(
                        "The sample interval must be a timedelta."
                        f"Provided value: {sample_interval}"
                    )
                if timestamps is not None:
                    raise ValueError(
                        "The timestamps argument is not supported." f"Provided value: {timestamps}"
                    )
            elif sample_interval_mode == SampleIntervalMode.IRREGULAR:
                if timestamp is not None:
                    raise TypeError(
                        "The timestamp argument is not supported.\n" f"Provided value: {timestamp}"
                    )
                if time_offset is not None:
                    raise TypeError(
                        f"The time offset argument is not supported."
                        f"Provided value: {time_offset}"
                    )
                if sample_interval is not None:
                    raise ValueError(
                        "The sample interval argument is not supported."
                        f"Provided value: {sample_interval}"
                    )
                if not isinstance(timestamps, Sequence) or not all(
                    isinstance(ts, datetime_type) for ts in timestamps
                ):
                    raise ValueError(
                        "The timestamps must be a sequence of datetime objects."
                        f"Provided value: {timestamps}"
                    )
            else:
                raise ValueError(f"Unsupported sample interval mode {sample_interval_mode}.")
        except (TypeError, ValueError) as e:
            add_note(e, f"Sample interval mode: {sample_interval_mode}")
            raise

        if time_offset is None:
            time_offset = self.__class__._get_default_time_offset()

        if timestamps is not None and not isinstance(timestamps, list):
            timestamps = list(timestamps)

        self._sample_interval_mode = sample_interval_mode
        self._timestamp = timestamp
        self._time_offset = time_offset
        self._sample_interval = sample_interval
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
    def sample_interval(self) -> _TTimeDelta_co:
        """The time interval between samples."""
        value = self._sample_interval
        if value is None:
            raise RuntimeError("The waveform timing does not have a sample interval.")
        return value

    @property
    def sample_interval_mode(self) -> SampleIntervalMode:
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

        if self._sample_interval_mode == SampleIntervalMode.REGULAR and self.has_timestamp:
            return self._generate_regular_timestamps(start_index, count)
        elif self._sample_interval_mode == SampleIntervalMode.IRREGULAR:
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
        if not isinstance(value, self.__class__):
            return NotImplemented
        return (
            self._timestamp == value._timestamp
            and self._time_offset == value._time_offset
            and self._sample_interval == value._sample_interval
            and self._sample_interval_mode == value._sample_interval_mode
            and self._timestamps == value._timestamps
        )

    def __repr__(self) -> str:  # noqa: D105 - Missing docstring in magic method
        # For Enum, __str__ is an unqualified ctor expression like E.V and __repr__ is <E.V: 0>.
        args = [f"{self.sample_interval_mode.__class__.__module__}.{self.sample_interval_mode}"]
        if self._timestamp is not None:
            args.append(f"timestamp={self._timestamp!r}")
        if self._time_offset != self.__class__._get_default_time_offset():
            args.append(f"time_offset={self._time_offset!r}")
        if self._sample_interval is not None:
            args.append(f"sample_interval={self.sample_interval!r}")
        if self._timestamps is not None:
            args.append(f"timestamps={self._timestamps!r}")
        return f"{self.__class__.__module__}.{self.__class__.__name__}({', '.join(args)})"
