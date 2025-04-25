from __future__ import annotations

import datetime as dt
import operator
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable, Sequence
from enum import Enum
from typing import Generic, SupportsIndex, TypeVar

from nitypes._arguments import validate_unsupported_arg
from nitypes._exceptions import add_note, invalid_arg_type
from nitypes._typing import Self


class SampleIntervalMode(Enum):
    """The sample interval mode that specifies how the waveform is sampled."""

    NONE = 0
    """No sample interval."""

    REGULAR = 1
    """Regular sample interval."""

    IRREGULAR = 2
    """Irregular sample interval."""


# TODO: should these be constrained types? I guess we'll find out when we add NI-BTF types.
_TDateTime = TypeVar("_TDateTime", bound=dt.datetime)
_TDateTime_co = TypeVar("_TDateTime_co", bound=dt.datetime, covariant=True)
_TTimeDelta_co = TypeVar("_TTimeDelta_co", bound=dt.timedelta, covariant=True)


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
        "_sample_interval_strategy",
        "_sample_interval_mode",
        "_timestamp",
        "_time_offset",
        "_sample_interval",
        "_timestamps",
        "__weakref__",
    ]

    _sample_interval_strategy: _SampleIntervalStrategy
    _sample_interval_mode: SampleIntervalMode
    _timestamp: _TDateTime_co | None
    _time_offset: _TTimeDelta_co | None
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
        sample_interval_strategy = _SAMPLE_INTERVAL_STRATEGY_FOR_MODE.get(sample_interval_mode)
        if sample_interval_strategy is None:
            raise ValueError(f"Unsupported sample interval mode {sample_interval_mode}.")

        try:
            sample_interval_strategy.validate_init_args(
                self, sample_interval_mode, timestamp, time_offset, sample_interval, timestamps
            )
        except (TypeError, ValueError) as e:
            add_note(e, f"Sample interval mode: {sample_interval_mode}")
            raise

        if timestamps is not None and not isinstance(timestamps, list):
            timestamps = list(timestamps)

        self._sample_interval_strategy = sample_interval_strategy
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
        value = self._time_offset
        if value is None:
            return self.__class__._get_default_time_offset()
        return value

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

        return self._sample_interval_strategy.get_timestamps(self, start_index, count)

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
        if self._time_offset is not None:
            args.append(f"time_offset={self._time_offset!r}")
        if self._sample_interval is not None:
            args.append(f"sample_interval={self._sample_interval!r}")
        if self._timestamps is not None:
            args.append(f"timestamps={self._timestamps!r}")
        return f"{self.__class__.__module__}.{self.__class__.__name__}({', '.join(args)})"

    def _append_timestamps(self, timestamps: Sequence[_TDateTime_co] | None) -> Self:
        new_timing = self._sample_interval_strategy.append_timestamps(self, timestamps)
        assert isinstance(new_timing, self.__class__)
        return new_timing

    def _append_timing(self, other: Self) -> Self:
        if not isinstance(other, self.__class__):
            raise TypeError(
                "The input waveform(s) must have the same waveform timing type as the current waveform."
            )

        is_irregular = self._sample_interval_mode == SampleIntervalMode.IRREGULAR
        other_is_irregular = other._sample_interval_mode == SampleIntervalMode.IRREGULAR
        if is_irregular != other_is_irregular:
            raise RuntimeError(
                "The timing of one or more waveforms does not match the timing of the current waveform."
            )

        new_timing = self._sample_interval_strategy.append_timing(self, other)
        assert isinstance(new_timing, self.__class__)
        return new_timing


def _are_timestamps_monotonic(timestamps: Sequence[_TDateTime_co]) -> bool:
    direction = 0
    for i in range(1, len(timestamps)):
        comparison = _get_direction(timestamps[i - 1], timestamps[i])
        if comparison == 0:
            continue

        if direction == 0:
            direction = comparison
        elif comparison != direction:
            return False
    return True


def _get_direction(left: _TDateTime, right: _TDateTime) -> int:
    if left < right:
        return -1
    if right < left:
        return 1
    return 0


class _SampleIntervalStrategy(ABC):
    @abstractmethod
    def validate_init_args(
        self,
        timing: BaseTiming[_TDateTime_co, _TTimeDelta_co],
        sample_interval_mode: SampleIntervalMode,
        timestamp: _TDateTime_co | None,
        time_offset: _TTimeDelta_co | None,
        sample_interval: _TTimeDelta_co | None,
        timestamps: Sequence[_TDateTime_co] | None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_timestamps(
        self, timing: BaseTiming[_TDateTime_co, _TTimeDelta_co], start_index: int, count: int
    ) -> Iterable[_TDateTime_co]:
        raise NotImplementedError

    @abstractmethod
    def append_timestamps(
        self,
        timing: BaseTiming[_TDateTime_co, _TTimeDelta_co],
        timestamps: Sequence[_TDateTime_co] | None,
    ) -> BaseTiming[_TDateTime_co, _TTimeDelta_co]:
        raise NotImplementedError

    @abstractmethod
    def append_timing(
        self,
        timing: BaseTiming[_TDateTime_co, _TTimeDelta_co],
        other: BaseTiming[_TDateTime_co, _TTimeDelta_co],
    ) -> BaseTiming[_TDateTime_co, _TTimeDelta_co]:
        raise NotImplementedError


class _NoneSampleIntervalStrategy(_SampleIntervalStrategy):
    def validate_init_args(
        self,
        timing: BaseTiming[_TDateTime_co, _TTimeDelta_co],
        sample_interval_mode: SampleIntervalMode,
        timestamp: _TDateTime_co | None,
        time_offset: _TTimeDelta_co | None,
        sample_interval: _TTimeDelta_co | None,
        timestamps: Sequence[_TDateTime_co] | None,
    ) -> None:
        datetime_type = timing.__class__._get_datetime_type()
        timedelta_type = timing.__class__._get_timedelta_type()
        if not isinstance(timestamp, (datetime_type, type(None))):
            raise invalid_arg_type("timestamp", "datetime or None", timestamp)
        if not isinstance(time_offset, (timedelta_type, type(None))):
            raise invalid_arg_type("time offset", "timedelta or None", time_offset)
        validate_unsupported_arg("sample interval", sample_interval)
        validate_unsupported_arg("timestamps", timestamps)

    def get_timestamps(
        self, timing: BaseTiming[_TDateTime_co, _TTimeDelta_co], start_index: int, count: int
    ) -> Iterable[_TDateTime_co]:
        raise _no_timestamp_information()

    def append_timestamps(
        self,
        timing: BaseTiming[_TDateTime_co, _TTimeDelta_co],
        timestamps: Sequence[_TDateTime_co] | None,
    ) -> BaseTiming[_TDateTime_co, _TTimeDelta_co]:
        try:
            validate_unsupported_arg("timestamps", timestamps)
        except (TypeError, ValueError) as e:
            add_note(e, f"Sample interval mode: {timing.sample_interval_mode}")
            raise
        return timing

    def append_timing(
        self,
        timing: BaseTiming[_TDateTime_co, _TTimeDelta_co],
        other: BaseTiming[_TDateTime_co, _TTimeDelta_co],
    ) -> BaseTiming[_TDateTime_co, _TTimeDelta_co]:
        return timing


class _RegularSampleIntervalStrategy(_SampleIntervalStrategy):
    def validate_init_args(
        self,
        timing: BaseTiming[_TDateTime_co, _TTimeDelta_co],
        sample_interval_mode: SampleIntervalMode,
        timestamp: _TDateTime_co | None,
        time_offset: _TTimeDelta_co | None,
        sample_interval: _TTimeDelta_co | None,
        timestamps: Sequence[_TDateTime_co] | None,
    ) -> None:
        datetime_type = timing.__class__._get_datetime_type()
        timedelta_type = timing.__class__._get_timedelta_type()
        if not isinstance(timestamp, (datetime_type, type(None))):
            raise invalid_arg_type("timestamp", "datetime or None", timestamp)
        if not isinstance(time_offset, (timedelta_type, type(None))):
            raise invalid_arg_type("time offset", "timedelta or None", time_offset)
        if not isinstance(sample_interval, timedelta_type):
            raise invalid_arg_type("sample interval", "timedelta", sample_interval)
        validate_unsupported_arg("timestamps", timestamps)

    def get_timestamps(
        self, timing: BaseTiming[_TDateTime_co, _TTimeDelta_co], start_index: int, count: int
    ) -> Iterable[_TDateTime_co]:
        if timing.has_timestamp:
            return self._generate_regular_timestamps(timing, start_index, count)
        raise _no_timestamp_information()

    def _generate_regular_timestamps(
        self, timing: BaseTiming[_TDateTime_co, _TTimeDelta_co], start_index: int, count: int
    ) -> Generator[_TDateTime_co]:
        sample_interval = timing.sample_interval
        timestamp = timing.start_time + start_index * sample_interval
        for i in range(count):
            if i != 0:
                timestamp += sample_interval
            yield timestamp

    def append_timestamps(
        self,
        timing: BaseTiming[_TDateTime_co, _TTimeDelta_co],
        timestamps: Sequence[_TDateTime_co] | None,
    ) -> BaseTiming[_TDateTime_co, _TTimeDelta_co]:
        try:
            validate_unsupported_arg("timestamps", timestamps)
        except (TypeError, ValueError) as e:
            add_note(e, f"Sample interval mode: {timing.sample_interval_mode}")
            raise
        return timing

    def append_timing(
        self,
        timing: BaseTiming[_TDateTime_co, _TTimeDelta_co],
        other: BaseTiming[_TDateTime_co, _TTimeDelta_co],
    ) -> BaseTiming[_TDateTime_co, _TTimeDelta_co]:
        return timing


class _IrregularSampleIntervalStrategy(_SampleIntervalStrategy):
    def validate_init_args(
        self,
        timing: BaseTiming[_TDateTime_co, _TTimeDelta_co],
        sample_interval_mode: SampleIntervalMode,
        timestamp: _TDateTime_co | None,
        time_offset: _TTimeDelta_co | None,
        sample_interval: _TTimeDelta_co | None,
        timestamps: Sequence[_TDateTime_co] | None,
    ) -> None:
        datetime_type = timing.__class__._get_datetime_type()
        validate_unsupported_arg("timestamp", timestamp)
        validate_unsupported_arg("time offset", time_offset)
        validate_unsupported_arg("sample interval", sample_interval)
        if not isinstance(timestamps, Sequence) or not all(
            isinstance(ts, datetime_type) for ts in timestamps
        ):
            raise invalid_arg_type("timestamps", "sequence of datetime objects", timestamps)
        if not _are_timestamps_monotonic(timestamps):
            raise ValueError("The timestamps must be in ascending or descending order.")

    def get_timestamps(
        self, timing: BaseTiming[_TDateTime_co, _TTimeDelta_co], start_index: int, count: int
    ) -> Iterable[_TDateTime_co]:
        assert timing._timestamps is not None
        if count > len(timing._timestamps):
            raise ValueError("The count must be less than or equal to the number of timestamps.")
        return timing._timestamps[start_index : start_index + count]

    def append_timestamps(
        self,
        timing: BaseTiming[_TDateTime_co, _TTimeDelta_co],
        timestamps: Sequence[_TDateTime_co] | None,
    ) -> BaseTiming[_TDateTime_co, _TTimeDelta_co]:
        assert timing._timestamps is not None

        if timestamps is None:
            raise RuntimeError(
                "The timestamps argument is required when appending to a waveform with irregular timing."
            )

        datetime_type = timing.__class__._get_datetime_type()
        if not all(isinstance(ts, datetime_type) for ts in timestamps):
            raise TypeError(
                "The timestamp data type must match the timing information of the current waveform."
            )

        if len(timestamps) == 0:
            return timing
        else:
            if not isinstance(timestamps, list):
                timestamps = list(timestamps)

            return timing.__class__(
                SampleIntervalMode.IRREGULAR,
                None,
                None,
                None,
                timing._timestamps + timestamps,
            )

    def append_timing(
        self,
        timing: BaseTiming[_TDateTime_co, _TTimeDelta_co],
        other: BaseTiming[_TDateTime_co, _TTimeDelta_co],
    ) -> BaseTiming[_TDateTime_co, _TTimeDelta_co]:
        assert timing._timestamps is not None and other._timestamps is not None

        if len(timing._timestamps) == 0:
            return other
        elif len(other._timestamps) == 0:
            return timing
        else:
            # The constructor will verify that the combined list of timestamps is monotonic. This is
            # not optimal for a large number of appends.
            return timing.__class__(
                SampleIntervalMode.IRREGULAR,
                None,
                None,
                None,
                timing._timestamps + other._timestamps,
            )


_SAMPLE_INTERVAL_STRATEGY_FOR_MODE: dict[SampleIntervalMode, _SampleIntervalStrategy] = {
    SampleIntervalMode.NONE: _NoneSampleIntervalStrategy(),
    SampleIntervalMode.REGULAR: _RegularSampleIntervalStrategy(),
    SampleIntervalMode.IRREGULAR: _IrregularSampleIntervalStrategy(),
}


def _no_timestamp_information() -> RuntimeError:
    return RuntimeError(
        "The waveform timing does not have valid timestamp information. "
        "To obtain timestamps, the waveform must be irregular or must be initialized "
        "with a valid time stamp and sample interval."
    )
