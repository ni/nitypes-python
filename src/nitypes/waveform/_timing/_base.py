from __future__ import annotations

import datetime as dt
import operator
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, Generic, SupportsIndex, TypeVar

from typing_extensions import Self

from nitypes._exceptions import add_note
from nitypes.waveform._timing._sample_interval import (
    SampleIntervalMode,
    SampleIntervalStrategy,
    create_sample_interval_strategy,
)

_TDateTime = TypeVar("_TDateTime", bound=dt.datetime)
_TTimeDelta = TypeVar("_TTimeDelta", bound=dt.timedelta)


class BaseTiming(ABC, Generic[_TDateTime, _TTimeDelta]):
    """Base class for waveform timing information.

    Waveform timing objects are immutable.
    """

    @classmethod
    @abstractmethod
    def create_with_no_interval(
        cls, timestamp: _TDateTime | None = None, time_offset: _TTimeDelta | None = None
    ) -> Self:
        """Create a waveform timing object with no sample interval.

        Args:
            timestamp: A timestamp representing the start of an acquisition or a related
                occurrence.
            time_offset: The time difference between the timestamp and the time that the first
                sample was acquired.

        Returns:
            A waveform timing object.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def create_with_regular_interval(
        cls,
        sample_interval: _TTimeDelta,
        timestamp: _TDateTime | None = None,
        time_offset: _TTimeDelta | None = None,
    ) -> Self:
        """Create a waveform timing object with a regular sample interval.

        Args:
            sample_interval: The time difference between samples.
            timestamp: A timestamp representing the start of an acquisition or a related
                occurrence.
            time_offset: The time difference between the timestamp and the time that the first
                sample was acquired.

        Returns:
            A waveform timing object.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def create_with_irregular_interval(
        cls,
        timestamps: Sequence[_TDateTime],
    ) -> Self:
        """Create a waveform timing object with an irregular sample interval.

        Args:
            timestamps: A sequence containing a timestamp for each sample in the waveform,
                specifying the time that the sample was acquired.

        Returns:
            A waveform timing object.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _get_datetime_type() -> type[_TDateTime]:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def _get_timedelta_type() -> type[_TTimeDelta]:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def _get_default_time_offset() -> _TTimeDelta:
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

    _sample_interval_strategy: SampleIntervalStrategy[_TDateTime, _TTimeDelta]
    _sample_interval_mode: SampleIntervalMode
    _timestamp: _TDateTime | None
    _time_offset: _TTimeDelta | None
    _sample_interval: _TTimeDelta | None
    _timestamps: list[_TDateTime] | None

    def __init__(
        self,
        sample_interval_mode: SampleIntervalMode,
        timestamp: _TDateTime | None,
        time_offset: _TTimeDelta | None,
        sample_interval: _TTimeDelta | None,
        timestamps: Sequence[_TDateTime] | None,
        *,
        copy_timestamps: bool = True,
    ) -> None:
        """Initialize a new waveform timing object.

        Args:
            sample_interval_mode: The sample interval mode of the waveform timing.
            timestamp: The timestamp of the waveform timing. This argument is optional for
                SampleIntervalMode.NONE and SampleIntervalMode.REGULAR and unsupported for
                SampleIntervalMode.IRREGULAR.
            time_offset: The time difference between the timestamp and the first sample. This
                argument is optional for SampleIntervalMode.NONE and SampleIntervalMode.REGULAR and
                unsupported for SampleIntervalMode.IRREGULAR.
            sample_interval: The time interval between samples. This argument is required for
                SampleIntervalMode.REGULAR and unsupported otherwise.
            timestamps: A sequence containing a timestamp for each sample in the waveform,
                specifying the time that the sample was acquired. This argument is required for
                SampleIntervalMode.IRREGULAR and unsupported otherwise.
            copy_timestamps: Specifies whether to copy the timestamps or take ownership.
        """
        sample_interval_strategy = create_sample_interval_strategy(sample_interval_mode)
        try:
            sample_interval_strategy.validate_init_args(
                self, sample_interval_mode, timestamp, time_offset, sample_interval, timestamps
            )
        except (TypeError, ValueError) as e:
            add_note(e, f"Sample interval mode: {sample_interval_mode}")
            raise

        if timestamps is not None and (copy_timestamps or not isinstance(timestamps, list)):
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
    def timestamp(self) -> _TDateTime:
        """A timestamp representing the start of an acquisition or a related occurrence."""
        value = self._timestamp
        if value is None:
            raise RuntimeError("The waveform timing does not have a timestamp.")
        return value

    @property
    def start_time(self) -> _TDateTime:
        """The time that the first sample in the waveform was acquired."""
        return self.timestamp + self.time_offset

    @property
    def time_offset(self) -> _TTimeDelta:
        """The time difference between the timestamp and the first sample."""
        value = self._time_offset
        if value is None:
            return self.__class__._get_default_time_offset()
        return value

    @property
    def sample_interval(self) -> _TTimeDelta:
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
    ) -> Iterable[_TDateTime]:
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

    def __eq__(self, value: object, /) -> bool:
        """Return self==value."""
        if not isinstance(value, self.__class__):
            return NotImplemented
        return (
            self._timestamp == value._timestamp
            and self._time_offset == value._time_offset
            and self._sample_interval == value._sample_interval
            and self._sample_interval_mode == value._sample_interval_mode
            and self._timestamps == value._timestamps
        )

    def __reduce__(self) -> tuple[Any, ...]:
        """Return object state for pickling."""
        ctor_args = (
            self._sample_interval_mode,
            self._timestamp,
            self._time_offset,
            self._sample_interval,
            self._timestamps,
        )
        ctor_kwargs: dict[str, Any] = {}
        if self._timestamps is not None:
            ctor_kwargs["copy_timestamps"] = False
        return (self.__class__._unpickle, (ctor_args, ctor_kwargs))

    @classmethod
    def _unpickle(cls, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Self:
        return cls(*args, **kwargs)

    def __repr__(self) -> str:
        """Return repr(self)."""
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

    def _append_timestamps(self, timestamps: Sequence[_TDateTime] | None) -> Self:
        new_timing = self._sample_interval_strategy.append_timestamps(self, timestamps)
        assert isinstance(new_timing, self.__class__)
        return new_timing

    def _append_timing(self, other: Self) -> Self:
        if not isinstance(other, self.__class__):
            raise TypeError(
                "The input waveform(s) must have the same waveform timing type as the current waveform."
            )

        new_timing = self._sample_interval_strategy.append_timing(self, other)
        assert isinstance(new_timing, self.__class__)
        return new_timing
