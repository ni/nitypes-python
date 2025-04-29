from __future__ import annotations

import datetime as dt
import warnings
from collections.abc import Generator, Iterable, Sequence
from typing import TYPE_CHECKING, TypeVar

from nitypes._arguments import validate_unsupported_arg
from nitypes._exceptions import add_note, invalid_arg_type
from nitypes.waveform._exceptions import (
    no_timestamp_information,
    sample_interval_mode_mismatch,
)
from nitypes.waveform._timing._sample_interval._base import SampleIntervalStrategy
from nitypes.waveform._timing._sample_interval._mode import SampleIntervalMode
from nitypes.waveform._warnings import sample_interval_mismatch

if TYPE_CHECKING:
    from nitypes.waveform._timing._base import BaseTiming  # circular import

_TDateTime = TypeVar("_TDateTime", bound=dt.datetime)
_TTimeDelta = TypeVar("_TTimeDelta", bound=dt.timedelta)


class RegularSampleIntervalStrategy(SampleIntervalStrategy[_TDateTime, _TTimeDelta]):
    """Implements SampleIntervalMode.REGULAR specific behavior."""

    def validate_init_args(  # noqa: D102 - Missing docstring in public method - override
        self,
        timing: BaseTiming[_TDateTime, _TTimeDelta],
        sample_interval_mode: SampleIntervalMode,
        timestamp: _TDateTime | None,
        time_offset: _TTimeDelta | None,
        sample_interval: _TTimeDelta | None,
        timestamps: Sequence[_TDateTime] | None,
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

    def get_timestamps(  # noqa: D102 - Missing docstring in public method - override
        self, timing: BaseTiming[_TDateTime, _TTimeDelta], start_index: int, count: int
    ) -> Iterable[_TDateTime]:
        if timing.has_timestamp:
            return self._generate_regular_timestamps(timing, start_index, count)
        raise no_timestamp_information()

    def _generate_regular_timestamps(
        self, timing: BaseTiming[_TDateTime, _TTimeDelta], start_index: int, count: int
    ) -> Generator[_TDateTime]:
        sample_interval = timing.sample_interval
        timestamp = timing.start_time + start_index * sample_interval
        for i in range(count):
            if i != 0:
                timestamp += sample_interval
            yield timestamp

    def append_timestamps(  # noqa: D102 - Missing docstring in public method - override
        self,
        timing: BaseTiming[_TDateTime, _TTimeDelta],
        timestamps: Sequence[_TDateTime] | None,
    ) -> BaseTiming[_TDateTime, _TTimeDelta]:
        try:
            validate_unsupported_arg("timestamps", timestamps)
        except (TypeError, ValueError) as e:
            add_note(e, f"Sample interval mode: {timing.sample_interval_mode}")
            raise
        return timing

    def append_timing(  # noqa: D102 - Missing docstring in public method - override
        self,
        timing: BaseTiming[_TDateTime, _TTimeDelta],
        other: BaseTiming[_TDateTime, _TTimeDelta],
    ) -> BaseTiming[_TDateTime, _TTimeDelta]:
        if other._sample_interval_mode not in (SampleIntervalMode.NONE, SampleIntervalMode.REGULAR):
            raise sample_interval_mode_mismatch()
        if timing._sample_interval != other._sample_interval:
            warnings.warn(sample_interval_mismatch())
        return timing
