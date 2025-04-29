from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence
from enum import Enum
from typing import TYPE_CHECKING, TypeVar

from nitypes._arguments import validate_unsupported_arg
from nitypes._exceptions import invalid_arg_type
from nitypes.waveform._exceptions import (
    TimingMismatchError,
    sample_interval_mode_mismatch,
)
from nitypes.waveform._timing._sample_interval._base import SampleIntervalStrategy
from nitypes.waveform._timing._sample_interval._mode import SampleIntervalMode

if TYPE_CHECKING:
    from nitypes.waveform._timing._base import BaseTiming  # circular import

_TDateTime = TypeVar("_TDateTime", bound=dt.datetime)
_TTimeDelta = TypeVar("_TTimeDelta", bound=dt.timedelta)


class _Direction(Enum):
    INCREASING = -1
    UNKNOWN = 0
    DECREASING = 1


def _are_timestamps_monotonic(timestamps: Sequence[_TDateTime]) -> bool:
    direction = _Direction.UNKNOWN
    for i in range(1, len(timestamps)):
        comparison = _get_direction(timestamps[i - 1], timestamps[i])
        if comparison == _Direction.UNKNOWN:
            continue

        if direction == _Direction.UNKNOWN:
            direction = comparison
        elif comparison != direction:
            return False
    return True


def _get_direction(left: _TDateTime, right: _TDateTime) -> _Direction:
    if left < right:
        return _Direction.INCREASING
    if right < left:
        return _Direction.DECREASING
    return _Direction.UNKNOWN


class IrregularSampleIntervalStrategy(SampleIntervalStrategy[_TDateTime, _TTimeDelta]):
    """Implements SampleIntervalMode.IRREGULAR specific behavior."""

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
        validate_unsupported_arg("timestamp", timestamp)
        validate_unsupported_arg("time offset", time_offset)
        validate_unsupported_arg("sample interval", sample_interval)
        if not isinstance(timestamps, Sequence) or not all(
            isinstance(ts, datetime_type) for ts in timestamps
        ):
            raise invalid_arg_type("timestamps", "sequence of datetime objects", timestamps)
        if not _are_timestamps_monotonic(timestamps):
            raise ValueError("The timestamps must be in ascending or descending order.")

    def get_timestamps(  # noqa: D102 - Missing docstring in public method - override
        self, timing: BaseTiming[_TDateTime, _TTimeDelta], start_index: int, count: int
    ) -> Iterable[_TDateTime]:
        assert timing._timestamps is not None
        if count > len(timing._timestamps):
            raise ValueError("The count must be less than or equal to the number of timestamps.")
        return timing._timestamps[start_index : start_index + count]

    def append_timestamps(  # noqa: D102 - Missing docstring in public method - override
        self,
        timing: BaseTiming[_TDateTime, _TTimeDelta],
        timestamps: Sequence[_TDateTime] | None,
    ) -> BaseTiming[_TDateTime, _TTimeDelta]:
        assert timing._timestamps is not None

        if timestamps is None:
            raise TimingMismatchError(
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

            return timing.__class__.create_with_irregular_interval(timing._timestamps + timestamps)

    def append_timing(  # noqa: D102 - Missing docstring in public method - override
        self,
        timing: BaseTiming[_TDateTime, _TTimeDelta],
        other: BaseTiming[_TDateTime, _TTimeDelta],
    ) -> BaseTiming[_TDateTime, _TTimeDelta]:
        if other._sample_interval_mode != SampleIntervalMode.IRREGULAR:
            raise sample_interval_mode_mismatch()

        assert timing._timestamps is not None and other._timestamps is not None

        if len(timing._timestamps) == 0:
            return other
        elif len(other._timestamps) == 0:
            return timing
        else:
            # The constructor will verify that the combined list of timestamps is monotonic. This is
            # not optimal for a large number of appends.
            return timing.__class__.create_with_irregular_interval(
                timing._timestamps + other._timestamps
            )
