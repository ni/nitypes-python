from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import ClassVar

from nitypes._typing import override
from nitypes.waveform._timing._base import BaseTiming, SampleIntervalMode


class Timing(BaseTiming[dt.datetime, dt.timedelta]):
    """Waveform timing using the standard datetime module.

    The standard datetime module has up to microsecond precision. For higher precision, use
    PrecisionTiming.
    """

    _DEFAULT_TIME_OFFSET = dt.timedelta()

    empty: ClassVar[Timing]
    """A waveform timing object with no timestamp, time offset, or sample interval."""

    @override
    @staticmethod
    def create_with_no_interval(  # noqa: D102 - Missing docstring in public method - override
        timestamp: dt.datetime | None = None, time_offset: dt.timedelta | None = None
    ) -> Timing:
        return Timing(SampleIntervalMode.NONE, timestamp, time_offset)

    @override
    @staticmethod
    def create_with_regular_interval(  # noqa: D102 - Missing docstring in public method - override
        sample_interval: dt.timedelta,
        timestamp: dt.datetime | None = None,
        time_offset: dt.timedelta | None = None,
    ) -> Timing:
        return Timing(SampleIntervalMode.REGULAR, timestamp, time_offset, sample_interval)

    @override
    @staticmethod
    def create_with_irregular_interval(  # noqa: D102 - Missing docstring in public method - override
        timestamps: Sequence[dt.datetime],
    ) -> Timing:
        return Timing(SampleIntervalMode.IRREGULAR, timestamps=timestamps)

    @override
    @staticmethod
    def _get_datetime_type() -> type[dt.datetime]:
        return dt.datetime

    @override
    @staticmethod
    def _get_timedelta_type() -> type[dt.timedelta]:
        return dt.timedelta

    @override
    @staticmethod
    def _get_default_time_offset() -> dt.timedelta:
        return Timing._DEFAULT_TIME_OFFSET

    def __init__(
        self,
        sample_interval_mode: SampleIntervalMode,
        timestamp: dt.datetime | None = None,
        time_offset: dt.timedelta | None = None,
        sample_interval: dt.timedelta | None = None,
        timestamps: Sequence[dt.datetime] | None = None,
    ) -> None:
        """Construct a waveform timing object.

        Most applications should use the named constructors instead:
        - Timing.create_with_no_interval
        - Timing.create_with_regular_interval
        - Timing.create_with_irregular_interval
        """
        super().__init__(sample_interval_mode, timestamp, time_offset, sample_interval, timestamps)


Timing.empty = Timing.create_with_no_interval()
