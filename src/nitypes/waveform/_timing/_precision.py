from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import hightime as ht

from nitypes._typing import override
from nitypes.waveform._timing._base import BaseTiming, SampleIntervalMode


class PrecisionTiming(BaseTiming[ht.datetime, ht.timedelta]):
    """High-precision waveform timing using the hightime package.

    The hightime package has up to yoctosecond precision.
    """

    _DEFAULT_TIME_OFFSET = ht.timedelta()

    empty: ClassVar[PrecisionTiming]
    """A waveform timing object with no timestamp, time offset, or sample interval."""

    @override
    @staticmethod
    def create_with_no_interval(  # noqa: D102 - Missing docstring in public method - override
        timestamp: ht.datetime | None = None, time_offset: ht.timedelta | None = None
    ) -> PrecisionTiming:
        return PrecisionTiming(SampleIntervalMode.NONE, timestamp, time_offset)

    @override
    @staticmethod
    def create_with_regular_interval(  # noqa: D102 - Missing docstring in public method - override
        sample_interval: ht.timedelta,
        timestamp: ht.datetime | None = None,
        time_offset: ht.timedelta | None = None,
    ) -> PrecisionTiming:
        return PrecisionTiming(SampleIntervalMode.REGULAR, timestamp, time_offset, sample_interval)

    @override
    @staticmethod
    def create_with_irregular_interval(  # noqa: D102 - Missing docstring in public method - override
        timestamps: Sequence[ht.datetime],
    ) -> PrecisionTiming:
        return PrecisionTiming(SampleIntervalMode.IRREGULAR, timestamps=timestamps)

    @override
    @staticmethod
    def _get_datetime_type() -> type[ht.datetime]:
        return ht.datetime

    @override
    @staticmethod
    def _get_timedelta_type() -> type[ht.timedelta]:
        return ht.timedelta

    @override
    @staticmethod
    def _get_default_time_offset() -> ht.timedelta:
        return PrecisionTiming._DEFAULT_TIME_OFFSET

    def __init__(
        self,
        sample_interval_mode: SampleIntervalMode,
        timestamp: ht.datetime | None = None,
        time_offset: ht.timedelta | None = None,
        sample_interval: ht.timedelta | None = None,
        timestamps: Sequence[ht.datetime] | None = None,
    ) -> None:
        """Construct a high-precision waveform timing object.

        Most applications should use the named constructors instead:
        - PrecisionTiming.create_with_no_interval
        - PrecisionTiming.create_with_regular_interval
        - PrecisionTiming.create_with_irregular_interval
        """
        super().__init__(sample_interval_mode, timestamp, time_offset, sample_interval, timestamps)


PrecisionTiming.empty = PrecisionTiming.create_with_no_interval()
