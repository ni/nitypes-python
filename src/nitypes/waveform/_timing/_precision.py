from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, final

import hightime as ht
from typing_extensions import Self, override

from nitypes.waveform._timing._base import BaseTiming
from nitypes.waveform._timing._sample_interval import SampleIntervalMode


@final
class PrecisionTiming(BaseTiming[ht.datetime, ht.timedelta]):
    """High-precision waveform timing using the hightime package.

    The hightime package has up to yoctosecond precision.

    Waveform timing objects are immutable.
    """

    _DEFAULT_TIME_OFFSET = ht.timedelta()

    empty: ClassVar[PrecisionTiming]
    """A waveform timing object with no timestamp, time offset, or sample interval."""

    @override
    @classmethod
    def create_with_no_interval(  # noqa: D102 - Missing docstring in public method - override
        cls, timestamp: ht.datetime | None = None, time_offset: ht.timedelta | None = None
    ) -> Self:
        return cls(SampleIntervalMode.NONE, timestamp, time_offset)

    @override
    @classmethod
    def create_with_regular_interval(  # noqa: D102 - Missing docstring in public method - override
        cls,
        sample_interval: ht.timedelta,
        timestamp: ht.datetime | None = None,
        time_offset: ht.timedelta | None = None,
    ) -> Self:
        return cls(SampleIntervalMode.REGULAR, timestamp, time_offset, sample_interval)

    @override
    @classmethod
    def create_with_irregular_interval(  # noqa: D102 - Missing docstring in public method - override
        cls,
        timestamps: Sequence[ht.datetime],
    ) -> Self:
        return cls(SampleIntervalMode.IRREGULAR, timestamps=timestamps)

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
        *,
        copy_timestamps: bool = True,
    ) -> None:
        """Initialize a new high-precision waveform timing object.

        Most applications should use the named constructors instead:
        - PrecisionTiming.create_with_no_interval
        - PrecisionTiming.create_with_regular_interval
        - PrecisionTiming.create_with_irregular_interval
        """
        super().__init__(
            sample_interval_mode,
            timestamp,
            time_offset,
            sample_interval,
            timestamps,
            copy_timestamps=copy_timestamps,
        )


PrecisionTiming.empty = PrecisionTiming.create_with_no_interval()
