from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import hightime as ht

from nitypes.waveform._timing._base import BaseTiming, SampleIntervalMode


class PrecisionTiming(BaseTiming[ht.datetime, ht.timedelta]):
    """High-precision waveform timing using the hightime package.

    The hightime package has up to yoctosecond precision.
    """

    _DEFAULT_TIME_OFFSET = ht.timedelta()

    empty: ClassVar[PrecisionTiming]

    @staticmethod
    def create_with_no_interval(
        timestamp: ht.datetime | None = None, time_offset: ht.timedelta | None = None
    ) -> PrecisionTiming:
        """Create a waveform timing object with no sample interval.

        Args:
            timestamp: A timestamp representing the start of an acquisition or a related
                occurrence.
            time_offset: The time difference between the timestamp and the time that the first
                sample was acquired.

        Returns:
            A waveform timing object.
        """
        return PrecisionTiming(SampleIntervalMode.NONE, timestamp, time_offset)

    @staticmethod
    def create_with_regular_interval(
        sample_interval: ht.timedelta,
        timestamp: ht.datetime | None = None,
        time_offset: ht.timedelta | None = None,
    ) -> PrecisionTiming:
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
        return PrecisionTiming(SampleIntervalMode.REGULAR, timestamp, time_offset, sample_interval)

    @staticmethod
    def create_with_irregular_interval(
        timestamps: Sequence[ht.datetime],
    ) -> PrecisionTiming:
        """Create a waveform timing object with an irregular sample interval.

        Args:
            timestamps: A sequence containing a timestamp for each sample in the waveform,
                specifying the time that the sample was acquired.

        Returns:
            A waveform timing object.
        """
        return PrecisionTiming(SampleIntervalMode.IRREGULAR, timestamps=timestamps)

    @staticmethod
    def _get_datetime_type() -> type[ht.datetime]:
        return ht.datetime

    @staticmethod
    def _get_timedelta_type() -> type[ht.timedelta]:
        return ht.timedelta

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
"""A waveform timing object with no timestamp, time offset, or sample interval."""
