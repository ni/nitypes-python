from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import ClassVar

from nitypes.waveform._timing._base import BaseTiming, SampleIntervalMode


class Timing(BaseTiming[dt.datetime, dt.timedelta]):
    """Waveform timing using the standard datetime module.

    The standard datetime module has up to microsecond precision. For higher precision, use
    PrecisionTiming.
    """

    _DEFAULT_TIME_OFFSET = dt.timedelta()

    empty: ClassVar[Timing]

    # TODO: can these be classmethods in BaseTiming?
    @staticmethod
    def create_with_no_interval(
        timestamp: dt.datetime | None = None, time_offset: dt.timedelta | None = None
    ) -> Timing:
        """Create a waveform timing object with no sample interval.

        Args:
            timestamp: A timestamp representing the start of an acquisition or a related
                occurrence.
            time_offset: The time difference between the timestamp and the time that the first
                sample was acquired.

        Returns:
            A waveform timing object.
        """
        return Timing(SampleIntervalMode.NONE, timestamp, time_offset)

    @staticmethod
    def create_with_regular_interval(
        sample_interval: dt.timedelta,
        timestamp: dt.datetime | None = None,
        time_offset: dt.timedelta | None = None,
    ) -> Timing:
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
        return Timing(SampleIntervalMode.REGULAR, timestamp, time_offset, sample_interval)

    @staticmethod
    def create_with_irregular_interval(
        timestamps: Sequence[dt.datetime],
    ) -> Timing:
        """Create a waveform timing object with an irregular sample interval.

        Args:
            timestamps: A sequence containing a timestamp for each sample in the waveform,
                specifying the time that the sample was acquired.

        Returns:
            A waveform timing object.
        """
        return Timing(SampleIntervalMode.IRREGULAR, timestamps=timestamps)

    @staticmethod
    def _get_datetime_type() -> type[dt.datetime]:
        return dt.datetime

    @staticmethod
    def _get_timedelta_type() -> type[dt.timedelta]:
        return dt.timedelta

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
"""A waveform timing object with no timestamp, time offset, or sample interval."""
