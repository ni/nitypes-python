from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import hightime as ht

from nitypes.waveform._base_timing import BaseWaveformTiming, SampleIntervalMode


class PrecisionWaveformTiming(BaseWaveformTiming[ht.datetime, ht.timedelta]):
    """High-precision waveform timing using the hightime package."""

    _DEFAULT_TIME_OFFSET = ht.timedelta()
    _DEFAULT_SAMPLE_INTERVAL = ht.timedelta()

    empty: ClassVar[PrecisionWaveformTiming]

    @staticmethod
    def create_with_no_interval(
        timestamp: ht.datetime | None = None, time_offset: ht.timedelta | None = None
    ) -> PrecisionWaveformTiming:
        """Create a waveform timing object with no sample interval.

        Args:
            timestamp: A timestamp representing the start of an acquisition or a related
                occurrence.
            time_offset: The time difference between the timestamp and the time that the first
                sample was acquired.

        Returns:
            A waveform timing object.
        """
        return PrecisionWaveformTiming(SampleIntervalMode.NONE, timestamp, time_offset)

    @staticmethod
    def create_with_regular_interval(
        sample_interval: ht.timedelta,
        timestamp: ht.datetime | None = None,
        time_offset: ht.timedelta | None = None,
    ) -> PrecisionWaveformTiming:
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
        return PrecisionWaveformTiming(
            SampleIntervalMode.REGULAR, timestamp, time_offset, sample_interval
        )

    @staticmethod
    def create_with_irregular_interval(
        timestamps: Sequence[ht.datetime],
    ) -> PrecisionWaveformTiming:
        """Create a waveform timing object with an irregular sample interval.

        Args:
            timestamps: A sequence containing a timestamp for each sample in the waveform,
                specifying the time that the sample was acquired.

        Returns:
            A waveform timing object.
        """
        return PrecisionWaveformTiming(SampleIntervalMode.IRREGULAR, timestamps=timestamps)

    @staticmethod
    def _get_datetime_type() -> type[ht.datetime]:
        return ht.datetime

    @staticmethod
    def _get_timedelta_type() -> type[ht.timedelta]:
        return ht.timedelta

    @staticmethod
    def _get_default_time_offset() -> ht.timedelta:
        return PrecisionWaveformTiming._DEFAULT_TIME_OFFSET

    @staticmethod
    def _get_default_sample_interval() -> ht.timedelta:
        return PrecisionWaveformTiming._DEFAULT_SAMPLE_INTERVAL

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
        - PrecisionWaveformTiming.create_with_no_interval
        - PrecisionWaveformTiming.create_with_regular_interval
        - PrecisionWaveformTiming.create_with_irregular_interval
        """
        super().__init__(sample_interval_mode, timestamp, time_offset, sample_interval, timestamps)


PrecisionWaveformTiming.empty = PrecisionWaveformTiming.create_with_no_interval()
"""A waveform timing object with no timestamp, time offset, or sample interval."""
