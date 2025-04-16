from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import ClassVar

from nitypes.waveform._base_timing import BaseWaveformTiming, WaveformSampleIntervalMode


class WaveformTiming(BaseWaveformTiming[dt.datetime, dt.timedelta]):
    """Waveform timing using the standard datetime module."""

    _DEFAULT_TIME_OFFSET = dt.timedelta()
    _DEFAULT_SAMPLE_INTERVAL = dt.timedelta()

    EMPTY: ClassVar[WaveformTiming]

    @staticmethod
    def create_with_no_interval(
        timestamp: dt.datetime | None = None, time_offset: dt.timedelta | None = None
    ) -> WaveformTiming:
        """Create a waveform timing object with no sample interval.

        Args:
            timestamp: A timestamp representing the start of an acquisition or a related
                occurrence.
            time_offset: The time difference between the timestamp and the time that the first
                sample was acquired.

        Returns:
            A waveform timing object.
        """
        if not isinstance(timestamp, (dt.datetime, type(None))):
            raise TypeError("The timestamp must be a datetime or None.")
        if not isinstance(time_offset, (dt.timedelta, type(None))):
            raise TypeError("The time offset must be a timedelta or None.")
        return WaveformTiming(
            timestamp,
            time_offset,
            WaveformTiming._DEFAULT_SAMPLE_INTERVAL,
            WaveformSampleIntervalMode.NONE,
            None,
        )

    @staticmethod
    def create_with_regular_interval(
        sample_interval: dt.timedelta,
        timestamp: dt.datetime | None = None,
        time_offset: dt.timedelta | None = None,
    ) -> WaveformTiming:
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
        if not isinstance(sample_interval, dt.timedelta):
            raise TypeError("The sample interval must be a timedelta.")
        if not isinstance(timestamp, (dt.datetime, type(None))):
            raise TypeError("The timestamp must be a datetime or None.")
        if not isinstance(time_offset, (dt.timedelta, type(None))):
            raise TypeError("The timestamp must be a timedelta or None.")
        return WaveformTiming(
            timestamp, time_offset, sample_interval, WaveformSampleIntervalMode.REGULAR, None
        )

    @staticmethod
    def create_with_irregular_interval(
        timestamps: Sequence[dt.datetime],
    ) -> WaveformTiming:
        """Create a waveform timing object with an irregular sample interval.

        Args:
            timestamps: A sequence containing a timestamp for each sample in the waveform,
                specifying the time that the sample was acquired.

        Returns:
            A waveform timing object.
        """
        if not isinstance(timestamps, Sequence) or not all(
            isinstance(ts, dt.datetime) for ts in timestamps
        ):
            raise TypeError("The timestamps argument must be a sequence of datetime objects.")
        return WaveformTiming(
            None,
            WaveformTiming._DEFAULT_TIME_OFFSET,
            WaveformTiming._DEFAULT_SAMPLE_INTERVAL,
            WaveformSampleIntervalMode.IRREGULAR,
            list(timestamps),
        )

    def __init__(
        self,
        timestamp: dt.datetime | None,
        time_offset: dt.timedelta | None,
        sample_interval: dt.timedelta | None,
        sample_interval_mode: WaveformSampleIntervalMode,
        timestamps: list[dt.datetime] | None,
    ) -> None:
        """Construct a waveform timing object.

        This constructor is a private implementation detail. Please use the static methods
        create_with_no_interval, create_with_regular_interval, and create_with_irregular_interval
        instead.
        """
        if time_offset is None:
            time_offset = WaveformTiming._DEFAULT_TIME_OFFSET
        if sample_interval is None:
            sample_interval = WaveformTiming._DEFAULT_SAMPLE_INTERVAL
        super().__init__(timestamp, time_offset, sample_interval, sample_interval_mode, timestamps)

    def __eq__(self, value: object) -> bool:  # noqa: D105 - Missing docstring in magic method
        if not isinstance(value, WaveformTiming):
            return NotImplemented
        return super().__eq__(value)


WaveformTiming.EMPTY = WaveformTiming.create_with_no_interval()
"""A waveform timing object with no timestamp, time offset, or sample interval."""
