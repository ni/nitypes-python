from __future__ import annotations

import datetime as dt
import sys

import hightime as ht
import pytest

from nitypes.waveform import PrecisionWaveformTiming, WaveformSampleIntervalMode

if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type


###############################################################################
# empty
###############################################################################
def test___empty___is_waveform_timing() -> None:
    assert_type(PrecisionWaveformTiming.EMPTY, PrecisionWaveformTiming)
    assert isinstance(PrecisionWaveformTiming.EMPTY, PrecisionWaveformTiming)


def test___empty___no_timestamp() -> None:
    assert not PrecisionWaveformTiming.EMPTY.has_timestamp
    with pytest.raises(RuntimeError) as exc:
        _ = PrecisionWaveformTiming.EMPTY.timestamp

    assert exc.value.args[0] == "The waveform timing does not have a timestamp."


def test___empty___no_start_time() -> None:
    with pytest.raises(RuntimeError) as exc:
        _ = PrecisionWaveformTiming.EMPTY.start_time

    assert exc.value.args[0] == "The waveform timing does not have a timestamp."


def test___empty___default_time_offset() -> None:
    assert PrecisionWaveformTiming.EMPTY.time_offset == ht.timedelta()


def test___empty___no_sample_interval() -> None:
    assert not PrecisionWaveformTiming.EMPTY._has_sample_interval
    with pytest.raises(RuntimeError) as exc:
        _ = PrecisionWaveformTiming.EMPTY.sample_interval

    assert exc.value.args[0] == "The waveform timing does not have a sample interval."


def test___empty___sample_interval_mode_none() -> None:
    assert PrecisionWaveformTiming.EMPTY.sample_interval_mode == WaveformSampleIntervalMode.NONE


###############################################################################
# create_with_no_interval
###############################################################################
def test___no_args___create_with_no_interval___creates_empty_waveform_timing() -> None:
    timing = PrecisionWaveformTiming.create_with_no_interval()

    assert_type(timing, PrecisionWaveformTiming)
    assert not timing.has_timestamp
    assert timing.time_offset == ht.timedelta()
    assert not timing._has_sample_interval
    assert timing.sample_interval_mode == WaveformSampleIntervalMode.NONE


def test___timestamp___create_with_no_interval___creates_waveform_timing_with_timestamp() -> None:
    timestamp = ht.datetime.now(dt.timezone.utc)
    timing = PrecisionWaveformTiming.create_with_no_interval(timestamp)

    assert_type(timing, PrecisionWaveformTiming)
    assert timing.timestamp == timestamp
    assert timing.time_offset == ht.timedelta()
    assert not timing._has_sample_interval
    assert timing.sample_interval_mode == WaveformSampleIntervalMode.NONE


def test___timestamp_and_time_offset___create_with_no_interval___creates_waveform_timing_with_timestamp_and_time_offset() -> (
    None
):
    timestamp = ht.datetime.now(dt.timezone.utc)
    time_offset = ht.timedelta(seconds=1.23)
    timing = PrecisionWaveformTiming.create_with_no_interval(timestamp, time_offset)

    assert_type(timing, PrecisionWaveformTiming)
    assert timing.timestamp == timestamp
    assert timing.time_offset == time_offset
    assert not timing._has_sample_interval
    assert timing.sample_interval_mode == WaveformSampleIntervalMode.NONE


def test___time_offset___create_with_no_interval___creates_waveform_timing_with_time_offset() -> (
    None
):
    time_offset = ht.timedelta(seconds=1.23)
    timing = PrecisionWaveformTiming.create_with_no_interval(time_offset=time_offset)

    assert_type(timing, PrecisionWaveformTiming)
    assert not timing.has_timestamp
    assert timing.time_offset == time_offset
    assert not timing._has_sample_interval
    assert timing.sample_interval_mode == WaveformSampleIntervalMode.NONE


###############################################################################
# create_with_regular_interval
###############################################################################
def test___sample_interval___create_with_regular_interval___creates_waveform_timing_with_sample_interval() -> (
    None
):
    sample_interval = ht.timedelta(milliseconds=1)

    timing = PrecisionWaveformTiming.create_with_regular_interval(sample_interval)

    assert_type(timing, PrecisionWaveformTiming)
    assert not timing.has_timestamp
    assert timing.time_offset == ht.timedelta()
    assert timing.sample_interval == sample_interval
    assert timing.sample_interval_mode == WaveformSampleIntervalMode.REGULAR


def test___sample_interval_and_timestamp___create_with_regular_interval___creates_waveform_timing_with_sample_interval_and_timestamp() -> (
    None
):
    sample_interval = ht.timedelta(milliseconds=1)
    timestamp = ht.datetime.now(dt.timezone.utc)

    timing = PrecisionWaveformTiming.create_with_regular_interval(sample_interval, timestamp)

    assert_type(timing, PrecisionWaveformTiming)
    assert timing.timestamp == timestamp
    assert timing.time_offset == ht.timedelta()
    assert timing.sample_interval == sample_interval
    assert timing.sample_interval_mode == WaveformSampleIntervalMode.REGULAR


def test___sample_interval_timestamp_and_time_offset___create_with_regular_interval___creates_waveform_timing_with_sample_interval_timestamp_and_time_offset() -> (
    None
):
    sample_interval = ht.timedelta(milliseconds=1)
    timestamp = ht.datetime.now(dt.timezone.utc)
    time_offset = ht.timedelta(seconds=1.23)

    timing = PrecisionWaveformTiming.create_with_regular_interval(
        sample_interval, timestamp, time_offset
    )

    assert_type(timing, PrecisionWaveformTiming)
    assert timing.timestamp == timestamp
    assert timing.time_offset == time_offset
    assert timing.sample_interval == sample_interval
    assert timing.sample_interval_mode == WaveformSampleIntervalMode.REGULAR


def test___sample_interval_and_time_offset___create_with_regular_interval___creates_waveform_timing_with_sample_interval_and_time_offset() -> (
    None
):
    sample_interval = ht.timedelta(milliseconds=1)
    time_offset = ht.timedelta(seconds=1.23)

    timing = PrecisionWaveformTiming.create_with_regular_interval(
        sample_interval, time_offset=time_offset
    )

    assert_type(timing, PrecisionWaveformTiming)
    assert not timing.has_timestamp
    assert timing.time_offset == time_offset
    assert timing.sample_interval == sample_interval
    assert timing.sample_interval_mode == WaveformSampleIntervalMode.REGULAR


###############################################################################
# create_with_irregular_interval
###############################################################################
def test___timestamps___create_with_irregular_interval___creates_waveform_timing_with_timestamps() -> (
    None
):
    start_time = ht.datetime.now(dt.timezone.utc)
    timestamps = [
        start_time,
        start_time + ht.timedelta(seconds=1),
        start_time + ht.timedelta(seconds=2.3),
        start_time + ht.timedelta(seconds=2.5),
    ]

    timing = PrecisionWaveformTiming.create_with_irregular_interval(timestamps)

    assert_type(timing, PrecisionWaveformTiming)
    assert not timing.has_timestamp
    assert timing.time_offset == ht.timedelta()
    assert not timing._has_sample_interval
    assert timing.sample_interval_mode == WaveformSampleIntervalMode.IRREGULAR
    assert timing._timestamps == timestamps


###############################################################################
# get_timestamps
###############################################################################
def test___no_interval___get_timestamps___raises_runtime_error() -> None:
    start_time = ht.datetime.now(dt.timezone.utc)
    timing = PrecisionWaveformTiming.create_with_no_interval(start_time)

    with pytest.raises(RuntimeError) as exc:
        _ = timing.get_timestamps(0, 5)

    assert exc.value.args[0].startswith(
        "The waveform timing does not have valid timestamp information."
    )


def test___regular_interval___get_timestamps___gets_timestamps() -> None:
    start_time = ht.datetime.now(dt.timezone.utc)
    sample_interval = ht.timedelta(milliseconds=1)
    timing = PrecisionWaveformTiming.create_with_regular_interval(sample_interval, start_time)

    assert list(timing.get_timestamps(3, 4)) == [
        start_time + 3 * sample_interval,
        start_time + 4 * sample_interval,
        start_time + 5 * sample_interval,
        start_time + 6 * sample_interval,
    ]


def test___irregular_interval___get_timestamps___gets_timestamps() -> None:
    start_time = ht.datetime.now(dt.timezone.utc)
    sample_interval = ht.timedelta(milliseconds=1)
    timestamps = [start_time + i * sample_interval for i in range(10)]
    timing = PrecisionWaveformTiming.create_with_irregular_interval(timestamps)

    assert list(timing.get_timestamps(0, 10)) == timestamps


def test___irregular_interval_subset___get_timestamps___gets_timestamps() -> None:
    start_time = ht.datetime.now(dt.timezone.utc)
    sample_interval = ht.timedelta(milliseconds=1)
    timestamps = [start_time + i * sample_interval for i in range(10)]
    timing = PrecisionWaveformTiming.create_with_irregular_interval(timestamps)

    assert list(timing.get_timestamps(3, 4)) == timestamps[3:7]
