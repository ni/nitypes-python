from __future__ import annotations

import datetime as dt
import sys

import pytest

from nitypes.waveform import WaveformSampleIntervalMode, WaveformTiming

if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type


###############################################################################
# empty
###############################################################################
def test___empty___is_waveform_timing() -> None:
    assert_type(WaveformTiming.EMPTY, WaveformTiming)
    assert isinstance(WaveformTiming.EMPTY, WaveformTiming)


def test___empty___no_timestamp() -> None:
    assert not WaveformTiming.EMPTY.has_timestamp
    with pytest.raises(RuntimeError) as exc:
        _ = WaveformTiming.EMPTY.timestamp

    assert exc.value.args[0] == "The waveform timing does not have a timestamp."


def test___empty___no_start_time() -> None:
    with pytest.raises(RuntimeError) as exc:
        _ = WaveformTiming.EMPTY.start_time

    assert exc.value.args[0] == "The waveform timing does not have a timestamp."


def test___empty___default_time_offset() -> None:
    assert WaveformTiming.EMPTY.time_offset == dt.timedelta()


def test___empty___no_sample_interval() -> None:
    assert not WaveformTiming.EMPTY._has_sample_interval
    with pytest.raises(RuntimeError) as exc:
        _ = WaveformTiming.EMPTY.sample_interval

    assert exc.value.args[0] == "The waveform timing does not have a sample interval."


def test___empty___sample_interval_mode_none() -> None:
    assert WaveformTiming.EMPTY.sample_interval_mode == WaveformSampleIntervalMode.NONE


###############################################################################
# create_with_no_interval
###############################################################################
def test___no_args___create_with_no_interval___creates_empty_waveform_timing() -> None:
    timing = WaveformTiming.create_with_no_interval()

    assert_type(timing, WaveformTiming)
    assert not timing.has_timestamp
    assert timing.time_offset == dt.timedelta()
    assert not timing._has_sample_interval
    assert timing.sample_interval_mode == WaveformSampleIntervalMode.NONE


def test___timestamp___create_with_no_interval___creates_waveform_timing_with_timestamp() -> None:
    timestamp = dt.datetime.now(dt.timezone.utc)
    timing = WaveformTiming.create_with_no_interval(timestamp)

    assert_type(timing, WaveformTiming)
    assert timing.timestamp == timestamp
    assert timing.time_offset == dt.timedelta()
    assert not timing._has_sample_interval
    assert timing.sample_interval_mode == WaveformSampleIntervalMode.NONE


def test___timestamp_and_time_offset___create_with_no_interval___creates_waveform_timing_with_timestamp_and_time_offset() -> (
    None
):
    timestamp = dt.datetime.now(dt.timezone.utc)
    time_offset = dt.timedelta(seconds=1.23)
    timing = WaveformTiming.create_with_no_interval(timestamp, time_offset)

    assert_type(timing, WaveformTiming)
    assert timing.timestamp == timestamp
    assert timing.time_offset == time_offset
    assert not timing._has_sample_interval
    assert timing.sample_interval_mode == WaveformSampleIntervalMode.NONE


def test___time_offset___create_with_no_interval___creates_waveform_timing_with_time_offset() -> (
    None
):
    time_offset = dt.timedelta(seconds=1.23)
    timing = WaveformTiming.create_with_no_interval(time_offset=time_offset)

    assert_type(timing, WaveformTiming)
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
    sample_interval = dt.timedelta(milliseconds=1)

    timing = WaveformTiming.create_with_regular_interval(sample_interval)

    assert_type(timing, WaveformTiming)
    assert not timing.has_timestamp
    assert timing.time_offset == dt.timedelta()
    assert timing.sample_interval == sample_interval
    assert timing.sample_interval_mode == WaveformSampleIntervalMode.REGULAR


def test___sample_interval_and_timestamp___create_with_regular_interval___creates_waveform_timing_with_sample_interval_and_timestamp() -> (
    None
):
    sample_interval = dt.timedelta(milliseconds=1)
    timestamp = dt.datetime.now(dt.timezone.utc)

    timing = WaveformTiming.create_with_regular_interval(sample_interval, timestamp)

    assert_type(timing, WaveformTiming)
    assert timing.timestamp == timestamp
    assert timing.time_offset == dt.timedelta()
    assert timing.sample_interval == sample_interval
    assert timing.sample_interval_mode == WaveformSampleIntervalMode.REGULAR


def test___sample_interval_timestamp_and_time_offset___create_with_regular_interval___creates_waveform_timing_with_sample_interval_timestamp_and_time_offset() -> (
    None
):
    sample_interval = dt.timedelta(milliseconds=1)
    timestamp = dt.datetime.now(dt.timezone.utc)
    time_offset = dt.timedelta(seconds=1.23)

    timing = WaveformTiming.create_with_regular_interval(sample_interval, timestamp, time_offset)

    assert_type(timing, WaveformTiming)
    assert timing.timestamp == timestamp
    assert timing.time_offset == time_offset
    assert timing.sample_interval == sample_interval
    assert timing.sample_interval_mode == WaveformSampleIntervalMode.REGULAR


def test___sample_interval_and_time_offset___create_with_regular_interval___creates_waveform_timing_with_sample_interval_and_time_offset() -> (
    None
):
    sample_interval = dt.timedelta(milliseconds=1)
    time_offset = dt.timedelta(seconds=1.23)

    timing = WaveformTiming.create_with_regular_interval(sample_interval, time_offset=time_offset)

    assert_type(timing, WaveformTiming)
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
    start_time = dt.datetime.now(dt.timezone.utc)
    timestamps = [
        start_time,
        start_time + dt.timedelta(seconds=1),
        start_time + dt.timedelta(seconds=2.3),
        start_time + dt.timedelta(seconds=2.5),
    ]

    timing = WaveformTiming.create_with_irregular_interval(timestamps)

    assert_type(timing, WaveformTiming)
    assert not timing.has_timestamp
    assert timing.time_offset == dt.timedelta()
    assert not timing._has_sample_interval
    assert timing.sample_interval_mode == WaveformSampleIntervalMode.IRREGULAR
    assert timing._timestamps == timestamps
