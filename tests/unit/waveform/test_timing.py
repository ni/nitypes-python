from __future__ import annotations

import datetime as dt
import sys
from copy import deepcopy

import pytest

from nitypes.waveform import SampleIntervalMode, WaveformTiming

if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type


###############################################################################
# empty
###############################################################################
def test___empty___is_waveform_timing() -> None:
    assert_type(WaveformTiming.empty, WaveformTiming)
    assert isinstance(WaveformTiming.empty, WaveformTiming)


def test___empty___no_timestamp() -> None:
    assert not WaveformTiming.empty.has_timestamp
    with pytest.raises(RuntimeError) as exc:
        _ = WaveformTiming.empty.timestamp

    assert exc.value.args[0] == "The waveform timing does not have a timestamp."


def test___empty___no_start_time() -> None:
    with pytest.raises(RuntimeError) as exc:
        _ = WaveformTiming.empty.start_time

    assert exc.value.args[0] == "The waveform timing does not have a timestamp."


def test___empty___default_time_offset() -> None:
    assert WaveformTiming.empty.time_offset == dt.timedelta()


def test___empty___no_sample_interval() -> None:
    assert WaveformTiming.empty._sample_interval is None
    with pytest.raises(RuntimeError) as exc:
        _ = WaveformTiming.empty.sample_interval

    assert exc.value.args[0] == "The waveform timing does not have a sample interval."


def test___empty___sample_interval_mode_none() -> None:
    assert WaveformTiming.empty.sample_interval_mode == SampleIntervalMode.NONE


###############################################################################
# create_with_no_interval
###############################################################################
def test___no_args___create_with_no_interval___creates_empty_waveform_timing() -> None:
    timing = WaveformTiming.create_with_no_interval()

    assert_type(timing, WaveformTiming)
    assert not timing.has_timestamp
    assert timing.time_offset == dt.timedelta()
    assert timing._sample_interval is None
    assert timing.sample_interval_mode == SampleIntervalMode.NONE


def test___timestamp___create_with_no_interval___creates_waveform_timing_with_timestamp() -> None:
    timestamp = dt.datetime.now(dt.timezone.utc)
    timing = WaveformTiming.create_with_no_interval(timestamp)

    assert_type(timing, WaveformTiming)
    assert timing.timestamp == timestamp
    assert timing.time_offset == dt.timedelta()
    assert timing._sample_interval is None
    assert timing.sample_interval_mode == SampleIntervalMode.NONE


def test___timestamp_and_time_offset___create_with_no_interval___creates_waveform_timing_with_timestamp_and_time_offset() -> (
    None
):
    timestamp = dt.datetime.now(dt.timezone.utc)
    time_offset = dt.timedelta(seconds=1.23)
    timing = WaveformTiming.create_with_no_interval(timestamp, time_offset)

    assert_type(timing, WaveformTiming)
    assert timing.timestamp == timestamp
    assert timing.time_offset == time_offset
    assert timing._sample_interval is None
    assert timing.sample_interval_mode == SampleIntervalMode.NONE


def test___time_offset___create_with_no_interval___creates_waveform_timing_with_time_offset() -> (
    None
):
    time_offset = dt.timedelta(seconds=1.23)
    timing = WaveformTiming.create_with_no_interval(time_offset=time_offset)

    assert_type(timing, WaveformTiming)
    assert not timing.has_timestamp
    assert timing.time_offset == time_offset
    assert timing._sample_interval is None
    assert timing.sample_interval_mode == SampleIntervalMode.NONE


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
    assert timing.sample_interval_mode == SampleIntervalMode.REGULAR


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
    assert timing.sample_interval_mode == SampleIntervalMode.REGULAR


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
    assert timing.sample_interval_mode == SampleIntervalMode.REGULAR


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
    assert timing.sample_interval_mode == SampleIntervalMode.REGULAR


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
    assert timing._sample_interval is None
    assert timing.sample_interval_mode == SampleIntervalMode.IRREGULAR
    assert timing._timestamps == timestamps


###############################################################################
# get_timestamps
###############################################################################
def test___no_interval___get_timestamps___raises_runtime_error() -> None:
    start_time = dt.datetime.now(dt.timezone.utc)
    timing = WaveformTiming.create_with_no_interval(start_time)

    with pytest.raises(RuntimeError) as exc:
        _ = timing.get_timestamps(0, 5)

    assert exc.value.args[0].startswith(
        "The waveform timing does not have valid timestamp information."
    )


def test___regular_interval___get_timestamps___gets_timestamps() -> None:
    start_time = dt.datetime.now(dt.timezone.utc)
    sample_interval = dt.timedelta(milliseconds=1)
    timing = WaveformTiming.create_with_regular_interval(sample_interval, start_time)

    assert list(timing.get_timestamps(3, 4)) == [
        start_time + 3 * sample_interval,
        start_time + 4 * sample_interval,
        start_time + 5 * sample_interval,
        start_time + 6 * sample_interval,
    ]


def test___irregular_interval___get_timestamps___gets_timestamps() -> None:
    start_time = dt.datetime.now(dt.timezone.utc)
    sample_interval = dt.timedelta(milliseconds=1)
    timestamps = [start_time + i * sample_interval for i in range(10)]
    timing = WaveformTiming.create_with_irregular_interval(timestamps)

    assert list(timing.get_timestamps(0, 10)) == timestamps


def test___irregular_interval_subset___get_timestamps___gets_timestamps() -> None:
    start_time = dt.datetime.now(dt.timezone.utc)
    sample_interval = dt.timedelta(milliseconds=1)
    timestamps = [start_time + i * sample_interval for i in range(10)]
    timing = WaveformTiming.create_with_irregular_interval(timestamps)

    assert list(timing.get_timestamps(3, 4)) == timestamps[3:7]


###############################################################################
# magic methods
###############################################################################
@pytest.mark.parametrize(
    "value",
    [
        WaveformTiming.create_with_no_interval(),
        WaveformTiming.create_with_no_interval(dt.datetime(2025, 1, 1)),
        WaveformTiming.create_with_no_interval(None, dt.timedelta(seconds=1)),
        WaveformTiming.create_with_no_interval(dt.datetime(2025, 1, 1), dt.timedelta(seconds=1)),
        WaveformTiming.create_with_regular_interval(dt.timedelta(milliseconds=1)),
        WaveformTiming.create_with_regular_interval(
            dt.timedelta(milliseconds=1), dt.datetime(2025, 1, 1)
        ),
        WaveformTiming.create_with_regular_interval(
            dt.timedelta(milliseconds=1), dt.datetime(2025, 1, 1), dt.timedelta(seconds=1)
        ),
        WaveformTiming.create_with_irregular_interval(
            [dt.datetime(2025, 1, 1), dt.datetime(2025, 1, 2)]
        ),
    ],
)
def test___deep_copy___equality___equal(value: WaveformTiming) -> None:
    other = deepcopy(value)

    assert value == other
    assert not (value != other)


@pytest.mark.parametrize(
    "lhs, rhs",
    [
        (
            WaveformTiming.create_with_no_interval(
                dt.datetime(2025, 1, 1), dt.timedelta(seconds=1)
            ),
            WaveformTiming.create_with_no_interval(
                dt.datetime(2025, 1, 2), dt.timedelta(seconds=1)
            ),
        ),
        (
            WaveformTiming.create_with_no_interval(
                dt.datetime(2025, 1, 1), dt.timedelta(seconds=1)
            ),
            WaveformTiming.create_with_no_interval(
                dt.datetime(2025, 1, 1), dt.timedelta(seconds=2)
            ),
        ),
        (
            WaveformTiming.create_with_regular_interval(
                dt.timedelta(milliseconds=1), dt.datetime(2025, 1, 1), dt.timedelta(seconds=1)
            ),
            WaveformTiming.create_with_regular_interval(
                dt.timedelta(milliseconds=2), dt.datetime(2025, 1, 1), dt.timedelta(seconds=1)
            ),
        ),
        (
            WaveformTiming.create_with_regular_interval(
                dt.timedelta(milliseconds=1), dt.datetime(2025, 1, 1), dt.timedelta(seconds=1)
            ),
            WaveformTiming.create_with_regular_interval(
                dt.timedelta(milliseconds=1), dt.datetime(2025, 1, 2), dt.timedelta(seconds=1)
            ),
        ),
        (
            WaveformTiming.create_with_regular_interval(
                dt.timedelta(milliseconds=1), dt.datetime(2025, 1, 1), dt.timedelta(seconds=1)
            ),
            WaveformTiming.create_with_regular_interval(
                dt.timedelta(milliseconds=1), dt.datetime(2025, 1, 1), dt.timedelta(seconds=2)
            ),
        ),
        (
            WaveformTiming.create_with_irregular_interval(
                [dt.datetime(2025, 1, 1), dt.datetime(2025, 1, 2)]
            ),
            WaveformTiming.create_with_irregular_interval(
                [dt.datetime(2025, 1, 3), dt.datetime(2025, 1, 2)]
            ),
        ),
    ],
)
def test___different_value___equality___not_equal(
    lhs: WaveformTiming,
    rhs: WaveformTiming,
) -> None:
    assert not (lhs == rhs)
    assert lhs != rhs


@pytest.mark.parametrize(
    "value, expected_repr",
    [
        (WaveformTiming.create_with_no_interval(), "WaveformTiming(SampleIntervalMode.NONE)"),
        (
            WaveformTiming.create_with_no_interval(dt.datetime(2025, 1, 1)),
            "WaveformTiming(SampleIntervalMode.NONE, timestamp=datetime.datetime(2025, 1, 1, 0, 0))",
        ),
        (
            WaveformTiming.create_with_no_interval(None, dt.timedelta(seconds=1)),
            "WaveformTiming(SampleIntervalMode.NONE, time_offset=datetime.timedelta(seconds=1))",
        ),
        (
            WaveformTiming.create_with_no_interval(
                dt.datetime(2025, 1, 1), dt.timedelta(seconds=1)
            ),
            "WaveformTiming(SampleIntervalMode.NONE, timestamp=datetime.datetime(2025, 1, 1, 0, 0), time_offset=datetime.timedelta(seconds=1))",
        ),
        (
            WaveformTiming.create_with_regular_interval(dt.timedelta(milliseconds=1)),
            "WaveformTiming(SampleIntervalMode.REGULAR, sample_interval=datetime.timedelta(microseconds=1000))",
        ),
        (
            WaveformTiming.create_with_regular_interval(
                dt.timedelta(milliseconds=1), dt.datetime(2025, 1, 1)
            ),
            "WaveformTiming(SampleIntervalMode.REGULAR, timestamp=datetime.datetime(2025, 1, 1, 0, 0), sample_interval=datetime.timedelta(microseconds=1000))",
        ),
        (
            WaveformTiming.create_with_regular_interval(
                dt.timedelta(milliseconds=1), dt.datetime(2025, 1, 1), dt.timedelta(seconds=1)
            ),
            "WaveformTiming(SampleIntervalMode.REGULAR, timestamp=datetime.datetime(2025, 1, 1, 0, 0), time_offset=datetime.timedelta(seconds=1), sample_interval=datetime.timedelta(microseconds=1000))",
        ),
        (
            WaveformTiming.create_with_irregular_interval(
                [dt.datetime(2025, 1, 1), dt.datetime(2025, 1, 2)]
            ),
            "WaveformTiming(SampleIntervalMode.IRREGULAR, timestamps=[datetime.datetime(2025, 1, 1, 0, 0), datetime.datetime(2025, 1, 2, 0, 0)])",
        ),
    ],
)
def test___various_values___repr___looks_ok(value: WaveformTiming, expected_repr: str) -> None:
    assert repr(value) == expected_repr
