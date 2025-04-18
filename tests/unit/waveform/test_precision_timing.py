from __future__ import annotations

import datetime as dt
import sys
from copy import deepcopy

import hightime as ht
import pytest

from nitypes.waveform import PrecisionWaveformTiming, SampleIntervalMode

if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type


###############################################################################
# empty
###############################################################################
def test___empty___is_waveform_timing() -> None:
    assert_type(PrecisionWaveformTiming.empty, PrecisionWaveformTiming)
    assert isinstance(PrecisionWaveformTiming.empty, PrecisionWaveformTiming)


def test___empty___no_timestamp() -> None:
    assert not PrecisionWaveformTiming.empty.has_timestamp
    with pytest.raises(RuntimeError) as exc:
        _ = PrecisionWaveformTiming.empty.timestamp

    assert exc.value.args[0] == "The waveform timing does not have a timestamp."


def test___empty___no_start_time() -> None:
    with pytest.raises(RuntimeError) as exc:
        _ = PrecisionWaveformTiming.empty.start_time

    assert exc.value.args[0] == "The waveform timing does not have a timestamp."


def test___empty___default_time_offset() -> None:
    assert PrecisionWaveformTiming.empty.time_offset == ht.timedelta()


def test___empty___no_sample_interval() -> None:
    assert not PrecisionWaveformTiming.empty._has_sample_interval
    with pytest.raises(RuntimeError) as exc:
        _ = PrecisionWaveformTiming.empty.sample_interval

    assert exc.value.args[0] == "The waveform timing does not have a sample interval."


def test___empty___sample_interval_mode_none() -> None:
    assert PrecisionWaveformTiming.empty.sample_interval_mode == SampleIntervalMode.NONE


###############################################################################
# create_with_no_interval
###############################################################################
def test___no_args___create_with_no_interval___creates_empty_waveform_timing() -> None:
    timing = PrecisionWaveformTiming.create_with_no_interval()

    assert_type(timing, PrecisionWaveformTiming)
    assert not timing.has_timestamp
    assert timing.time_offset == ht.timedelta()
    assert not timing._has_sample_interval
    assert timing.sample_interval_mode == SampleIntervalMode.NONE


def test___timestamp___create_with_no_interval___creates_waveform_timing_with_timestamp() -> None:
    timestamp = ht.datetime.now(dt.timezone.utc)
    timing = PrecisionWaveformTiming.create_with_no_interval(timestamp)

    assert_type(timing, PrecisionWaveformTiming)
    assert timing.timestamp == timestamp
    assert timing.time_offset == ht.timedelta()
    assert not timing._has_sample_interval
    assert timing.sample_interval_mode == SampleIntervalMode.NONE


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
    assert timing.sample_interval_mode == SampleIntervalMode.NONE


def test___time_offset___create_with_no_interval___creates_waveform_timing_with_time_offset() -> (
    None
):
    time_offset = ht.timedelta(seconds=1.23)
    timing = PrecisionWaveformTiming.create_with_no_interval(time_offset=time_offset)

    assert_type(timing, PrecisionWaveformTiming)
    assert not timing.has_timestamp
    assert timing.time_offset == time_offset
    assert not timing._has_sample_interval
    assert timing.sample_interval_mode == SampleIntervalMode.NONE


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
    assert timing.sample_interval_mode == SampleIntervalMode.REGULAR


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
    assert timing.sample_interval_mode == SampleIntervalMode.REGULAR


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
    assert timing.sample_interval_mode == SampleIntervalMode.REGULAR


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
    assert timing.sample_interval_mode == SampleIntervalMode.REGULAR


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
    assert timing.sample_interval_mode == SampleIntervalMode.IRREGULAR
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


###############################################################################
# magic methods
###############################################################################
@pytest.mark.xfail(raises=TypeError, reason="https://github.com/ni/hightime/issues/49")
@pytest.mark.parametrize(
    "value",
    [
        PrecisionWaveformTiming.create_with_no_interval(),
        PrecisionWaveformTiming.create_with_no_interval(ht.datetime(2025, 1, 1)),
        PrecisionWaveformTiming.create_with_no_interval(None, ht.timedelta(seconds=1)),
        PrecisionWaveformTiming.create_with_no_interval(
            ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)
        ),
        PrecisionWaveformTiming.create_with_regular_interval(ht.timedelta(milliseconds=1)),
        PrecisionWaveformTiming.create_with_regular_interval(
            ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 1)
        ),
        PrecisionWaveformTiming.create_with_regular_interval(
            ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)
        ),
        PrecisionWaveformTiming.create_with_irregular_interval(
            [ht.datetime(2025, 1, 1), ht.datetime(2025, 1, 2)]
        ),
    ],
)
def test___deep_copy___equality___equal(value: PrecisionWaveformTiming) -> None:
    other = deepcopy(value)

    assert value == other
    assert not (value != other)


@pytest.mark.parametrize(
    "lhs, rhs",
    [
        (
            PrecisionWaveformTiming.create_with_no_interval(
                ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)
            ),
            PrecisionWaveformTiming.create_with_no_interval(
                ht.datetime(2025, 1, 2), ht.timedelta(seconds=1)
            ),
        ),
        (
            PrecisionWaveformTiming.create_with_no_interval(
                ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)
            ),
            PrecisionWaveformTiming.create_with_no_interval(
                ht.datetime(2025, 1, 1), ht.timedelta(seconds=2)
            ),
        ),
        (
            PrecisionWaveformTiming.create_with_regular_interval(
                ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)
            ),
            PrecisionWaveformTiming.create_with_regular_interval(
                ht.timedelta(milliseconds=2), ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)
            ),
        ),
        (
            PrecisionWaveformTiming.create_with_regular_interval(
                ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)
            ),
            PrecisionWaveformTiming.create_with_regular_interval(
                ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 2), ht.timedelta(seconds=1)
            ),
        ),
        (
            PrecisionWaveformTiming.create_with_regular_interval(
                ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)
            ),
            PrecisionWaveformTiming.create_with_regular_interval(
                ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 1), ht.timedelta(seconds=2)
            ),
        ),
        (
            PrecisionWaveformTiming.create_with_irregular_interval(
                [ht.datetime(2025, 1, 1), ht.datetime(2025, 1, 2)]
            ),
            PrecisionWaveformTiming.create_with_irregular_interval(
                [ht.datetime(2025, 1, 3), ht.datetime(2025, 1, 2)]
            ),
        ),
    ],
)
def test___different_value___equality___not_equal(
    lhs: PrecisionWaveformTiming,
    rhs: PrecisionWaveformTiming,
) -> None:
    assert not (lhs == rhs)
    assert lhs != rhs


@pytest.mark.parametrize(
    "value, expected_repr",
    [
        (
            PrecisionWaveformTiming.create_with_no_interval(),
            "PrecisionWaveformTiming(SampleIntervalMode.NONE)",
        ),
        (
            PrecisionWaveformTiming.create_with_no_interval(ht.datetime(2025, 1, 1)),
            "PrecisionWaveformTiming(SampleIntervalMode.NONE, timestamp=hightime.datetime(2025, 1, 1, 0, 0))",
        ),
        (
            PrecisionWaveformTiming.create_with_no_interval(None, ht.timedelta(seconds=1)),
            "PrecisionWaveformTiming(SampleIntervalMode.NONE, time_offset=hightime.timedelta(seconds=1))",
        ),
        (
            PrecisionWaveformTiming.create_with_no_interval(
                ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)
            ),
            "PrecisionWaveformTiming(SampleIntervalMode.NONE, timestamp=hightime.datetime(2025, 1, 1, 0, 0), time_offset=hightime.timedelta(seconds=1))",
        ),
        (
            PrecisionWaveformTiming.create_with_regular_interval(ht.timedelta(milliseconds=1)),
            "PrecisionWaveformTiming(SampleIntervalMode.REGULAR, sample_interval=hightime.timedelta(microseconds=1000))",
        ),
        (
            PrecisionWaveformTiming.create_with_regular_interval(
                ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 1)
            ),
            "PrecisionWaveformTiming(SampleIntervalMode.REGULAR, timestamp=hightime.datetime(2025, 1, 1, 0, 0), sample_interval=hightime.timedelta(microseconds=1000))",
        ),
        (
            PrecisionWaveformTiming.create_with_regular_interval(
                ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)
            ),
            "PrecisionWaveformTiming(SampleIntervalMode.REGULAR, timestamp=hightime.datetime(2025, 1, 1, 0, 0), time_offset=hightime.timedelta(seconds=1), sample_interval=hightime.timedelta(microseconds=1000))",
        ),
        (
            PrecisionWaveformTiming.create_with_irregular_interval(
                [ht.datetime(2025, 1, 1), ht.datetime(2025, 1, 2)]
            ),
            "PrecisionWaveformTiming(SampleIntervalMode.IRREGULAR, timestamps=[hightime.datetime(2025, 1, 1, 0, 0), hightime.datetime(2025, 1, 2, 0, 0)])",
        ),
    ],
)
def test___various_values___repr___looks_ok(
    value: PrecisionWaveformTiming, expected_repr: str
) -> None:
    assert repr(value) == expected_repr
