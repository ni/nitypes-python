from __future__ import annotations

import datetime as dt
from copy import deepcopy

import hightime as ht
import pytest

from nitypes._typing import assert_type
from nitypes.waveform import PrecisionTiming, SampleIntervalMode


###############################################################################
# empty
###############################################################################
def test___empty___is_waveform_timing() -> None:
    assert_type(PrecisionTiming.empty, PrecisionTiming)
    assert isinstance(PrecisionTiming.empty, PrecisionTiming)


def test___empty___no_timestamp() -> None:
    assert not PrecisionTiming.empty.has_timestamp
    with pytest.raises(RuntimeError) as exc:
        _ = PrecisionTiming.empty.timestamp

    assert exc.value.args[0] == "The waveform timing does not have a timestamp."


def test___empty___no_start_time() -> None:
    with pytest.raises(RuntimeError) as exc:
        _ = PrecisionTiming.empty.start_time

    assert exc.value.args[0] == "The waveform timing does not have a timestamp."


def test___empty___default_time_offset() -> None:
    assert PrecisionTiming.empty.time_offset == ht.timedelta()


def test___empty___no_sample_interval() -> None:
    assert PrecisionTiming.empty._sample_interval is None
    with pytest.raises(RuntimeError) as exc:
        _ = PrecisionTiming.empty.sample_interval

    assert exc.value.args[0] == "The waveform timing does not have a sample interval."


def test___empty___sample_interval_mode_none() -> None:
    assert PrecisionTiming.empty.sample_interval_mode == SampleIntervalMode.NONE


###############################################################################
# create_with_no_interval
###############################################################################
def test___no_args___create_with_no_interval___creates_empty_waveform_timing() -> None:
    timing = PrecisionTiming.create_with_no_interval()

    assert_type(timing, PrecisionTiming)
    assert not timing.has_timestamp
    assert timing.time_offset == ht.timedelta()
    assert timing._sample_interval is None
    assert timing.sample_interval_mode == SampleIntervalMode.NONE


def test___timestamp___create_with_no_interval___creates_waveform_timing_with_timestamp() -> None:
    timestamp = ht.datetime.now(dt.timezone.utc)
    timing = PrecisionTiming.create_with_no_interval(timestamp)

    assert_type(timing, PrecisionTiming)
    assert timing.timestamp == timestamp
    assert timing.time_offset == ht.timedelta()
    assert timing._sample_interval is None
    assert timing.sample_interval_mode == SampleIntervalMode.NONE


def test___timestamp_and_time_offset___create_with_no_interval___creates_waveform_timing_with_timestamp_and_time_offset() -> (
    None
):
    timestamp = ht.datetime.now(dt.timezone.utc)
    time_offset = ht.timedelta(seconds=1.23)
    timing = PrecisionTiming.create_with_no_interval(timestamp, time_offset)

    assert_type(timing, PrecisionTiming)
    assert timing.timestamp == timestamp
    assert timing.time_offset == time_offset
    assert timing._sample_interval is None
    assert timing.sample_interval_mode == SampleIntervalMode.NONE


def test___time_offset___create_with_no_interval___creates_waveform_timing_with_time_offset() -> (
    None
):
    time_offset = ht.timedelta(seconds=1.23)
    timing = PrecisionTiming.create_with_no_interval(time_offset=time_offset)

    assert_type(timing, PrecisionTiming)
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
    sample_interval = ht.timedelta(milliseconds=1)

    timing = PrecisionTiming.create_with_regular_interval(sample_interval)

    assert_type(timing, PrecisionTiming)
    assert not timing.has_timestamp
    assert timing.time_offset == ht.timedelta()
    assert timing.sample_interval == sample_interval
    assert timing.sample_interval_mode == SampleIntervalMode.REGULAR


def test___sample_interval_and_timestamp___create_with_regular_interval___creates_waveform_timing_with_sample_interval_and_timestamp() -> (
    None
):
    sample_interval = ht.timedelta(milliseconds=1)
    timestamp = ht.datetime.now(dt.timezone.utc)

    timing = PrecisionTiming.create_with_regular_interval(sample_interval, timestamp)

    assert_type(timing, PrecisionTiming)
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

    timing = PrecisionTiming.create_with_regular_interval(sample_interval, timestamp, time_offset)

    assert_type(timing, PrecisionTiming)
    assert timing.timestamp == timestamp
    assert timing.time_offset == time_offset
    assert timing.sample_interval == sample_interval
    assert timing.sample_interval_mode == SampleIntervalMode.REGULAR


def test___sample_interval_and_time_offset___create_with_regular_interval___creates_waveform_timing_with_sample_interval_and_time_offset() -> (
    None
):
    sample_interval = ht.timedelta(milliseconds=1)
    time_offset = ht.timedelta(seconds=1.23)

    timing = PrecisionTiming.create_with_regular_interval(sample_interval, time_offset=time_offset)

    assert_type(timing, PrecisionTiming)
    assert not timing.has_timestamp
    assert timing.time_offset == time_offset
    assert timing.sample_interval == sample_interval
    assert timing.sample_interval_mode == SampleIntervalMode.REGULAR


###############################################################################
# create_with_irregular_interval
###############################################################################
@pytest.mark.parametrize(
    "time_offsets",
    [
        [],
        [ht.timedelta(0)],
        [ht.timedelta(0), ht.timedelta(0)],
        [ht.timedelta(0), ht.timedelta(1)],
        [ht.timedelta(0), ht.timedelta(1), ht.timedelta(2)],
        [
            ht.timedelta(0),
            ht.timedelta(1),
            ht.timedelta(2),
            ht.timedelta(3),
        ],
        [
            ht.timedelta(3),
            ht.timedelta(2),
            ht.timedelta(1),
            ht.timedelta(0),
        ],
        [ht.timedelta(0, 0, 1), ht.timedelta(0, 1, 0), ht.timedelta(1, 0, 0)],
        [ht.timedelta(1, 0, 0), ht.timedelta(0, 1, 0), ht.timedelta(0, 0, 1)],
        [
            ht.timedelta(0),
            ht.timedelta(1),
            ht.timedelta(1),
            ht.timedelta(2),
        ],
    ],
)
def test___monotonic_timestamps___create_with_irregular_interval___creates_waveform_timing_with_timestamps(
    time_offsets: list[ht.timedelta],
) -> None:
    start_time = ht.datetime.now(dt.timezone.utc)
    timestamps = [start_time + offset for offset in time_offsets]

    timing = PrecisionTiming.create_with_irregular_interval(timestamps)

    assert_type(timing, PrecisionTiming)
    assert timing.sample_interval_mode == SampleIntervalMode.IRREGULAR
    assert timing._timestamps == timestamps


@pytest.mark.parametrize(
    "time_offsets",
    [
        [ht.timedelta(0), ht.timedelta(1), ht.timedelta(0)],
        [ht.timedelta(1), ht.timedelta(0), ht.timedelta(1)],
    ],
)
def test___non_monotonic_timestamps___create_with_irregular_interval___raises_value_error(
    time_offsets: list[ht.timedelta],
) -> None:
    start_time = ht.datetime.now(dt.timezone.utc)
    timestamps = [start_time + offset for offset in time_offsets]

    with pytest.raises(ValueError) as exc:
        _ = PrecisionTiming.create_with_irregular_interval(timestamps)

    assert exc.value.args[0].startswith("The timestamps must be in ascending or descending order.")


def test___timestamps_tuple___create_with_irregular_interval___creates_waveform_timing_with_timestamps() -> (
    None
):
    start_time = ht.datetime.now(dt.timezone.utc)
    timestamps = (start_time,)

    timing = PrecisionTiming.create_with_irregular_interval(timestamps)

    assert_type(timing, PrecisionTiming)
    assert timing.sample_interval_mode == SampleIntervalMode.IRREGULAR
    assert timing._timestamps == list(timestamps)


###############################################################################
# get_timestamps
###############################################################################
def test___no_interval___get_timestamps___raises_runtime_error() -> None:
    start_time = ht.datetime.now(dt.timezone.utc)
    timing = PrecisionTiming.create_with_no_interval(start_time)

    with pytest.raises(RuntimeError) as exc:
        _ = timing.get_timestamps(0, 5)

    assert exc.value.args[0].startswith(
        "The waveform timing does not have valid timestamp information."
    )


def test___regular_interval___get_timestamps___gets_timestamps() -> None:
    start_time = ht.datetime.now(dt.timezone.utc)
    sample_interval = ht.timedelta(milliseconds=1)
    timing = PrecisionTiming.create_with_regular_interval(sample_interval, start_time)

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
    timing = PrecisionTiming.create_with_irregular_interval(timestamps)

    assert list(timing.get_timestamps(0, 10)) == timestamps


def test___irregular_interval_subset___get_timestamps___gets_timestamps() -> None:
    start_time = ht.datetime.now(dt.timezone.utc)
    sample_interval = ht.timedelta(milliseconds=1)
    timestamps = [start_time + i * sample_interval for i in range(10)]
    timing = PrecisionTiming.create_with_irregular_interval(timestamps)

    assert list(timing.get_timestamps(3, 4)) == timestamps[3:7]


###############################################################################
# magic methods
###############################################################################
@pytest.mark.xfail(raises=TypeError, reason="https://github.com/ni/hightime/issues/49")
@pytest.mark.parametrize(
    "value",
    [
        PrecisionTiming.create_with_no_interval(),
        PrecisionTiming.create_with_no_interval(ht.datetime(2025, 1, 1)),
        PrecisionTiming.create_with_no_interval(None, ht.timedelta(seconds=1)),
        PrecisionTiming.create_with_no_interval(ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)),
        PrecisionTiming.create_with_regular_interval(ht.timedelta(milliseconds=1)),
        PrecisionTiming.create_with_regular_interval(
            ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 1)
        ),
        PrecisionTiming.create_with_regular_interval(
            ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)
        ),
        PrecisionTiming.create_with_irregular_interval(
            [ht.datetime(2025, 1, 1), ht.datetime(2025, 1, 2)]
        ),
    ],
)
def test___deep_copy___equality___equal(value: PrecisionTiming) -> None:
    other = deepcopy(value)

    assert value == other
    assert not (value != other)


@pytest.mark.parametrize(
    "lhs, rhs",
    [
        (
            PrecisionTiming.create_with_no_interval(
                ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)
            ),
            PrecisionTiming.create_with_no_interval(
                ht.datetime(2025, 1, 2), ht.timedelta(seconds=1)
            ),
        ),
        (
            PrecisionTiming.create_with_no_interval(
                ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)
            ),
            PrecisionTiming.create_with_no_interval(
                ht.datetime(2025, 1, 1), ht.timedelta(seconds=2)
            ),
        ),
        (
            PrecisionTiming.create_with_regular_interval(
                ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)
            ),
            PrecisionTiming.create_with_regular_interval(
                ht.timedelta(milliseconds=2), ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)
            ),
        ),
        (
            PrecisionTiming.create_with_regular_interval(
                ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)
            ),
            PrecisionTiming.create_with_regular_interval(
                ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 2), ht.timedelta(seconds=1)
            ),
        ),
        (
            PrecisionTiming.create_with_regular_interval(
                ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)
            ),
            PrecisionTiming.create_with_regular_interval(
                ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 1), ht.timedelta(seconds=2)
            ),
        ),
        (
            PrecisionTiming.create_with_irregular_interval(
                [ht.datetime(2025, 1, 1), ht.datetime(2025, 1, 2)]
            ),
            PrecisionTiming.create_with_irregular_interval(
                [ht.datetime(2025, 1, 3), ht.datetime(2025, 1, 2)]
            ),
        ),
    ],
)
def test___different_value___equality___not_equal(
    lhs: PrecisionTiming,
    rhs: PrecisionTiming,
) -> None:
    assert not (lhs == rhs)
    assert lhs != rhs


@pytest.mark.parametrize(
    "value, expected_repr",
    [
        (
            PrecisionTiming.create_with_no_interval(),
            "nitypes.waveform.PrecisionTiming(nitypes.waveform.SampleIntervalMode.NONE)",
        ),
        (
            PrecisionTiming.create_with_no_interval(ht.datetime(2025, 1, 1)),
            "nitypes.waveform.PrecisionTiming(nitypes.waveform.SampleIntervalMode.NONE, timestamp=hightime.datetime(2025, 1, 1, 0, 0))",
        ),
        (
            PrecisionTiming.create_with_no_interval(None, ht.timedelta(seconds=1)),
            "nitypes.waveform.PrecisionTiming(nitypes.waveform.SampleIntervalMode.NONE, time_offset=hightime.timedelta(seconds=1))",
        ),
        (
            PrecisionTiming.create_with_no_interval(
                ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)
            ),
            "nitypes.waveform.PrecisionTiming(nitypes.waveform.SampleIntervalMode.NONE, timestamp=hightime.datetime(2025, 1, 1, 0, 0), time_offset=hightime.timedelta(seconds=1))",
        ),
        (
            PrecisionTiming.create_with_no_interval(ht.datetime(2025, 1, 1), ht.timedelta()),
            "nitypes.waveform.PrecisionTiming(nitypes.waveform.SampleIntervalMode.NONE, timestamp=hightime.datetime(2025, 1, 1, 0, 0), time_offset=hightime.timedelta())",
        ),
        (
            PrecisionTiming.create_with_regular_interval(ht.timedelta(milliseconds=1)),
            "nitypes.waveform.PrecisionTiming(nitypes.waveform.SampleIntervalMode.REGULAR, sample_interval=hightime.timedelta(microseconds=1000))",
        ),
        (
            PrecisionTiming.create_with_regular_interval(
                ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 1)
            ),
            "nitypes.waveform.PrecisionTiming(nitypes.waveform.SampleIntervalMode.REGULAR, timestamp=hightime.datetime(2025, 1, 1, 0, 0), sample_interval=hightime.timedelta(microseconds=1000))",
        ),
        (
            PrecisionTiming.create_with_regular_interval(
                ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)
            ),
            "nitypes.waveform.PrecisionTiming(nitypes.waveform.SampleIntervalMode.REGULAR, timestamp=hightime.datetime(2025, 1, 1, 0, 0), time_offset=hightime.timedelta(seconds=1), sample_interval=hightime.timedelta(microseconds=1000))",
        ),
        (
            PrecisionTiming.create_with_irregular_interval(
                [ht.datetime(2025, 1, 1), ht.datetime(2025, 1, 2)]
            ),
            "nitypes.waveform.PrecisionTiming(nitypes.waveform.SampleIntervalMode.IRREGULAR, timestamps=[hightime.datetime(2025, 1, 1, 0, 0), hightime.datetime(2025, 1, 2, 0, 0)])",
        ),
    ],
)
def test___various_values___repr___looks_ok(value: PrecisionTiming, expected_repr: str) -> None:
    assert repr(value) == expected_repr


###############################################################################
# _append_timing
###############################################################################
@pytest.mark.parametrize(
    "left_offsets, right_offsets",
    [
        ([], []),
        ([ht.timedelta(0)], []),
        ([ht.timedelta(0), ht.timedelta(1)], []),
        ([ht.timedelta(0), ht.timedelta(1), ht.timedelta(2)], []),
        ([ht.timedelta(0)], [ht.timedelta(1)]),
        ([ht.timedelta(0)], [ht.timedelta(1), ht.timedelta(2)]),
        (
            [ht.timedelta(0), ht.timedelta(1)],
            [ht.timedelta(2), ht.timedelta(3)],
        ),
        (
            [ht.timedelta(3), ht.timedelta(2)],
            [ht.timedelta(1), ht.timedelta(0)],
        ),
        (
            [ht.timedelta(0), ht.timedelta(1)],
            [ht.timedelta(1), ht.timedelta(2)],
        ),
        (
            [ht.timedelta(2), ht.timedelta(1)],
            [ht.timedelta(1), ht.timedelta(0)],
        ),
    ],
)
def test___monotonic_timestamps___append_timing___appends_timestamps(
    left_offsets: list[ht.timedelta],
    right_offsets: list[ht.timedelta],
) -> None:
    start_time = ht.datetime.now(dt.timezone.utc)
    left_timestamps = [start_time + offset for offset in left_offsets]
    right_timestamps = [start_time + offset for offset in right_offsets]
    left_timing = PrecisionTiming.create_with_irregular_interval(left_timestamps)
    right_timing = PrecisionTiming.create_with_irregular_interval(right_timestamps)

    new_timing = left_timing._append_timing(right_timing)

    assert_type(new_timing, PrecisionTiming)
    assert isinstance(new_timing, PrecisionTiming)
    assert new_timing._timestamps == left_timestamps + right_timestamps


@pytest.mark.parametrize(
    "left_offsets, right_offsets",
    [
        ([ht.timedelta(0)], [ht.timedelta(1), ht.timedelta(0)]),
        ([ht.timedelta(1)], [ht.timedelta(0), ht.timedelta(2)]),
        ([ht.timedelta(0), ht.timedelta(1)], [ht.timedelta(0)]),
        ([ht.timedelta(1), ht.timedelta(0)], [ht.timedelta(1)]),
        ([ht.timedelta(0), ht.timedelta(1)], [ht.timedelta(2), ht.timedelta(0)]),
    ],
)
def test___non_monotonic_timestamps___append_timing___raises_value_error(
    left_offsets: list[ht.timedelta],
    right_offsets: list[ht.timedelta],
) -> None:
    start_time = ht.datetime.now(dt.timezone.utc)
    left_timestamps = [start_time + offset for offset in left_offsets]
    right_timestamps = [start_time + offset for offset in right_offsets]
    left_timing = PrecisionTiming.create_with_irregular_interval(left_timestamps)
    right_timing = PrecisionTiming.create_with_irregular_interval(right_timestamps)

    with pytest.raises(ValueError) as exc:
        _ = left_timing._append_timing(right_timing)

    assert exc.value.args[0].startswith("The timestamps must be in ascending or descending order.")
