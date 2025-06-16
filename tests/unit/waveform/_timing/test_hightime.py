from __future__ import annotations

import copy
import datetime as dt
import pickle
from typing import Any

import hightime as ht
import pytest
from typing_extensions import assert_type

from nitypes.waveform import SampleIntervalMode, Timing
from tests.unit.waveform._timing._utils import assert_deep_copy, assert_shallow_copy


###############################################################################
# empty
###############################################################################
def test___empty___is_timing() -> None:
    assert_type(Timing.empty, Timing[dt.datetime, dt.timedelta, dt.timedelta])
    assert isinstance(Timing.empty, Timing)


def test___empty___no_timestamp() -> None:
    assert not Timing.empty.has_timestamp
    with pytest.raises(RuntimeError) as exc:
        _ = Timing.empty.timestamp

    assert exc.value.args[0] == "The waveform timing does not have a timestamp."


def test___empty___no_start_time() -> None:
    with pytest.raises(RuntimeError) as exc:
        _ = Timing.empty.start_time

    assert exc.value.args[0] == "The waveform timing does not have a timestamp."


def test___empty___no_time_offset() -> None:
    assert not Timing.empty.has_time_offset
    with pytest.raises(RuntimeError) as exc:
        _ = Timing.empty.time_offset

    assert exc.value.args[0] == "The waveform timing does not have a time offset."


def test___empty___no_sample_interval() -> None:
    assert not Timing.empty.has_sample_interval
    with pytest.raises(RuntimeError) as exc:
        _ = Timing.empty.sample_interval

    assert exc.value.args[0] == "The waveform timing does not have a sample interval."


def test___empty___sample_interval_mode_none() -> None:
    assert Timing.empty.sample_interval_mode == SampleIntervalMode.NONE


###############################################################################
# create_with_no_interval
###############################################################################
def test___no_args___create_with_no_interval___creates_empty_waveform_timing() -> None:
    timing = Timing.create_with_no_interval()

    assert_type(timing, Timing[dt.datetime, dt.timedelta, dt.timedelta])
    assert not timing.has_timestamp
    assert not timing.has_time_offset
    assert not timing.has_sample_interval
    assert timing.sample_interval_mode == SampleIntervalMode.NONE


def test___timestamp___create_with_no_interval___creates_waveform_timing_with_timestamp() -> None:
    timestamp = ht.datetime.now(dt.timezone.utc)
    timing = Timing.create_with_no_interval(timestamp)

    assert_type(timing, Timing[ht.datetime, dt.timedelta, dt.timedelta])
    assert timing.timestamp == timestamp
    assert not timing.has_time_offset
    assert not timing.has_sample_interval
    assert timing.sample_interval_mode == SampleIntervalMode.NONE


def test___timestamp_and_time_offset___create_with_no_interval___creates_waveform_timing_with_timestamp_and_time_offset() -> (
    None
):
    timestamp = ht.datetime.now(dt.timezone.utc)
    time_offset = ht.timedelta(seconds=1.23)
    timing = Timing.create_with_no_interval(timestamp, time_offset)

    assert_type(timing, Timing[ht.datetime, ht.timedelta, dt.timedelta])
    assert timing.timestamp == timestamp
    assert timing.time_offset == time_offset
    assert not timing.has_sample_interval
    assert timing.sample_interval_mode == SampleIntervalMode.NONE


def test___time_offset___create_with_no_interval___creates_waveform_timing_with_time_offset() -> (
    None
):
    time_offset = ht.timedelta(seconds=1.23)
    timing = Timing.create_with_no_interval(time_offset=time_offset)

    assert_type(timing, Timing[dt.datetime, ht.timedelta, dt.timedelta])
    assert not timing.has_timestamp
    assert timing.time_offset == time_offset
    assert not timing.has_sample_interval
    assert timing.sample_interval_mode == SampleIntervalMode.NONE


###############################################################################
# create_with_regular_interval
###############################################################################
def test___sample_interval___create_with_regular_interval___creates_waveform_timing_with_sample_interval() -> (
    None
):
    sample_interval = ht.timedelta(milliseconds=1)

    timing = Timing.create_with_regular_interval(sample_interval)

    assert_type(timing, Timing[dt.datetime, dt.timedelta, ht.timedelta])
    assert not timing.has_timestamp
    assert not timing.has_time_offset
    assert timing.sample_interval == sample_interval
    assert timing.sample_interval_mode == SampleIntervalMode.REGULAR


def test___sample_interval_and_timestamp___create_with_regular_interval___creates_waveform_timing_with_sample_interval_and_timestamp() -> (
    None
):
    sample_interval = ht.timedelta(milliseconds=1)
    timestamp = ht.datetime.now(dt.timezone.utc)

    timing = Timing.create_with_regular_interval(sample_interval, timestamp)

    assert_type(timing, Timing[ht.datetime, dt.timedelta, ht.timedelta])
    assert timing.timestamp == timestamp
    assert not timing.has_time_offset
    assert timing.sample_interval == sample_interval
    assert timing.sample_interval_mode == SampleIntervalMode.REGULAR


def test___sample_interval_timestamp_and_time_offset___create_with_regular_interval___creates_waveform_timing_with_sample_interval_timestamp_and_time_offset() -> (
    None
):
    sample_interval = ht.timedelta(milliseconds=1)
    timestamp = ht.datetime.now(dt.timezone.utc)
    time_offset = ht.timedelta(seconds=1.23)

    timing = Timing.create_with_regular_interval(sample_interval, timestamp, time_offset)

    assert_type(timing, Timing[ht.datetime, ht.timedelta, ht.timedelta])
    assert timing.timestamp == timestamp
    assert timing.time_offset == time_offset
    assert timing.sample_interval == sample_interval
    assert timing.sample_interval_mode == SampleIntervalMode.REGULAR


def test___sample_interval_and_time_offset___create_with_regular_interval___creates_waveform_timing_with_sample_interval_and_time_offset() -> (
    None
):
    sample_interval = ht.timedelta(milliseconds=1)
    time_offset = ht.timedelta(seconds=1.23)

    timing = Timing.create_with_regular_interval(sample_interval, time_offset=time_offset)

    assert_type(timing, Timing[dt.datetime, ht.timedelta, ht.timedelta])
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

    timing = Timing.create_with_irregular_interval(timestamps)

    assert_type(timing, Timing[ht.datetime, dt.timedelta, dt.timedelta])
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
        _ = Timing.create_with_irregular_interval(timestamps)

    assert exc.value.args[0].startswith("The timestamps must be in ascending or descending order.")


def test___timestamps_tuple___create_with_irregular_interval___creates_waveform_timing_with_timestamps() -> (
    None
):
    start_time = ht.datetime.now(dt.timezone.utc)
    timestamps = (start_time,)

    timing = Timing.create_with_irregular_interval(timestamps)

    assert_type(timing, Timing[ht.datetime, dt.timedelta, dt.timedelta])
    assert timing.sample_interval_mode == SampleIntervalMode.IRREGULAR
    assert timing._timestamps == list(timestamps)


###############################################################################
# get_timestamps
###############################################################################
def test___no_interval___get_timestamps___raises_runtime_error() -> None:
    start_time = ht.datetime.now(dt.timezone.utc)
    timing = Timing.create_with_no_interval(start_time)

    with pytest.raises(RuntimeError) as exc:
        _ = timing.get_timestamps(0, 5)

    assert exc.value.args[0].startswith(
        "The waveform timing does not have valid timestamp information."
    )


def test___regular_interval___get_timestamps___gets_timestamps() -> None:
    start_time = ht.datetime.now(dt.timezone.utc)
    sample_interval = ht.timedelta(milliseconds=1)
    timing = Timing.create_with_regular_interval(sample_interval, start_time)

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
    timing = Timing.create_with_irregular_interval(timestamps)

    assert list(timing.get_timestamps(0, 10)) == timestamps


def test___irregular_interval_subset___get_timestamps___gets_timestamps() -> None:
    start_time = ht.datetime.now(dt.timezone.utc)
    sample_interval = ht.timedelta(milliseconds=1)
    timestamps = [start_time + i * sample_interval for i in range(10)]
    timing = Timing.create_with_irregular_interval(timestamps)

    assert list(timing.get_timestamps(3, 4)) == timestamps[3:7]


###############################################################################
# magic methods
###############################################################################
@pytest.mark.parametrize(
    "left, right",
    [
        (Timing.create_with_no_interval(), Timing.create_with_no_interval()),
        (
            Timing.create_with_no_interval(ht.datetime(2025, 1, 1)),
            Timing.create_with_no_interval(ht.datetime(2025, 1, 1)),
        ),
        (
            Timing.create_with_no_interval(None, ht.timedelta(seconds=1)),
            Timing.create_with_no_interval(None, ht.timedelta(seconds=1)),
        ),
        (
            Timing.create_with_no_interval(ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)),
            Timing.create_with_no_interval(ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)),
        ),
        (
            Timing.create_with_regular_interval(ht.timedelta(milliseconds=1)),
            Timing.create_with_regular_interval(ht.timedelta(milliseconds=1)),
        ),
        (
            Timing.create_with_regular_interval(
                ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 1)
            ),
            Timing.create_with_regular_interval(
                ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 1)
            ),
        ),
        (
            Timing.create_with_regular_interval(
                ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)
            ),
            Timing.create_with_regular_interval(
                ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)
            ),
        ),
        (
            Timing.create_with_irregular_interval(
                [ht.datetime(2025, 1, 1), ht.datetime(2025, 1, 2)]
            ),
            Timing.create_with_irregular_interval(
                [ht.datetime(2025, 1, 1), ht.datetime(2025, 1, 2)]
            ),
        ),
    ],
)
def test___same_value___equality___equal(
    left: Timing[Any, Any, Any], right: Timing[Any, Any, Any]
) -> None:
    assert left == right
    assert not (left != right)


@pytest.mark.parametrize(
    "left, right",
    [
        (
            Timing.create_with_no_interval(ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)),
            Timing.create_with_no_interval(ht.datetime(2025, 1, 2), ht.timedelta(seconds=1)),
        ),
        (
            Timing.create_with_no_interval(ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)),
            Timing.create_with_no_interval(ht.datetime(2025, 1, 1), ht.timedelta(seconds=2)),
        ),
        (
            Timing.create_with_regular_interval(
                ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)
            ),
            Timing.create_with_regular_interval(
                ht.timedelta(milliseconds=2), ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)
            ),
        ),
        (
            Timing.create_with_regular_interval(
                ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)
            ),
            Timing.create_with_regular_interval(
                ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 2), ht.timedelta(seconds=1)
            ),
        ),
        (
            Timing.create_with_regular_interval(
                ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)
            ),
            Timing.create_with_regular_interval(
                ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 1), ht.timedelta(seconds=2)
            ),
        ),
        (
            Timing.create_with_irregular_interval(
                [ht.datetime(2025, 1, 1), ht.datetime(2025, 1, 2)]
            ),
            Timing.create_with_irregular_interval(
                [ht.datetime(2025, 1, 3), ht.datetime(2025, 1, 2)]
            ),
        ),
    ],
)
def test___different_value___equality___not_equal(
    left: Timing[Any, Any, Any],
    right: Timing[Any, Any, Any],
) -> None:
    assert not (left == right)
    assert left != right


@pytest.mark.parametrize(
    "value, expected_repr",
    [
        (
            Timing.create_with_no_interval(),
            "nitypes.waveform.Timing(nitypes.waveform.SampleIntervalMode.NONE)",
        ),
        (
            Timing.create_with_no_interval(ht.datetime(2025, 1, 1)),
            "nitypes.waveform.Timing(nitypes.waveform.SampleIntervalMode.NONE, timestamp=hightime.datetime(2025, 1, 1, 0, 0))",
        ),
        (
            Timing.create_with_no_interval(None, ht.timedelta(seconds=1)),
            "nitypes.waveform.Timing(nitypes.waveform.SampleIntervalMode.NONE, time_offset=hightime.timedelta(seconds=1))",
        ),
        (
            Timing.create_with_no_interval(ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)),
            "nitypes.waveform.Timing(nitypes.waveform.SampleIntervalMode.NONE, timestamp=hightime.datetime(2025, 1, 1, 0, 0), time_offset=hightime.timedelta(seconds=1))",
        ),
        (
            Timing.create_with_no_interval(ht.datetime(2025, 1, 1), ht.timedelta()),
            "nitypes.waveform.Timing(nitypes.waveform.SampleIntervalMode.NONE, timestamp=hightime.datetime(2025, 1, 1, 0, 0), time_offset=hightime.timedelta())",
        ),
        (
            Timing.create_with_regular_interval(ht.timedelta(milliseconds=1)),
            "nitypes.waveform.Timing(nitypes.waveform.SampleIntervalMode.REGULAR, sample_interval=hightime.timedelta(microseconds=1000))",
        ),
        (
            Timing.create_with_regular_interval(
                ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 1)
            ),
            "nitypes.waveform.Timing(nitypes.waveform.SampleIntervalMode.REGULAR, timestamp=hightime.datetime(2025, 1, 1, 0, 0), sample_interval=hightime.timedelta(microseconds=1000))",
        ),
        (
            Timing.create_with_regular_interval(
                ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)
            ),
            "nitypes.waveform.Timing(nitypes.waveform.SampleIntervalMode.REGULAR, timestamp=hightime.datetime(2025, 1, 1, 0, 0), time_offset=hightime.timedelta(seconds=1), sample_interval=hightime.timedelta(microseconds=1000))",
        ),
        (
            Timing.create_with_irregular_interval(
                [ht.datetime(2025, 1, 1), ht.datetime(2025, 1, 2)]
            ),
            "nitypes.waveform.Timing(nitypes.waveform.SampleIntervalMode.IRREGULAR, timestamps=[hightime.datetime(2025, 1, 1, 0, 0), hightime.datetime(2025, 1, 2, 0, 0)])",
        ),
    ],
)
def test___various_values___repr___looks_ok(
    value: Timing[Any, Any, Any], expected_repr: str
) -> None:
    assert repr(value) == expected_repr


_VARIOUS_VALUES = [
    Timing.create_with_no_interval(),
    Timing.create_with_no_interval(ht.datetime(2025, 1, 1)),
    Timing.create_with_no_interval(None, ht.timedelta(seconds=1)),
    Timing.create_with_no_interval(ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)),
    Timing.create_with_regular_interval(ht.timedelta(milliseconds=1)),
    Timing.create_with_regular_interval(ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 1)),
    Timing.create_with_regular_interval(
        ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)
    ),
    Timing.create_with_irregular_interval([ht.datetime(2025, 1, 1), ht.datetime(2025, 1, 2)]),
]


@pytest.mark.parametrize("value", _VARIOUS_VALUES)
def test___various_values___copy___makes_shallow_copy(value: Timing[Any, Any, Any]) -> None:
    new_value = copy.copy(value)

    assert_shallow_copy(new_value, value)


@pytest.mark.parametrize("value", _VARIOUS_VALUES)
def test___various_values___deepcopy___makes_deep_copy(value: Timing[Any, Any, Any]) -> None:
    new_value = copy.deepcopy(value)

    assert_deep_copy(new_value, value)


@pytest.mark.parametrize("value", _VARIOUS_VALUES)
def test___various_values___pickle_unpickle___makes_deep_copy(
    value: Timing[Any, Any, Any],
) -> None:
    new_value = pickle.loads(pickle.dumps(value))

    assert_deep_copy(new_value, value)


def test___timing___pickle___references_public_modules() -> None:
    value = Timing.create_with_regular_interval(ht.timedelta(milliseconds=1))

    value_bytes = pickle.dumps(value)

    assert b"nitypes.waveform" in value_bytes
    assert b"nitypes.waveform._timing" not in value_bytes


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
    left_timing = Timing.create_with_irregular_interval(left_timestamps)
    right_timing = Timing.create_with_irregular_interval(right_timestamps)

    new_timing = left_timing._append_timing(right_timing)

    assert_type(new_timing, Timing[ht.datetime, dt.timedelta, dt.timedelta])
    assert isinstance(new_timing, Timing)
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
    left_timing = Timing.create_with_irregular_interval(left_timestamps)
    right_timing = Timing.create_with_irregular_interval(right_timestamps)

    with pytest.raises(ValueError) as exc:
        _ = left_timing._append_timing(right_timing)

    assert exc.value.args[0].startswith("The timestamps must be in ascending or descending order.")
