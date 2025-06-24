from __future__ import annotations

import array
import copy
import datetime as dt
import pickle
import weakref
from typing import Any, Union

import hightime as ht
import numpy as np
import numpy.typing as npt
import pytest
from typing_extensions import assert_type

import nitypes.bintime as bt
from nitypes.waveform import (
    DigitalState,
    DigitalWaveform,
    SampleIntervalMode,
    Timing,
    TimingMismatchError,
    TimingMismatchWarning,
)


###############################################################################
# create
###############################################################################
def test___no_args___create___creates_empty_waveform_with_default_signal_count_and_default_dtype() -> (
    None
):
    waveform = DigitalWaveform()

    assert waveform.sample_count == waveform.capacity == len(waveform.data) == 0
    assert waveform.signal_count == 1
    assert waveform.dtype == np.uint8
    assert_type(waveform, DigitalWaveform[np.uint8])


def test___sample_count___create___creates_waveform_with_sample_count_default_signal_count_and_default_dtype() -> (
    None
):
    waveform = DigitalWaveform(10)

    assert waveform.sample_count == waveform.capacity == len(waveform.data) == 10
    assert waveform.signal_count == 1
    assert waveform.dtype == np.uint8
    assert_type(waveform, DigitalWaveform[np.uint8])


def test___sample_count_signal_count_and_dtype___create___creates_waveform_with_sample_count_and_dtype() -> (
    None
):
    waveform = DigitalWaveform(10, 3, np.bool)

    assert waveform.sample_count == waveform.capacity == len(waveform.data) == 10
    assert waveform.signal_count == 3
    assert waveform.dtype == np.bool
    # https://github.com/numpy/numpy/issues/29245 - TYP: mypy returns dtype of
    # np.bool[Literal[False]] for array of bools
    assert_type(waveform, DigitalWaveform[np.bool])  # type: ignore[assert-type]


def test___sample_count_and_dtype_str___create___creates_waveform_with_sample_count_and_dtype() -> (
    None
):
    waveform = DigitalWaveform(10, dtype="i1")

    assert waveform.sample_count == waveform.capacity == len(waveform.data) == 10
    assert waveform.signal_count == 1
    assert waveform.dtype == np.int8
    assert_type(waveform, DigitalWaveform[Any])  # dtype not inferred from string


def test___sample_count_and_dtype_any___create___creates_waveform_with_sample_count_and_dtype() -> (
    None
):
    dtype: np.dtype[Any] = np.dtype(np.int8)
    waveform = DigitalWaveform(10, dtype=dtype)

    assert waveform.sample_count == waveform.capacity == len(waveform.data) == 10
    assert waveform.signal_count == 1
    assert waveform.dtype == np.int8
    assert_type(waveform, DigitalWaveform[Any])  # dtype not inferred from np.dtype[Any]


def test___sample_count_dtype_and_capacity___create___creates_waveform_with_sample_count_dtype_and_capacity() -> (
    None
):
    waveform = DigitalWaveform(10, dtype=np.int8, capacity=20)

    assert waveform.sample_count == len(waveform.data) == 10
    assert waveform.signal_count == 1
    assert waveform.capacity == 20
    assert waveform.dtype == np.int8
    assert_type(waveform, DigitalWaveform[np.int8])


@pytest.mark.parametrize("dtype", [np.complex128, np.str_, np.void, "u1, u1"])
def test___sample_count_and_unsupported_dtype___create___raises_type_error(
    dtype: npt.DTypeLike,
) -> None:
    with pytest.raises(TypeError) as exc:
        _ = DigitalWaveform(10, dtype=dtype)

    assert exc.value.args[0].startswith("The requested data type is not supported.")


def test___dtype_str_with_unsupported_tdata_hint___create___mypy_type_var_warning() -> None:
    waveform1: DigitalWaveform[np.complex128] = DigitalWaveform(dtype="u1")  # type: ignore[type-var]
    waveform2: DigitalWaveform[np.str_] = DigitalWaveform(dtype="u1")  # type: ignore[type-var]
    waveform3: DigitalWaveform[np.void] = DigitalWaveform(dtype="u1")  # type: ignore[type-var]
    _ = waveform1, waveform2, waveform3


def test___dtype_str_with_tdata_hint___create___narrows_tdata() -> None:
    waveform: DigitalWaveform[np.uint8] = DigitalWaveform(dtype="u1")

    assert_type(waveform, DigitalWaveform[np.uint8])


@pytest.mark.parametrize(
    "default_value",
    [
        False,
        True,
        0,
        1,
        3,
        DigitalState.FORCE_DOWN,
        DigitalState.FORCE_UP,
        DigitalState.COMPARE_VALID,
    ],
)
def test___default_value___create___creates_waveform_with_default_value(
    default_value: bool | int | DigitalState,
) -> None:
    waveform = DigitalWaveform(2, 3, default_value=default_value)

    assert waveform.sample_count == len(waveform.data) == 2
    assert waveform.signal_count == 3
    # default_value does not affect the dtype.
    assert waveform.dtype == np.uint8
    assert_type(waveform, DigitalWaveform[np.uint8])
    assert waveform.data.tolist() == [
        [default_value, default_value, default_value],
        [default_value, default_value, default_value],
    ]


###############################################################################
# data
###############################################################################
def test___uint8_waveform___data___returns_uint8_data() -> None:
    waveform = DigitalWaveform.from_lines([0, 1, 2, 3], np.uint8)

    data = waveform.data

    assert_type(data, npt.NDArray[np.uint8])
    assert isinstance(data, np.ndarray) and data.dtype == np.uint8
    assert list(data) == [0, 1, 2, 3]


###############################################################################
# get_data
###############################################################################
def test___uint8_waveform___get_data___returns_data() -> None:
    waveform = DigitalWaveform.from_lines([0, 1, 2, 3], np.uint8)

    data = waveform.get_data()

    assert_type(data, npt.NDArray[np.uint8])
    assert isinstance(data, np.ndarray) and data.dtype == np.uint8
    assert list(data) == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "start_index, sample_count, expected_data",
    [
        (None, None, [0, 1, 2, 3]),
        (0, None, [0, 1, 2, 3]),
        (1, None, [1, 2, 3]),
        (3, None, [3]),
        (4, None, []),
        (None, None, [0, 1, 2, 3]),
        (None, 1, [0]),
        (None, 3, [0, 1, 2]),
        (None, 4, [0, 1, 2, 3]),
        (1, 2, [1, 2]),
        (4, 0, []),
    ],
)
def test___array_subset___get_data___returns_array_subset(
    start_index: int, sample_count: int, expected_data: list[int]
) -> None:
    waveform = DigitalWaveform.from_lines([0, 1, 2, 3], np.uint8)

    data = waveform.get_data(start_index=start_index, sample_count=sample_count)

    assert_type(data, npt.NDArray[np.uint8])
    assert isinstance(data, np.ndarray) and data.dtype == np.uint8
    assert list(data) == expected_data


@pytest.mark.parametrize(
    "start_index, sample_count, expected_message",
    [
        (
            5,
            None,
            "The start index must be less than or equal to the number of samples in the waveform.",
        ),
        (
            0,
            5,
            "The sum of the start index and sample count must be less than or equal to the number of samples in the waveform.",
        ),
        (
            4,
            1,
            "The sum of the start index and sample count must be less than or equal to the number of samples in the waveform.",
        ),
    ],
)
def test___invalid_array_subset___get_data___returns_array_subset(
    start_index: int, sample_count: int, expected_message: str
) -> None:
    waveform = DigitalWaveform.from_lines([0, 1, 2, 3], np.uint8)

    with pytest.raises((TypeError, ValueError)) as exc:
        _ = waveform.get_data(start_index=start_index, sample_count=sample_count)

    assert exc.value.args[0].startswith(expected_message)


###############################################################################
# capacity
###############################################################################
@pytest.mark.parametrize(
    "capacity, expected_data",
    [
        (3, [[1], [2], [3]]),
        (4, [[1], [2], [3], [0]]),
        (10, [[1], [2], [3], [0], [0], [0], [0], [0], [0], [0]]),
    ],
)
def test___waveform___set_capacity___resizes_array_and_pads_with_zeros(
    capacity: int, expected_data: list[int]
) -> None:
    data = [[1], [2], [3]]
    waveform = DigitalWaveform.from_lines(data, np.uint8)

    waveform.capacity = capacity

    assert waveform.capacity == capacity
    assert waveform.data.tolist() == data
    assert waveform._data.tolist() == expected_data


@pytest.mark.parametrize(
    "capacity, expected_message",
    [
        (-2, "The capacity must be a non-negative integer."),
        (-1, "The capacity must be a non-negative integer."),
        (0, "The capacity must be equal to or greater than the number of samples in the waveform."),
        (2, "The capacity must be equal to or greater than the number of samples in the waveform."),
    ],
)
def test___invalid_capacity___set_capacity___raises_value_error(
    capacity: int, expected_message: str
) -> None:
    data = [1, 2, 3]
    waveform = DigitalWaveform.from_lines(data, np.uint8)

    with pytest.raises(ValueError) as exc:
        waveform.capacity = capacity

    assert exc.value.args[0].startswith(expected_message)


def test___referenced_array___set_capacity___reference_sees_size_change() -> None:
    data = np.array([[1, 2], [3, 4], [5, 6]], np.uint8)
    waveform = DigitalWaveform.from_lines(data, np.uint8, copy=False)

    waveform.capacity = 5

    assert len(data) == 5
    assert waveform.capacity == 5
    assert data.tolist() == [[1, 2], [3, 4], [5, 6], [0, 0], [0, 0]]
    assert waveform.data.tolist() == [[1, 2], [3, 4], [5, 6]]
    assert waveform._data.tolist() == [[1, 2], [3, 4], [5, 6], [0, 0], [0, 0]]


def test___array_with_external_buffer___set_capacity___raises_value_error() -> None:
    data = array.array("B", [1, 2, 3])
    waveform = DigitalWaveform.from_lines(data, np.uint8, copy=False)

    with pytest.raises(ValueError) as exc:
        waveform.capacity = 10

    assert exc.value.args[0].startswith("cannot resize this array: it does not own its data")


###############################################################################
# extended properties
###############################################################################
def test___waveform___set_channel_name___sets_extended_property() -> None:
    waveform = DigitalWaveform()

    waveform.channel_name = "Dev1/ai0"

    assert waveform.channel_name == "Dev1/ai0"
    assert waveform.extended_properties["NI_ChannelName"] == "Dev1/ai0"


def test___invalid_type___set_channel_name___raises_type_error() -> None:
    waveform = DigitalWaveform()

    with pytest.raises(TypeError) as exc:
        waveform.channel_name = 1  # type: ignore[assignment]

    assert exc.value.args[0].startswith("The channel name must be a str.")


def test___waveform___set_undefined_property___raises_attribute_error() -> None:
    waveform = DigitalWaveform()

    with pytest.raises(AttributeError):
        waveform.undefined_property = "Whatever"  # type: ignore[attr-defined]


def test___waveform___take_weak_ref___references_waveform() -> None:
    waveform = DigitalWaveform()

    waveform_ref = weakref.ref(waveform)

    assert waveform_ref() is waveform


###############################################################################
# timing
###############################################################################
def test___waveform___has_empty_timing() -> None:
    waveform = DigitalWaveform()

    assert waveform.timing is Timing.empty


def test___bintime___waveform_with_timing___static_type_erased() -> None:
    sample_interval = bt.TimeDelta(1e-3)
    timestamp = bt.DateTime.now(dt.timezone.utc)
    time_offset = bt.TimeDelta(1e-6)
    waveform = DigitalWaveform(
        timing=Timing.create_with_regular_interval(sample_interval, timestamp, time_offset)
    )

    assert_type(waveform.timing.sample_interval, Union[bt.TimeDelta, dt.timedelta, ht.timedelta])
    assert_type(waveform.timing.timestamp, Union[bt.DateTime, dt.datetime, ht.datetime])
    assert_type(waveform.timing.start_time, Union[bt.DateTime, dt.datetime, ht.datetime])
    assert_type(waveform.timing.time_offset, Union[bt.TimeDelta, dt.timedelta, ht.timedelta])
    assert waveform.timing.sample_interval == sample_interval
    assert waveform.timing.timestamp == timestamp
    assert waveform.timing.start_time == timestamp + time_offset
    assert waveform.timing.time_offset == time_offset


def test___datetime___waveform_with_timing___static_type_erased() -> None:
    sample_interval = dt.timedelta(milliseconds=1)
    timestamp = dt.datetime.now(dt.timezone.utc)
    time_offset = dt.timedelta(microseconds=1)
    waveform = DigitalWaveform(
        timing=Timing.create_with_regular_interval(sample_interval, timestamp, time_offset)
    )

    assert_type(waveform.timing.sample_interval, Union[bt.TimeDelta, dt.timedelta, ht.timedelta])
    assert_type(waveform.timing.timestamp, Union[bt.DateTime, dt.datetime, ht.datetime])
    assert_type(waveform.timing.start_time, Union[bt.DateTime, dt.datetime, ht.datetime])
    assert_type(waveform.timing.time_offset, Union[bt.TimeDelta, dt.timedelta, ht.timedelta])
    assert waveform.timing.sample_interval == sample_interval
    assert waveform.timing.timestamp == timestamp
    assert waveform.timing.start_time == timestamp + time_offset
    assert waveform.timing.time_offset == time_offset


def test___hightime___waveform_with_timing___static_type_erased() -> None:
    sample_interval = ht.timedelta(milliseconds=1)
    timestamp = ht.datetime.now(dt.timezone.utc)
    time_offset = ht.timedelta(microseconds=1)
    waveform = DigitalWaveform(
        timing=Timing.create_with_regular_interval(sample_interval, timestamp, time_offset)
    )

    assert_type(waveform.timing.sample_interval, Union[bt.TimeDelta, dt.timedelta, ht.timedelta])
    assert_type(waveform.timing.timestamp, Union[bt.DateTime, dt.datetime, ht.datetime])
    assert_type(waveform.timing.start_time, Union[bt.DateTime, dt.datetime, ht.datetime])
    assert_type(waveform.timing.time_offset, Union[bt.TimeDelta, dt.timedelta, ht.timedelta])
    assert waveform.timing.sample_interval == sample_interval
    assert waveform.timing.timestamp == timestamp
    assert waveform.timing.start_time == timestamp + time_offset
    assert waveform.timing.time_offset == time_offset


@pytest.mark.parametrize(
    "timing",
    [
        Timing.create_with_regular_interval(
            bt.TimeDelta(1e-3), bt.DateTime(2025, 1, 1, tzinfo=dt.timezone.utc), bt.TimeDelta(1e-6)
        ),
        Timing.create_with_regular_interval(
            dt.timedelta(milliseconds=1),
            dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc),
            dt.timedelta(microseconds=1),
        ),
        Timing.create_with_regular_interval(
            ht.timedelta(milliseconds=1),
            ht.datetime(2025, 1, 1, tzinfo=dt.timezone.utc),
            dt.timedelta(microseconds=1),
        ),
    ],
)
def test___polymorphic_timing___get_timing_properties___behaves_polymorphically(
    timing: Timing[Any, Any, Any],
) -> None:
    waveform = DigitalWaveform(timing=timing)

    assert waveform.timing.sample_interval.total_seconds() == pytest.approx(1e-3)
    assert (
        waveform.timing.timestamp.year,
        waveform.timing.timestamp.month,
        waveform.timing.timestamp.day,
    ) == (2025, 1, 1)
    assert waveform.timing.time_offset.total_seconds() == pytest.approx(1e-6)


###############################################################################
# append array
###############################################################################
def test___empty_ndarray___append___no_effect() -> None:
    waveform = DigitalWaveform.from_lines([[0], [1], [2]], np.uint8)
    array = np.array([], np.uint8)

    waveform.append(array)

    assert waveform.data.tolist() == [[0], [1], [2]]


def test___uint8_ndarray___append___appends_array() -> None:
    waveform = DigitalWaveform.from_lines([[0], [1], [2]], np.uint8)
    array = np.array([3, 4, 5], np.uint8)

    waveform.append(array)

    assert waveform.data.tolist() == [[0], [1], [2], [3], [4], [5]]


def test___bool_ndarray___append___appends_array() -> None:
    waveform = DigitalWaveform.from_lines([False, True, False], np.bool)
    array = np.array([True, False, True], np.bool)

    waveform.append(array)

    assert waveform.data.tolist() == [[False], [True], [False], [True], [False], [True]]


def test___ndarray_with_mismatched_dtype___append___raises_type_error() -> None:
    waveform = DigitalWaveform.from_lines([0, 1, 2], np.bool)
    array = np.array([3, 4, 5], np.uint8)

    with pytest.raises(TypeError) as exc:
        waveform.append(array)  # type: ignore[arg-type]

    assert exc.value.args[0].startswith(
        "The data type of the input array must match the waveform data type."
    )


def test___ndarray_2d___append___appends_array() -> None:
    waveform = DigitalWaveform.from_lines([[0, 1, 2], [3, 4, 5]], np.uint8)
    array = np.array([[6, 7, 8], [9, 10, 11]], np.uint8)

    waveform.append(array)

    assert waveform.data.tolist() == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]


def test___irregular_waveform_and_uint8_ndarray_with_timestamps___append___appends_array() -> None:
    start_time = dt.datetime.now(dt.timezone.utc)
    waveform_offsets = [dt.timedelta(0), dt.timedelta(1), dt.timedelta(2)]
    waveform_timestamps = [start_time + offset for offset in waveform_offsets]
    waveform = DigitalWaveform.from_lines([[0], [1], [2]], np.uint8)
    waveform.timing = Timing.create_with_irregular_interval(waveform_timestamps)
    array_offsets = [dt.timedelta(3), dt.timedelta(4), dt.timedelta(5)]
    array_timestamps = [start_time + offset for offset in array_offsets]
    array = np.array([3, 4, 5], np.uint8)

    waveform.append(array, array_timestamps)

    assert waveform.data.tolist() == [[0], [1], [2], [3], [4], [5]]
    assert waveform.timing.sample_interval_mode == SampleIntervalMode.IRREGULAR
    assert waveform.timing._timestamps == waveform_timestamps + array_timestamps


def test___irregular_waveform_and_uint8_ndarray_without_timestamps___append___raises_timing_mismatch_error_and_does_not_append() -> (
    None
):
    start_time = dt.datetime.now(dt.timezone.utc)
    waveform_offsets = [dt.timedelta(0), dt.timedelta(1), dt.timedelta(2)]
    waveform_timestamps = [start_time + offset for offset in waveform_offsets]
    waveform = DigitalWaveform.from_lines([[0], [1], [2]], np.uint8)
    waveform.timing = Timing.create_with_irregular_interval(waveform_timestamps)
    array = np.array([3, 4, 5], np.uint8)

    with pytest.raises(TimingMismatchError) as exc:
        waveform.append(array)

    assert exc.value.args[0].startswith(
        "The timestamps argument is required when appending to a waveform with irregular timing."
    )
    assert waveform.data.tolist() == [[0], [1], [2]]
    assert waveform.timing.sample_interval_mode == SampleIntervalMode.IRREGULAR
    assert waveform.timing._timestamps == waveform_timestamps


def test___irregular_waveform_and_uint8_ndarray_with_wrong_timestamp_count___append___raises_value_error_and_does_not_append() -> (
    None
):
    start_time = dt.datetime.now(dt.timezone.utc)
    waveform_offsets = [dt.timedelta(0), dt.timedelta(1), dt.timedelta(2)]
    waveform_timestamps = [start_time + offset for offset in waveform_offsets]
    waveform = DigitalWaveform.from_lines([[0], [1], [2]], np.uint8)
    waveform.timing = Timing.create_with_irregular_interval(waveform_timestamps)
    array_offsets = [dt.timedelta(3), dt.timedelta(4)]
    array_timestamps = [start_time + offset for offset in array_offsets]
    array = np.array([3, 4, 5], np.uint8)

    with pytest.raises(ValueError) as exc:
        waveform.append(array, array_timestamps)

    assert exc.value.args[0].startswith(
        "The number of irregular timestamps must be equal to the input array length."
    )
    assert waveform.data.tolist() == [[0], [1], [2]]
    assert waveform.timing.sample_interval_mode == SampleIntervalMode.IRREGULAR
    assert waveform.timing._timestamps == waveform_timestamps


def test___regular_waveform_and_uint8_ndarray_with_timestamps___append___raises_runtime_error_and_does_not_append() -> (
    None
):
    start_time = dt.datetime.now(dt.timezone.utc)
    waveform = DigitalWaveform.from_lines([[0], [1], [2]], np.uint8)
    waveform.timing = Timing.create_with_regular_interval(dt.timedelta(milliseconds=1))
    array_offsets = [dt.timedelta(3), dt.timedelta(4), dt.timedelta(5)]
    array_timestamps = [start_time + offset for offset in array_offsets]
    array = np.array([3, 4, 5], np.uint8)

    with pytest.raises(ValueError) as exc:
        waveform.append(array, array_timestamps)

    assert exc.value.args[0].startswith("The timestamps argument is not supported.")
    assert waveform.data.tolist() == [[0], [1], [2]]
    assert waveform.timing.sample_interval_mode == SampleIntervalMode.REGULAR
    assert waveform.timing.sample_interval == dt.timedelta(milliseconds=1)


###############################################################################
# append waveform
###############################################################################
def test___empty_waveform___append___no_effect() -> None:
    waveform = DigitalWaveform.from_lines([[0], [1], [2]], np.uint8)
    other = DigitalWaveform(dtype=np.uint8)

    waveform.append(other)

    assert waveform.data.tolist() == [[0], [1], [2]]


def test___uint8_waveform___append___appends_waveform() -> None:
    waveform = DigitalWaveform.from_lines([[0], [1], [2]], np.uint8)
    other = DigitalWaveform.from_lines([[3], [4], [5]], np.uint8)

    waveform.append(other)

    assert waveform.data.tolist() == [[0], [1], [2], [3], [4], [5]]


def test___bool_waveform___append___appends_waveform() -> None:
    waveform = DigitalWaveform.from_lines([[False], [True], [False]], np.bool)
    other = DigitalWaveform.from_lines([[True], [False], [True]], np.bool)

    waveform.append(other)

    assert waveform.data.tolist() == [[False], [True], [False], [True], [False], [True]]


def test___waveform_with_mismatched_dtype___append___raises_type_error() -> None:
    waveform = DigitalWaveform.from_lines([[False], [True], [False]], np.bool)
    other = DigitalWaveform.from_lines([[3], [4], [5]], np.uint8)

    with pytest.raises(TypeError) as exc:
        waveform.append(other)  # type: ignore[arg-type]

    assert exc.value.args[0].startswith(
        "The data type of the input waveform must match the waveform data type."
    )


def test___irregular_waveform_and_irregular_waveform___append___appends_waveform() -> None:
    start_time = dt.datetime.now(dt.timezone.utc)
    waveform_offsets = [dt.timedelta(0), dt.timedelta(1), dt.timedelta(2)]
    waveform_timestamps = [start_time + offset for offset in waveform_offsets]
    waveform = DigitalWaveform.from_lines([[0], [1], [2]], np.uint8)
    waveform.timing = Timing.create_with_irregular_interval(waveform_timestamps)
    other_offsets = [dt.timedelta(3), dt.timedelta(4), dt.timedelta(5)]
    other_timestamps = [start_time + offset for offset in other_offsets]
    other = DigitalWaveform.from_lines([[3], [4], [5]], np.uint8)
    other.timing = Timing.create_with_irregular_interval(other_timestamps)

    waveform.append(other)

    assert waveform.data.tolist() == [[0], [1], [2], [3], [4], [5]]
    assert waveform.timing.sample_interval_mode == SampleIntervalMode.IRREGULAR
    assert waveform.timing._timestamps == waveform_timestamps + other_timestamps


def test___irregular_waveform_and_regular_waveform___append___raises_timing_mismatch_error() -> (
    None
):
    start_time = dt.datetime.now(dt.timezone.utc)
    waveform_offsets = [dt.timedelta(0), dt.timedelta(1), dt.timedelta(2)]
    waveform_timestamps = [start_time + offset for offset in waveform_offsets]
    waveform = DigitalWaveform.from_lines([[0], [1], [2]], np.uint8)
    waveform.timing = Timing.create_with_irregular_interval(waveform_timestamps)
    other = DigitalWaveform.from_lines([[3], [4], [5]], np.uint8)

    with pytest.raises(TimingMismatchError) as exc:
        waveform.append(other)

    assert exc.value.args[0].startswith(
        "The timing of one or more waveforms does not match the timing of the current waveform."
    )
    assert waveform.data.tolist() == [[0], [1], [2]]
    assert waveform.timing.sample_interval_mode == SampleIntervalMode.IRREGULAR
    assert waveform.timing._timestamps == waveform_timestamps


def test___regular_waveform_and_irregular_waveform___append___raises_timing_mismatch_error() -> (
    None
):
    start_time = dt.datetime.now(dt.timezone.utc)
    waveform = DigitalWaveform.from_lines([[0], [1], [2]], np.uint8)
    waveform.timing = Timing.create_with_regular_interval(dt.timedelta(milliseconds=1))
    other_offsets = [dt.timedelta(3), dt.timedelta(4), dt.timedelta(5)]
    other_timestamps = [start_time + offset for offset in other_offsets]
    other = DigitalWaveform.from_lines([[3], [4], [5]], np.uint8)
    other.timing = Timing.create_with_irregular_interval(other_timestamps)

    with pytest.raises(TimingMismatchError) as exc:
        waveform.append(other)

    assert exc.value.args[0].startswith(
        "The timing of one or more waveforms does not match the timing of the current waveform."
    )
    assert waveform.data.tolist() == [[0], [1], [2]]
    assert waveform.timing.sample_interval_mode == SampleIntervalMode.REGULAR
    assert waveform.timing.sample_interval == dt.timedelta(milliseconds=1)


def test___regular_waveform_and_regular_waveform_with_different_sample_interval___append___appends_waveform_with_timing_mismatch_warning() -> (
    None
):
    waveform = DigitalWaveform.from_lines([[0], [1], [2]], np.uint8)
    waveform.timing = Timing.create_with_regular_interval(dt.timedelta(milliseconds=1))
    other = DigitalWaveform.from_lines([[3], [4], [5]], np.uint8)
    other.timing = Timing.create_with_regular_interval(dt.timedelta(milliseconds=2))

    with pytest.warns(TimingMismatchWarning):
        waveform.append(other)

    assert waveform.data.tolist() == [[0], [1], [2], [3], [4], [5]]
    assert waveform.timing.sample_interval_mode == SampleIntervalMode.REGULAR
    assert waveform.timing.sample_interval == dt.timedelta(milliseconds=1)


def test___regular_waveform_and_regular_waveform_with_different_extended_properties___append___merges_extended_properties() -> (
    None
):
    waveform = DigitalWaveform.from_lines([[0], [1], [2]], np.uint8)
    waveform.extended_properties["A"] = 1
    waveform.extended_properties["B"] = 2
    other = DigitalWaveform.from_lines([[3], [4], [5]], np.uint8)
    other.extended_properties["B"] = 3
    other.extended_properties["C"] = 4

    waveform.append(other)

    assert waveform.data.tolist() == [[0], [1], [2], [3], [4], [5]]
    assert waveform.extended_properties == {"A": 1, "B": 2, "C": 4}


###############################################################################
# append waveforms
###############################################################################
def test___empty_waveform_list___append___no_effect() -> None:
    waveform = DigitalWaveform.from_lines([[0], [1], [2]], np.uint8)
    other: list[DigitalWaveform[np.uint8]] = []

    waveform.append(other)

    assert waveform.data.tolist() == [[0], [1], [2]]


def test___uint8_waveform_list___append___appends_waveform() -> None:
    waveform = DigitalWaveform.from_lines([[0], [1], [2]], np.uint8)
    other = [
        DigitalWaveform.from_lines([[3], [4], [5]], np.uint8),
        DigitalWaveform.from_lines([[6]], np.uint8),
        DigitalWaveform.from_lines([[7], [8]], np.uint8),
    ]

    waveform.append(other)

    assert waveform.data.tolist() == [[0], [1], [2], [3], [4], [5], [6], [7], [8]]


def test___bool_waveform_tuple___append___appends_waveform() -> None:
    waveform = DigitalWaveform.from_lines([[False], [True], [False]], np.bool)
    other = (
        DigitalWaveform.from_lines([[True], [False], [True]], np.bool),
        DigitalWaveform.from_lines([[True], [True], [False]], np.bool),
    )

    waveform.append(other)

    assert waveform.data.tolist() == [
        [False],
        [True],
        [False],
        [True],
        [False],
        [True],
        [True],
        [True],
        [False],
    ]


def test___waveform_list_with_mismatched_dtype___append___raises_type_error_and_does_not_append() -> (
    None
):
    waveform = DigitalWaveform.from_lines([[False], [True], [False]], np.bool)
    other = [
        DigitalWaveform.from_lines([[True], [False], [True]], np.bool),
        DigitalWaveform.from_lines([[6], [7], [8]], np.uint8),
    ]

    with pytest.raises(TypeError) as exc:
        waveform.append(other)  # type: ignore[arg-type]

    assert exc.value.args[0].startswith(
        "The data type of the input waveform must match the waveform data type."
    )
    assert waveform.data.tolist() == [[False], [True], [False]]


def test___irregular_waveform_and_irregular_waveform_list___append___appends_waveform() -> None:
    start_time = dt.datetime.now(dt.timezone.utc)
    waveform_offsets = [dt.timedelta(0), dt.timedelta(1), dt.timedelta(2)]
    waveform_timestamps = [start_time + offset for offset in waveform_offsets]
    waveform = DigitalWaveform.from_lines([[0], [1], [2]], np.uint8)
    waveform.timing = Timing.create_with_irregular_interval(waveform_timestamps)
    other1_offsets = [dt.timedelta(3), dt.timedelta(4), dt.timedelta(5)]
    other1_timestamps = [start_time + offset for offset in other1_offsets]
    other1 = DigitalWaveform.from_lines([[3], [4], [5]], np.uint8)
    other1.timing = Timing.create_with_irregular_interval(other1_timestamps)
    other2_offsets = [dt.timedelta(6), dt.timedelta(7), dt.timedelta(8)]
    other2_timestamps = [start_time + offset for offset in other2_offsets]
    other2 = DigitalWaveform.from_lines([[6], [7], [8]], np.uint8)
    other2.timing = Timing.create_with_irregular_interval(other2_timestamps)
    other = [other1, other2]

    waveform.append(other)

    assert waveform.data.tolist() == [[0], [1], [2], [3], [4], [5], [6], [7], [8]]
    assert waveform.timing.sample_interval_mode == SampleIntervalMode.IRREGULAR
    assert (
        waveform.timing._timestamps == waveform_timestamps + other1_timestamps + other2_timestamps
    )


def test___irregular_waveform_and_regular_waveform_list___append___raises_timing_mismatch_error_and_does_not_append() -> (
    None
):
    start_time = dt.datetime.now(dt.timezone.utc)
    waveform_offsets = [dt.timedelta(0), dt.timedelta(1), dt.timedelta(2)]
    waveform_timestamps = [start_time + offset for offset in waveform_offsets]
    waveform = DigitalWaveform.from_lines([[0], [1], [2]], np.uint8)
    waveform.timing = Timing.create_with_irregular_interval(waveform_timestamps)
    other1_offsets = [dt.timedelta(3), dt.timedelta(4), dt.timedelta(5)]
    other1_timestamps = [start_time + offset for offset in other1_offsets]
    other1 = DigitalWaveform.from_lines([[3], [4], [5]], np.uint8)
    other1.timing = Timing.create_with_irregular_interval(other1_timestamps)
    other2 = DigitalWaveform.from_lines([[6], [7], [8]], np.uint8)
    other2.timing = Timing.create_with_regular_interval(dt.timedelta(milliseconds=1))
    other = [other1, other2]

    with pytest.raises(TimingMismatchError) as exc:
        waveform.append(other)

    assert exc.value.args[0].startswith(
        "The timing of one or more waveforms does not match the timing of the current waveform."
    )
    assert waveform.data.tolist() == [[0], [1], [2]]
    assert waveform.timing.sample_interval_mode == SampleIntervalMode.IRREGULAR
    assert waveform.timing._timestamps == waveform_timestamps


def test___regular_waveform_and_irregular_waveform_list___append___raises_runtime_error_and_does_not_append() -> (
    None
):
    start_time = dt.datetime.now(dt.timezone.utc)
    waveform = DigitalWaveform.from_lines([[0], [1], [2]], np.uint8)
    waveform.timing = Timing.create_with_regular_interval(dt.timedelta(milliseconds=1))
    other1 = DigitalWaveform.from_lines([[3], [4], [5]], np.uint8)
    other1.timing = Timing.create_with_regular_interval(dt.timedelta(milliseconds=1))
    other2_offsets = [dt.timedelta(3), dt.timedelta(4), dt.timedelta(5)]
    other2_timestamps = [start_time + offset for offset in other2_offsets]
    other2 = DigitalWaveform.from_lines([[3], [4], [5]], np.uint8)
    other2.timing = Timing.create_with_irregular_interval(other2_timestamps)
    other = [other1, other2]

    with pytest.raises(RuntimeError) as exc:
        waveform.append(other)

    assert exc.value.args[0].startswith(
        "The timing of one or more waveforms does not match the timing of the current waveform."
    )
    assert waveform.data.tolist() == [[0], [1], [2]]
    assert waveform.timing.sample_interval_mode == SampleIntervalMode.REGULAR
    assert waveform.timing.sample_interval == dt.timedelta(milliseconds=1)


###############################################################################
# load data
###############################################################################
def test___empty_ndarray___load_data___clears_data() -> None:
    waveform = DigitalWaveform.from_lines([[0], [1], [2]], np.uint8)
    array = np.array([], np.uint8)

    waveform.load_data(array)

    assert waveform.data.tolist() == []


def test___uint8_ndarray___load_data___overwrites_data() -> None:
    waveform = DigitalWaveform.from_lines([[0], [1], [2]], np.uint8)
    array = np.array([[3], [4], [5]], np.uint8)

    waveform.load_data(array)

    assert waveform.data.tolist() == [[3], [4], [5]]


def test___bool_ndarray___load_data___overwrites_data() -> None:
    waveform = DigitalWaveform.from_lines([[False], [True], [False]], np.bool)
    array = np.array([[True], [False], [True]], np.bool)

    waveform.load_data(array)

    assert waveform.data.tolist() == [[True], [False], [True]]


def test___ndarray_with_mismatched_dtype___load_data___raises_type_error() -> None:
    waveform = DigitalWaveform.from_lines([[False], [True], [False]], np.bool)
    array = np.array([[3], [4], [5]], np.uint8)

    with pytest.raises(TypeError) as exc:
        waveform.load_data(array)  # type: ignore[arg-type]

    assert exc.value.args[0].startswith(
        "The data type of the input array must match the waveform data type."
    )


def test___ndarray_2d___load_data___overwrites_data() -> None:
    waveform = DigitalWaveform.from_lines([[False, True], [False, False]], np.bool)
    array = np.array([[True, True], [False, False], [True, False]], np.bool)

    waveform.load_data(array)

    assert waveform.data.tolist() == [[True, True], [False, False], [True, False]]


def test___smaller_ndarray___load_data___preserves_capacity() -> None:
    waveform = DigitalWaveform.from_lines([[0], [1], [2]], np.uint8)
    array = np.array([[3]], np.uint8)

    waveform.load_data(array)

    assert waveform.data.tolist() == [[3]]
    assert waveform.capacity == 3


def test___larger_ndarray___load_data___grows_capacity() -> None:
    waveform = DigitalWaveform.from_lines([[0], [1], [2]], np.uint8)
    array = np.array([[3], [4], [5], [6]], np.uint8)

    waveform.load_data(array)

    assert waveform.data.tolist() == [[3], [4], [5], [6]]
    assert waveform.capacity == 4


def test___waveform_with_start_index___load_data___clears_start_index() -> None:
    waveform = DigitalWaveform.from_lines(
        np.array([[0], [1], [2]], np.uint8), np.uint8, copy=False, start_index=1, sample_count=1
    )
    assert waveform._start_index == 1
    array = np.array([[3]], np.uint8)

    waveform.load_data(array)

    assert waveform.data.tolist() == [[3]]
    assert waveform._start_index == 0


def test___ndarray_subset___load_data___overwrites_data() -> None:
    waveform = DigitalWaveform.from_lines([[0], [1], [2]], np.uint8)
    array = np.array([[3], [4], [5]], np.uint8)

    waveform.load_data(array, start_index=1, sample_count=1)

    assert waveform.data.tolist() == [[4]]
    assert waveform._start_index == 0
    assert waveform.capacity == 3


def test___smaller_ndarray_no_copy___load_data___takes_ownership_of_array() -> None:
    waveform = DigitalWaveform.from_lines([[0], [1], [2]], np.uint8)
    array = np.array([[3]], np.uint8)

    waveform.load_data(array, copy=False)

    assert waveform.data.tolist() == [[3]]
    assert waveform._data is array


def test___larger_ndarray_no_copy___load_data___takes_ownership_of_array() -> None:
    waveform = DigitalWaveform.from_lines([[0], [1], [2]], np.uint8)
    array = np.array([[3], [4], [5], [6]], np.uint8)

    waveform.load_data(array, copy=False)

    assert waveform.data.tolist() == [[3], [4], [5], [6]]
    assert waveform._data is array


def test___ndarray_subset_no_copy___load_data___takes_ownership_of_array_subset() -> None:
    waveform = DigitalWaveform.from_lines([[0], [1], [2]], np.uint8)
    array = np.array([[3], [4], [5], [6]], np.uint8)

    waveform.load_data(array, copy=False, start_index=1, sample_count=2)

    assert waveform.data.tolist() == [[4], [5]]
    assert waveform._data is array


def test___irregular_waveform_and_uint8_ndarray_with_timestamps___load_data___overwrites_data_but_not_timestamps() -> (
    None
):
    start_time = dt.datetime.now(dt.timezone.utc)
    waveform_offsets = [dt.timedelta(0), dt.timedelta(1), dt.timedelta(2)]
    waveform_timestamps = [start_time + offset for offset in waveform_offsets]
    waveform = DigitalWaveform.from_lines([[0], [1], [2]], np.uint8)
    waveform.timing = Timing.create_with_irregular_interval(waveform_timestamps)
    array = np.array([3, 4, 5], np.uint8)

    waveform.load_data(array)

    assert waveform.data.tolist() == [[3], [4], [5]]
    assert waveform.timing.sample_interval_mode == SampleIntervalMode.IRREGULAR
    assert waveform.timing._timestamps == waveform_timestamps


def test___irregular_waveform_and_uint8_ndarray_with_wrong_sample_count___load_data___raises_value_error_and_does_not_overwrite_data() -> (
    None
):
    start_time = dt.datetime.now(dt.timezone.utc)
    waveform_offsets = [dt.timedelta(0), dt.timedelta(1), dt.timedelta(2)]
    waveform_timestamps = [start_time + offset for offset in waveform_offsets]
    waveform = DigitalWaveform.from_lines([[0], [1], [2]], np.uint8)
    waveform.timing = Timing.create_with_irregular_interval(waveform_timestamps)
    array = np.array([3, 4], np.uint8)

    with pytest.raises(ValueError) as exc:
        waveform.load_data(array)

    assert exc.value.args[0].startswith(
        "The input array length must be equal to the number of irregular timestamps."
    )
    assert waveform.data.tolist() == [[0], [1], [2]]
    assert waveform.timing.sample_interval_mode == SampleIntervalMode.IRREGULAR
    assert waveform.timing._timestamps == waveform_timestamps


###############################################################################
# magic methods
###############################################################################
@pytest.mark.parametrize(
    "left, right",
    [
        (DigitalWaveform(), DigitalWaveform()),
        (DigitalWaveform(10), DigitalWaveform(10)),
        (DigitalWaveform(10, 1), DigitalWaveform(10, 1)),
        (DigitalWaveform(10, 1, np.bool), DigitalWaveform(10, 1, np.bool)),
        (DigitalWaveform(10, 1, np.uint8), DigitalWaveform(10, 1, np.uint8)),
        (
            DigitalWaveform(10, 1, np.uint8, start_index=5, capacity=20),
            DigitalWaveform(10, 1, np.uint8, start_index=5, capacity=20),
        ),
        (
            DigitalWaveform.from_lines([0, 1, 2, 3], np.bool),
            DigitalWaveform.from_lines([0, 1, 2, 3], np.bool),
        ),
        # np.bool coerces non-zero values to True, so in this case, 4 == 2.
        (
            DigitalWaveform.from_lines([0, 1, 4, 3], np.bool),
            DigitalWaveform.from_lines([0, 1, 2, 3], np.bool),
        ),
        (
            DigitalWaveform.from_lines([0, 1, 2, 3], np.uint8),
            DigitalWaveform.from_lines([0, 1, 2, 3], np.uint8),
        ),
        (
            DigitalWaveform(
                timing=Timing.create_with_regular_interval(dt.timedelta(milliseconds=1))
            ),
            DigitalWaveform(
                timing=Timing.create_with_regular_interval(dt.timedelta(milliseconds=1))
            ),
        ),
        (
            DigitalWaveform(
                timing=Timing.create_with_regular_interval(ht.timedelta(milliseconds=1))
            ),
            DigitalWaveform(
                timing=Timing.create_with_regular_interval(ht.timedelta(milliseconds=1))
            ),
        ),
        (
            DigitalWaveform(
                extended_properties={"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Volts"}
            ),
            DigitalWaveform(
                extended_properties={"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Volts"}
            ),
        ),
        # start_index and capacity may differ as long as data and sample_count are the same.
        (
            DigitalWaveform(10, 1, np.uint8, start_index=5, capacity=20),
            DigitalWaveform(10, 1, np.uint8, start_index=10, capacity=25),
        ),
        (
            DigitalWaveform.from_lines(
                [0, 0, 1, 2, 3, 4, 5, 0], np.uint8, start_index=2, sample_count=5
            ),
            DigitalWaveform.from_lines(
                [0, 1, 2, 3, 4, 5, 0, 0, 0], np.uint8, start_index=1, sample_count=5
            ),
        ),
        # Same value, different time type
        (
            DigitalWaveform(
                timing=Timing.create_with_regular_interval(dt.timedelta(milliseconds=1))
            ),
            DigitalWaveform(
                timing=Timing.create_with_regular_interval(ht.timedelta(milliseconds=1))
            ),
        ),
        (
            DigitalWaveform(
                timing=Timing.create_with_regular_interval(ht.timedelta(milliseconds=1))
            ),
            DigitalWaveform(
                timing=Timing.create_with_regular_interval(dt.timedelta(milliseconds=1))
            ),
        ),
    ],
)
def test___same_value___equality___equal(
    left: DigitalWaveform[Any], right: DigitalWaveform[Any]
) -> None:
    assert left == right
    assert not (left != right)


@pytest.mark.parametrize(
    "left, right",
    [
        (DigitalWaveform(), DigitalWaveform(10)),
        (DigitalWaveform(10), DigitalWaveform(11)),
        (DigitalWaveform(10, 1), DigitalWaveform(10, 2)),
        (DigitalWaveform(10, 1, np.bool), DigitalWaveform(10, 1, np.uint8)),
        (
            DigitalWaveform(15, 1, np.uint8, start_index=5, capacity=20),
            DigitalWaveform(10, 1, np.uint8, start_index=5, capacity=20),
        ),
        (
            DigitalWaveform.from_lines([0, 1, 0, 3], np.bool),
            DigitalWaveform.from_lines([0, 1, 2, 3], np.bool),
        ),
        (
            DigitalWaveform.from_lines([0, 1, 4, 3], np.uint8),
            DigitalWaveform.from_lines([0, 1, 2, 3], np.uint8),
        ),
        (
            DigitalWaveform.from_lines([0, 1, 2, 3], np.uint8),
            DigitalWaveform.from_lines([0, 1, 2, 3], np.bool),
        ),
        (
            DigitalWaveform(
                timing=Timing.create_with_regular_interval(dt.timedelta(milliseconds=1))
            ),
            DigitalWaveform(
                timing=Timing.create_with_regular_interval(dt.timedelta(milliseconds=2))
            ),
        ),
        (
            DigitalWaveform(
                timing=Timing.create_with_regular_interval(ht.timedelta(milliseconds=1))
            ),
            DigitalWaveform(
                timing=Timing.create_with_regular_interval(ht.timedelta(milliseconds=2))
            ),
        ),
        (
            DigitalWaveform(
                extended_properties={"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Volts"}
            ),
            DigitalWaveform(
                extended_properties={"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Amps"}
            ),
        ),
    ],
)
def test___different_value___equality___not_equal(
    left: DigitalWaveform[Any], right: DigitalWaveform[Any]
) -> None:
    assert not (left == right)
    assert left != right


@pytest.mark.parametrize(
    "value, expected_repr",
    [
        (DigitalWaveform(), "nitypes.waveform.DigitalWaveform(0, 1)"),
        (
            DigitalWaveform(3, 2),
            "nitypes.waveform.DigitalWaveform(3, 2, data=array([[0, 0], [0, 0], [0, 0]], dtype=uint8))",
        ),
        (
            DigitalWaveform(3, 2, np.bool),
            "nitypes.waveform.DigitalWaveform(3, 2, bool, data=array([[False, False], [False, False], [False, False]]))",
        ),
        (DigitalWaveform(0, 1, np.uint8), "nitypes.waveform.DigitalWaveform(0, 1)"),
        (
            DigitalWaveform(5, 1, np.uint8),
            f"nitypes.waveform.DigitalWaveform(5, 1, data=array([[0], [0], [0], [0], [0]], dtype=uint8))",
        ),
        (
            DigitalWaveform(5, 1, np.uint8, start_index=5, capacity=20),
            f"nitypes.waveform.DigitalWaveform(5, 1, data=array([[0], [0], [0], [0], [0]], dtype=uint8))",
        ),
        (
            DigitalWaveform.from_lines([0, 1, 2, 3], np.bool),
            "nitypes.waveform.DigitalWaveform(4, 1, bool, data=array([[False], [ True], [ True], [ True]]))",
        ),
        (
            DigitalWaveform.from_lines([0, 1, 2, 3], np.uint8),
            f"nitypes.waveform.DigitalWaveform(4, 1, data=array([[0], [1], [2], [3]], dtype=uint8))",
        ),
        (
            DigitalWaveform(
                timing=Timing.create_with_regular_interval(dt.timedelta(milliseconds=1))
            ),
            "nitypes.waveform.DigitalWaveform(0, 1, "
            "timing=nitypes.waveform.Timing(nitypes.waveform.SampleIntervalMode.REGULAR, "
            "sample_interval=datetime.timedelta(microseconds=1000)))",
        ),
        (
            DigitalWaveform(
                timing=Timing.create_with_regular_interval(ht.timedelta(milliseconds=1))
            ),
            "nitypes.waveform.DigitalWaveform(0, 1, "
            "timing=nitypes.waveform.Timing(nitypes.waveform.SampleIntervalMode.REGULAR, "
            "sample_interval=hightime.timedelta(microseconds=1000)))",
        ),
        (
            DigitalWaveform(
                extended_properties={"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Volts"}
            ),
            "nitypes.waveform.DigitalWaveform(0, 1, extended_properties={'NI_ChannelName': 'Dev1/ai0', "
            "'NI_UnitDescription': 'Volts'})",
        ),
        (
            DigitalWaveform.from_lines(
                [1, 2, 3],
                np.uint8,
                timing=Timing.create_with_regular_interval(dt.timedelta(milliseconds=1)),
            ),
            f"nitypes.waveform.DigitalWaveform(3, 1, data=array([[1], [2], [3]], dtype=uint8), "
            "timing=nitypes.waveform.Timing(nitypes.waveform.SampleIntervalMode.REGULAR, "
            "sample_interval=datetime.timedelta(microseconds=1000)))",
        ),
        (
            DigitalWaveform.from_lines(
                [1, 2, 3],
                np.uint8,
                extended_properties={"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Volts"},
            ),
            f"nitypes.waveform.DigitalWaveform(3, 1, data=array([[1], [2], [3]], dtype=uint8), "
            "extended_properties={'NI_ChannelName': 'Dev1/ai0', 'NI_UnitDescription': 'Volts'})",
        ),
    ],
)
def test___various_values___repr___looks_ok(
    value: DigitalWaveform[Any], expected_repr: str
) -> None:
    assert repr(value) == expected_repr


_VARIOUS_VALUES = [
    DigitalWaveform(),
    DigitalWaveform(10, 2),
    DigitalWaveform(10, 2, np.bool),
    DigitalWaveform(10, 2, np.uint8),
    DigitalWaveform(10, 2, np.uint8, start_index=5, capacity=20),
    DigitalWaveform.from_lines([0, 1, 2, 3], np.bool),
    DigitalWaveform.from_lines([0, 1, 2, 3], np.uint8),
    DigitalWaveform(timing=Timing.create_with_regular_interval(dt.timedelta(milliseconds=1))),
    DigitalWaveform(timing=Timing.create_with_regular_interval(ht.timedelta(milliseconds=1))),
    DigitalWaveform(
        extended_properties={"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Volts"}
    ),
    DigitalWaveform(10, 2, np.uint8, start_index=5, capacity=20),
    DigitalWaveform.from_lines([0, 0, 1, 2, 3, 4, 5, 0], np.uint8, start_index=2, sample_count=5),
]


@pytest.mark.parametrize("value", _VARIOUS_VALUES)
def test___various_values___copy___makes_shallow_copy(value: DigitalWaveform[Any]) -> None:
    new_value = copy.copy(value)

    _assert_shallow_copy(new_value, value)


def _assert_shallow_copy(value: DigitalWaveform[Any], other: DigitalWaveform[Any]) -> None:
    assert value == other
    assert value is not other
    # _data may be a view of the original array. If the original array is 1D, then look at _data_1d.
    assert (
        value._data is other._data
        or value._data.base is other._data
        or value._data.base is other._data_1d
    )
    assert value._extended_properties is other._extended_properties
    assert value._timing is other._timing


@pytest.mark.parametrize("value", _VARIOUS_VALUES)
def test___various_values___deepcopy___makes_shallow_copy(value: DigitalWaveform[Any]) -> None:
    new_value = copy.deepcopy(value)

    _assert_deep_copy(new_value, value)


def _assert_deep_copy(value: DigitalWaveform[Any], other: DigitalWaveform[Any]) -> None:
    assert value == other
    assert value is not other
    assert value._data is not other._data and value._data.base is not other._data
    assert value._extended_properties is not other._extended_properties
    if other._timing is not Timing.empty:
        assert value._timing is not other._timing


@pytest.mark.parametrize("value", _VARIOUS_VALUES)
def test___various_values___pickle_unpickle___makes_deep_copy(value: DigitalWaveform[Any]) -> None:
    new_value = pickle.loads(pickle.dumps(value))

    _assert_deep_copy(new_value, value)


def test___waveform___pickle___references_public_modules() -> None:
    value = DigitalWaveform(
        data=np.array([1, 2, 3], np.bool),
        extended_properties={"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Volts"},
        timing=Timing.create_with_regular_interval(dt.timedelta(milliseconds=1)),
    )

    value_bytes = pickle.dumps(value)

    assert b"nitypes.waveform" in value_bytes
    assert b"nitypes.waveform._digital" not in value_bytes
    assert b"nitypes.waveform._extended_properties" not in value_bytes
    assert b"nitypes.waveform._timing" not in value_bytes
