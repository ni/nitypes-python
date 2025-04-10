import weakref
from typing import Any, assert_type

import numpy as np
import pytest

from nitypes.waveform import AnalogWaveform


def test___no_args___create___creates_empty_waveform_with_default_dtype() -> None:
    waveform = AnalogWaveform.create()

    assert waveform.sample_count == waveform.capacity == len(waveform.raw_data) == 0
    assert waveform.dtype == np.float64
    assert_type(waveform, AnalogWaveform[np.float64])


def test___sample_count___create___creates_waveform_with_sample_count_and_default_dtype() -> None:
    waveform = AnalogWaveform.create(10)

    assert waveform.sample_count == waveform.capacity == len(waveform.raw_data) == 10
    assert waveform.dtype == np.float64
    assert_type(waveform, AnalogWaveform[np.float64])


def test___sample_count_and_dtype___create___creates_waveform_with_sample_count_and_dtype() -> None:
    waveform = AnalogWaveform.create(10, np.int32)

    assert waveform.sample_count == waveform.capacity == len(waveform.raw_data) == 10
    assert waveform.dtype == np.int32
    assert_type(waveform, AnalogWaveform[np.int32])


def test___sample_count_and_dtype_str___create___creates_waveform_with_sample_count_and_dtype() -> (
    None
):
    waveform = AnalogWaveform.create(10, "i4")

    assert waveform.sample_count == waveform.capacity == len(waveform.raw_data) == 10
    assert waveform.dtype == np.int32
    assert_type(waveform, AnalogWaveform[Any])  # dtype not inferred from string


def test___sample_count_and_dtype_any___create___creates_waveform_with_sample_count_and_dtype() -> (
    None
):
    dtype: np.dtype[Any] = np.dtype(np.int32)
    waveform = AnalogWaveform.create(10, dtype)

    assert waveform.sample_count == waveform.capacity == len(waveform.raw_data) == 10
    assert waveform.dtype == np.int32
    assert_type(waveform, AnalogWaveform[Any])  # dtype not inferred from np.dtype[Any]


def test___sample_count_dtype_and_capacity___create___creates_waveform_with_sample_count_dtype_and_capacity() -> (
    None
):
    waveform = AnalogWaveform.create(10, np.int32, 20)

    assert waveform.sample_count == len(waveform.raw_data) == 10
    assert waveform.capacity == 20
    assert waveform.dtype == np.int32
    assert_type(waveform, AnalogWaveform[np.int32])


def test___float64_array___from_array_1d___creates_waveform_with_float64_dtype() -> None:
    data = np.array([1.1, 2.2, 3.3, 4.4, 5.5], np.float64)

    waveform = AnalogWaveform.from_array_1d(data)

    assert waveform.raw_data.tolist() == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert waveform.dtype == np.float64
    assert_type(waveform, AnalogWaveform[np.float64])


def test___int32_array___from_array_1d___creates_waveform_with_int32_dtype() -> None:
    data = np.array([1, 2, 3, 4, 5], np.int32)

    waveform = AnalogWaveform.from_array_1d(data)

    assert waveform.raw_data.tolist() == [1, 2, 3, 4, 5]
    assert waveform.dtype == np.int32
    assert_type(waveform, AnalogWaveform[np.int32])


def test___array_like___from_array_1d___raises_value_error() -> None:
    data = [1, 2, 3, 4, 5]

    with pytest.raises(ValueError) as exc:
        _ = AnalogWaveform.from_array_1d(data)

    assert exc.value.args[0] == "The dtype parameter must be specified for array-like objects."


def test___array_like_and_dtype___from_array_1d___creates_waveform_with_specified_dtype() -> None:
    data = [1, 2, 3, 4, 5]

    waveform = AnalogWaveform.from_array_1d(data, np.int32)

    assert waveform.raw_data.tolist() == data
    assert waveform.dtype == np.int32
    assert_type(waveform, AnalogWaveform[np.int32])


def test___copy___from_array_1d___creates_waveform_with_array_copy() -> None:
    data = np.array([1, 2, 3, 4, 5], np.int32)

    waveform = AnalogWaveform.from_array_1d(data, copy=True)

    assert waveform._data is not data
    assert waveform.raw_data.tolist() == data.tolist()


def test___no_copy___from_array_1d___creates_waveform_with_array_reference() -> None:
    data = np.array([1, 2, 3, 4, 5], np.int32)

    waveform = AnalogWaveform.from_array_1d(data, copy=False)

    assert waveform._data is data
    assert waveform.raw_data.tolist() == data.tolist()


def test___no_copy_array_like___from_array_1d___raises_value_error() -> None:
    data = [1, 2, 3, 4, 5]

    with pytest.raises(ValueError) as exc:
        _ = AnalogWaveform.from_array_1d(data, np.int32, copy=False)

    assert exc.value.args[0].startswith(
        "Unable to avoid copy while creating an array as requested."
    )


@pytest.mark.parametrize(
    "start_index, sample_count, expected_data",
    [
        (0, -1, [1, 2, 3, 4, 5]),
        (1, -1, [2, 3, 4, 5]),
        (4, -1, [5]),
        (5, -1, []),
        (0, 1, [1]),
        (0, 4, [1, 2, 3, 4]),
        (1, 1, [2]),
        (1, 3, [2, 3, 4]),
        (1, 4, [2, 3, 4, 5]),
    ],
)
def test___array_subset___from_array_1d___creates_waveform_with_array_subset(
    start_index: int, sample_count: int, expected_data: list[int]
) -> None:
    data = np.array([1, 2, 3, 4, 5], np.int32)

    waveform = AnalogWaveform.from_array_1d(
        data, start_index=start_index, sample_count=sample_count
    )

    assert waveform.raw_data.tolist() == expected_data


@pytest.mark.parametrize(
    "start_index, sample_count, expected_message",
    [
        (-2, -1, "Start index -2 is less than zero."),
        (-1, -1, "Start index -1 is less than zero."),
        (6, -1, "Start index 6 is greater than array length 5."),
        (0, -2, "Sample count -2 is less than zero."),
        (
            0,
            6,
            "Sample count 6 is greater than array length 5.",
        ),
        (
            1,
            5,
            "The capacity must be equal to or greater than the number of samples in the waveform.",
        ),
        (
            5,
            1,
            "The capacity must be equal to or greater than the number of samples in the waveform.",
        ),
    ],
)
def test___invalid_array_subset___from_array_1d___raises_value_error(
    start_index: int, sample_count: int, expected_message: str
) -> None:
    data = np.array([1, 2, 3, 4, 5], np.int32)

    with pytest.raises(ValueError) as exc:
        _ = AnalogWaveform.from_array_1d(data, start_index=start_index, sample_count=sample_count)

    assert exc.value.args[0] == expected_message


@pytest.mark.parametrize(
    "capacity, expected_data",
    [(3, [1, 2, 3]), (4, [1, 2, 3, 0]), (10, [1, 2, 3, 0, 0, 0, 0, 0, 0, 0])],
)
def test___waveform___set_capacity___resizes_array_and_pads_with_zeros(
    capacity: int, expected_data: list[int]
) -> None:
    waveform = AnalogWaveform.from_array_1d([1, 2, 3], np.int32)

    waveform.capacity = capacity

    assert waveform.capacity == capacity
    assert waveform.raw_data.tolist() == [1, 2, 3]
    assert waveform._data.tolist() == expected_data


@pytest.mark.parametrize(
    "capacity, expected_message",
    [
        (-2, "Capacity -2 is less than zero."),
        (-1, "Capacity -1 is less than zero."),
        (0, "The capacity must be equal to or greater than the number of samples in the waveform."),
        (2, "The capacity must be equal to or greater than the number of samples in the waveform."),
    ],
)
def test___invalid_capacity___set_capacity___raises_value_error(
    capacity: int, expected_message: str
) -> None:
    waveform = AnalogWaveform.from_array_1d([1, 2, 3], np.int32)

    with pytest.raises(ValueError) as exc:
        waveform.capacity = capacity

    assert exc.value.args[0] == expected_message


def test___waveform___set_channel_name___sets_extended_property() -> None:
    waveform = AnalogWaveform.create()

    waveform.channel_name = "Dev1/ai0"

    assert waveform.channel_name == "Dev1/ai0"
    assert waveform.extended_properties["NI_ChannelName"] == "Dev1/ai0"


def test___waveform___set_unit_description___sets_extended_property() -> None:
    waveform = AnalogWaveform.create()

    waveform.unit_description = "Volts"

    assert waveform.unit_description == "Volts"
    assert waveform.extended_properties["NI_UnitDescription"] == "Volts"


def test___waveform___set_undefined_property___raises_attribute_error() -> None:
    waveform = AnalogWaveform.create()

    with pytest.raises(AttributeError):
        waveform.undefined_property = "Whatever"  # type: ignore[attr-defined]


def test___waveform___take_weak_ref___references_waveform() -> None:
    waveform = AnalogWaveform.create()

    waveform_ref = weakref.ref(waveform)

    assert waveform_ref() is waveform
