from __future__ import annotations

import array
import itertools
import sys
import weakref
from typing import Any, SupportsIndex

import numpy as np
import pytest

from nitypes.waveform import AnalogWaveform

if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type


###############################################################################
# create
###############################################################################
def test___no_args___create___creates_empty_waveform_with_default_dtype() -> None:
    waveform = AnalogWaveform()

    assert waveform.sample_count == waveform.capacity == len(waveform.raw_data) == 0
    assert waveform.dtype == np.float64
    assert_type(waveform, AnalogWaveform[np.float64])


def test___sample_count___create___creates_waveform_with_sample_count_and_default_dtype() -> None:
    waveform = AnalogWaveform(10)

    assert waveform.sample_count == waveform.capacity == len(waveform.raw_data) == 10
    assert waveform.dtype == np.float64
    assert_type(waveform, AnalogWaveform[np.float64])


def test___sample_count_and_dtype___create___creates_waveform_with_sample_count_and_dtype() -> None:
    waveform = AnalogWaveform(10, np.int32)

    assert waveform.sample_count == waveform.capacity == len(waveform.raw_data) == 10
    assert waveform.dtype == np.int32
    assert_type(waveform, AnalogWaveform[np.int32])


def test___sample_count_and_dtype_str___create___creates_waveform_with_sample_count_and_dtype() -> (
    None
):
    waveform = AnalogWaveform(10, "i4")

    assert waveform.sample_count == waveform.capacity == len(waveform.raw_data) == 10
    assert waveform.dtype == np.int32
    assert_type(waveform, AnalogWaveform[Any])  # dtype not inferred from string


def test___sample_count_and_dtype_any___create___creates_waveform_with_sample_count_and_dtype() -> (
    None
):
    dtype: np.dtype[Any] = np.dtype(np.int32)
    waveform = AnalogWaveform(10, dtype)

    assert waveform.sample_count == waveform.capacity == len(waveform.raw_data) == 10
    assert waveform.dtype == np.int32
    assert_type(waveform, AnalogWaveform[Any])  # dtype not inferred from np.dtype[Any]


def test___sample_count_dtype_and_capacity___create___creates_waveform_with_sample_count_dtype_and_capacity() -> (
    None
):
    waveform = AnalogWaveform(10, np.int32, capacity=20)

    assert waveform.sample_count == len(waveform.raw_data) == 10
    assert waveform.capacity == 20
    assert waveform.dtype == np.int32
    assert_type(waveform, AnalogWaveform[np.int32])


def test___sample_count_and_unsupported_dtype___create___raises_type_error() -> None:
    with pytest.raises(TypeError) as exc:
        _ = AnalogWaveform(10, np.complex128)

    assert exc.value.args[0].startswith("The requested data type is not supported.")


###############################################################################
# from_array_1d
###############################################################################
def test___float64_ndarray___from_array_1d___creates_waveform_with_float64_dtype() -> None:
    data = np.array([1.1, 2.2, 3.3, 4.4, 5.5], np.float64)

    waveform = AnalogWaveform.from_array_1d(data)

    assert waveform.raw_data.tolist() == data.tolist()
    assert waveform.dtype == np.float64
    assert_type(waveform, AnalogWaveform[np.float64])


def test___int32_ndarray___from_array_1d___creates_waveform_with_int32_dtype() -> None:
    data = np.array([1, 2, 3, 4, 5], np.int32)

    waveform = AnalogWaveform.from_array_1d(data)

    assert waveform.raw_data.tolist() == data.tolist()
    assert waveform.dtype == np.int32
    assert_type(waveform, AnalogWaveform[np.int32])


def test___int32_array_with_dtype___from_array_1d___creates_waveform_with_specified_dtype() -> None:
    data = array.array("l", [1, 2, 3, 4, 5])

    waveform = AnalogWaveform.from_array_1d(data, np.int32)

    assert waveform.raw_data.tolist() == data.tolist()
    assert waveform.dtype == np.int32
    assert_type(waveform, AnalogWaveform[np.int32])


def test___int16_ndarray_with_mismatched_dtype___from_array_1d___creates_waveform_with_specified_dtype() -> (
    None
):
    data = np.array([1, 2, 3, 4, 5], np.int16)

    waveform = AnalogWaveform.from_array_1d(data, np.int32)

    assert waveform.raw_data.tolist() == data.tolist()
    assert waveform.dtype == np.int32
    assert_type(waveform, AnalogWaveform[np.int32])


def test___int_list_with_dtype___from_array_1d___creates_waveform_with_specified_dtype() -> None:
    data = [1, 2, 3, 4, 5]

    waveform = AnalogWaveform.from_array_1d(data, np.int32)

    assert waveform.raw_data.tolist() == data
    assert waveform.dtype == np.int32
    assert_type(waveform, AnalogWaveform[np.int32])


def test___int_list_with_dtype_str___from_array_1d___creates_waveform_with_specified_dtype() -> (
    None
):
    data = [1, 2, 3, 4, 5]

    waveform = AnalogWaveform.from_array_1d(data, "int32")

    assert waveform.raw_data.tolist() == data  # type: ignore[comparison-overlap]
    assert waveform.dtype == np.int32
    assert_type(waveform, AnalogWaveform[Any])  # dtype not inferred from string


def test___int32_ndarray_2d___from_array_1d___raises_value_error() -> None:
    data = np.array([[1, 2, 3], [4, 5, 6]], np.int32)

    with pytest.raises(ValueError) as exc:
        _ = AnalogWaveform.from_array_1d(data)

    assert exc.value.args[0].startswith(
        "The input array must be a one-dimensional array or sequence."
    )


def test___int_list_without_dtype___from_array_1d___raises_value_error() -> None:
    data = [1, 2, 3, 4, 5]

    with pytest.raises(ValueError) as exc:
        _ = AnalogWaveform.from_array_1d(data)

    assert exc.value.args[0].startswith(
        "You must specify a dtype when the input array is a sequence."
    )


def test___bytes___from_array_1d___raises_value_error() -> None:
    data = b"\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00"

    with pytest.raises(ValueError) as exc:
        _ = AnalogWaveform.from_array_1d(data, np.int32)

    assert exc.value.args[0].startswith("invalid literal for int() with base 10:")


def test___iterable___from_array_1d___raises_type_error() -> None:
    data = itertools.repeat(3)

    with pytest.raises(TypeError) as exc:
        _ = AnalogWaveform.from_array_1d(data, np.int32)  # type: ignore[call-overload]

    assert exc.value.args[0].startswith(
        "The input array must be a one-dimensional array or sequence."
    )


def test___ndarray_with_unsupported_dtype___from_array_1d___raises_type_error() -> None:
    data = np.zeros(3, np.complex128)

    with pytest.raises(TypeError) as exc:
        _ = AnalogWaveform.from_array_1d(data)

    assert exc.value.args[0].startswith("The requested data type is not supported.")


def test___copy___from_array_1d___creates_waveform_linked_to_different_buffer() -> None:
    data = np.array([1, 2, 3, 4, 5], np.int32)

    waveform = AnalogWaveform.from_array_1d(data, copy=True)

    assert waveform._data is not data
    assert waveform.raw_data.tolist() == data.tolist()
    data[:] = [5, 4, 3, 2, 1]
    assert waveform.raw_data.tolist() != data.tolist()


def test___int32_ndarray_no_copy___from_array_1d___creates_waveform_linked_to_same_buffer() -> None:
    data = np.array([1, 2, 3, 4, 5], np.int32)

    waveform = AnalogWaveform.from_array_1d(data, copy=False)

    assert waveform._data is data
    assert waveform.raw_data.tolist() == data.tolist()
    data[:] = [5, 4, 3, 2, 1]
    assert waveform.raw_data.tolist() == data.tolist()


def test___int32_array_no_copy___from_array_1d___creates_waveform_linked_to_same_buffer() -> None:
    data = array.array("l", [1, 2, 3, 4, 5])

    waveform = AnalogWaveform.from_array_1d(data, dtype=np.int32, copy=False)

    assert waveform.raw_data.tolist() == data.tolist()
    data[:] = array.array("l", [5, 4, 3, 2, 1])
    assert waveform.raw_data.tolist() == data.tolist()


def test___int_list_no_copy___from_array_1d___raises_value_error() -> None:
    data = [1, 2, 3, 4, 5]

    with pytest.raises(ValueError) as exc:
        _ = AnalogWaveform.from_array_1d(data, np.int32, copy=False)

    assert exc.value.args[0].startswith(
        "Unable to avoid copy while creating an array as requested."
    )


@pytest.mark.parametrize(
    "start_index, sample_count, expected_data",
    [
        (0, None, [1, 2, 3, 4, 5]),
        (1, None, [2, 3, 4, 5]),
        (4, None, [5]),
        (5, None, []),
        (0, 1, [1]),
        (0, 4, [1, 2, 3, 4]),
        (1, 1, [2]),
        (1, 3, [2, 3, 4]),
        (1, 4, [2, 3, 4, 5]),
    ],
)
def test___array_subset___from_array_1d___creates_waveform_with_array_subset(
    start_index: SupportsIndex, sample_count: SupportsIndex | None, expected_data: list[int]
) -> None:
    data = np.array([1, 2, 3, 4, 5], np.int32)

    waveform = AnalogWaveform.from_array_1d(
        data, start_index=start_index, sample_count=sample_count
    )

    assert waveform.raw_data.tolist() == expected_data


@pytest.mark.parametrize(
    "start_index, sample_count, expected_message",
    [
        (-2, None, "The start index must be a non-negative integer."),
        (-1, None, "The start index must be a non-negative integer."),
        (6, None, "The start index must be less than or equal to the input array length."),
        (0, -2, "The sample count must be a non-negative integer."),
        (0, -1, "The sample count must be a non-negative integer."),
        (
            0,
            6,
            "The sum of the start index and sample count must be less than or equal to the input array length.",
        ),
        (
            1,
            5,
            "The sum of the start index and sample count must be less than or equal to the input array length.",
        ),
        (
            5,
            1,
            "The sum of the start index and sample count must be less than or equal to the input array length.",
        ),
    ],
)
def test___invalid_array_subset___from_array_1d___raises_value_error(
    start_index: SupportsIndex, sample_count: SupportsIndex | None, expected_message: str
) -> None:
    data = np.array([1, 2, 3, 4, 5], np.int32)

    with pytest.raises(ValueError) as exc:
        _ = AnalogWaveform.from_array_1d(data, start_index=start_index, sample_count=sample_count)

    assert exc.value.args[0].startswith(expected_message)


###############################################################################
# from_array_2d
###############################################################################
def test___float64_ndarray___from_array_2d___creates_waveform_with_float64_dtype() -> None:
    data = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], np.float64)

    waveforms = AnalogWaveform.from_array_2d(data)

    assert len(waveforms) == 2
    for i in range(len(waveforms)):
        assert waveforms[i].raw_data.tolist() == data[i].tolist()
        assert waveforms[i].dtype == np.float64
        assert_type(waveforms[i], AnalogWaveform[np.float64])


def test___int32_ndarray___from_array_2d___creates_waveform_with_int32_dtype() -> None:
    data = np.array([[1, 2, 3], [4, 5, 6]], np.int32)

    waveforms = AnalogWaveform.from_array_2d(data)

    assert len(waveforms) == 2
    for i in range(len(waveforms)):
        assert waveforms[i].raw_data.tolist() == data[i].tolist()
        assert waveforms[i].dtype == np.int32
        assert_type(waveforms[i], AnalogWaveform[np.int32])


def test___int16_ndarray_with_mismatched_dtype___from_array_2d___creates_waveform_with_specified_dtype() -> (
    None
):
    data = np.array([[1, 2, 3], [4, 5, 6]], np.int16)

    waveforms = AnalogWaveform.from_array_2d(data, np.int32)

    assert len(waveforms) == 2
    for i in range(len(waveforms)):
        assert waveforms[i].raw_data.tolist() == data[i].tolist()
        assert waveforms[i].dtype == np.int32
        assert_type(waveforms[i], AnalogWaveform[np.int32])


def test___int32_array_list_with_dtype___from_array_2d___creates_waveform_with_specified_dtype() -> (
    None
):
    data = [array.array("l", [1, 2, 3]), array.array("l", [4, 5, 6])]

    waveforms = AnalogWaveform.from_array_2d(data, np.int32)

    assert len(waveforms) == 2
    for i in range(len(waveforms)):
        assert waveforms[i].raw_data.tolist() == data[i].tolist()
        assert waveforms[i].dtype == np.int32
        assert_type(waveforms[i], AnalogWaveform[np.int32])


def test___int_list_list_with_dtype___from_array_2d___creates_waveform_with_specified_dtype() -> (
    None
):
    data = [[1, 2, 3], [4, 5, 6]]

    waveforms = AnalogWaveform.from_array_2d(data, np.int32)

    assert len(waveforms) == 2
    for i in range(len(waveforms)):
        assert waveforms[i].raw_data.tolist() == data[i]
        assert waveforms[i].dtype == np.int32
        assert_type(waveforms[i], AnalogWaveform[np.int32])


def test___int_list_list_with_dtype_str___from_array_2d___creates_waveform_with_specified_dtype() -> (
    None
):
    data = [[1, 2, 3], [4, 5, 6]]

    waveforms = AnalogWaveform.from_array_2d(data, "int32")

    assert len(waveforms) == 2
    for i in range(len(waveforms)):
        assert waveforms[i].raw_data.tolist() == data[i]  # type: ignore[comparison-overlap]
        assert waveforms[i].dtype == np.int32
        assert_type(waveforms[i], AnalogWaveform[Any])  # dtype not inferred from string


def test___int32_ndarray_1d___from_array_2d___raises_value_error() -> None:
    data = np.array([1, 2, 3, 4, 5], np.int32)

    with pytest.raises(ValueError) as exc:
        _ = AnalogWaveform.from_array_2d(data)

    assert exc.value.args[0].startswith(
        "The input array must be a two-dimensional array or nested sequence."
    )


def test___int_list_list_without_dtype___from_array_2d___raises_value_error() -> None:
    data = [[1, 2, 3], [4, 5, 6]]

    with pytest.raises(ValueError) as exc:
        _ = AnalogWaveform.from_array_2d(data)

    assert exc.value.args[0].startswith(
        "You must specify a dtype when the input array is a sequence."
    )


def test___bytes_list___from_array_2d___raises_value_error() -> None:
    data = [
        b"\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00",
        b"\x04\x00\x00\x00\x05\x00\x00\x00\x06\x00\x00\x00",
    ]

    with pytest.raises(ValueError) as exc:
        _ = AnalogWaveform.from_array_2d(data, np.int32)

    assert exc.value.args[0].startswith("invalid literal for int() with base 10:")


def test___list_iterable___from_array_2d___raises_type_error() -> None:
    data = itertools.repeat([3])

    with pytest.raises(TypeError) as exc:
        _ = AnalogWaveform.from_array_2d(data, np.int32)  # type: ignore[call-overload]

    assert exc.value.args[0].startswith(
        "The input array must be a two-dimensional array or nested sequence."
    )


def test___iterable_list___from_array_2d___raises_type_error() -> None:
    data = [itertools.repeat(3), itertools.repeat(4)]

    with pytest.raises(TypeError) as exc:
        _ = AnalogWaveform.from_array_2d(data, np.int32)  # type: ignore[arg-type]

    assert exc.value.args[0].startswith("int() argument must be")


def test___ndarray_with_unsupported_dtype___from_array_2d___raises_type_error() -> None:
    data = np.zeros((2, 3), np.complex128)

    with pytest.raises(TypeError) as exc:
        _ = AnalogWaveform.from_array_2d(data)

    assert exc.value.args[0].startswith("The requested data type is not supported.")


def test___copy___from_array_2d___creates_waveform_linked_to_different_buffer() -> None:
    data = np.array([[1, 2, 3], [4, 5, 6]], np.int32)

    waveforms = AnalogWaveform.from_array_2d(data, copy=True)

    assert len(waveforms) == 2
    for i in range(len(waveforms)):
        assert waveforms[i].raw_data.tolist() == data[i].tolist()
    data[0][:] = [3, 2, 1]
    data[1][:] = [6, 5, 4]
    for i in range(len(waveforms)):
        assert waveforms[i].raw_data.tolist() != data[i].tolist()


def test___int32_ndarray_no_copy___from_array_2d___creates_waveform_linked_to_same_buffer() -> None:
    data = np.array([[1, 2, 3], [4, 5, 6]], np.int32)

    waveforms = AnalogWaveform.from_array_2d(data, copy=False)

    assert len(waveforms) == 2
    for i in range(len(waveforms)):
        assert waveforms[i].raw_data.tolist() == data[i].tolist()
    data[0][:] = [3, 2, 1]
    data[1][:] = [6, 5, 4]
    for i in range(len(waveforms)):
        assert waveforms[i].raw_data.tolist() == data[i].tolist()


def test___int32_array_list_no_copy___from_array_2d___creates_waveform_linked_to_same_buffer() -> (
    None
):
    data = [array.array("l", [1, 2, 3]), array.array("l", [4, 5, 6])]

    waveforms = AnalogWaveform.from_array_2d(data, dtype=np.int32, copy=False)

    assert len(waveforms) == 2
    for i in range(len(waveforms)):
        assert waveforms[i].raw_data.tolist() == data[i].tolist()
    data[0][:] = array.array("l", [3, 2, 1])
    data[1][:] = array.array("l", [6, 5, 4])
    for i in range(len(waveforms)):
        assert waveforms[i].raw_data.tolist() == data[i].tolist()


def test___int_list_list_no_copy___from_array_2d___raises_value_error() -> None:
    data = [[1, 2, 3], [4, 5, 6]]

    with pytest.raises(ValueError) as exc:
        _ = AnalogWaveform.from_array_2d(data, np.int32, copy=False)

    assert exc.value.args[0].startswith(
        "Unable to avoid copy while creating an array as requested."
    )


@pytest.mark.parametrize(
    "start_index, sample_count, expected_data",
    [
        (0, None, [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]),
        (1, None, [[2, 3, 4, 5], [7, 8, 9, 10]]),
        (4, None, [[5], [10]]),
        (5, None, [[], []]),
        (0, 1, [[1], [6]]),
        (0, 4, [[1, 2, 3, 4], [6, 7, 8, 9]]),
        (1, 1, [[2], [7]]),
        (1, 3, [[2, 3, 4], [7, 8, 9]]),
        (1, 4, [[2, 3, 4, 5], [7, 8, 9, 10]]),
    ],
)
def test___array_subset___from_array_2d___creates_waveform_with_array_subset(
    start_index: SupportsIndex, sample_count: SupportsIndex | None, expected_data: list[list[int]]
) -> None:
    data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], np.int32)

    waveforms = AnalogWaveform.from_array_2d(
        data, start_index=start_index, sample_count=sample_count
    )

    assert len(waveforms) == 2
    for i in range(len(waveforms)):
        assert waveforms[i].raw_data.tolist() == expected_data[i]


@pytest.mark.parametrize(
    "start_index, sample_count, expected_message",
    [
        (-2, None, "The start index must be a non-negative integer."),
        (-1, None, "The start index must be a non-negative integer."),
        (6, None, "The start index must be less than or equal to the input array length."),
        (0, -2, "The sample count must be a non-negative integer."),
        (0, -1, "The sample count must be a non-negative integer."),
        (
            0,
            6,
            "The sum of the start index and sample count must be less than or equal to the input array length.",
        ),
        (
            1,
            5,
            "The sum of the start index and sample count must be less than or equal to the input array length.",
        ),
        (
            5,
            1,
            "The sum of the start index and sample count must be less than or equal to the input array length.",
        ),
    ],
)
def test___invalid_array_subset___from_array_2d___raises_value_error(
    start_index: SupportsIndex, sample_count: SupportsIndex | None, expected_message: str
) -> None:
    data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], np.int32)

    with pytest.raises(ValueError) as exc:
        _ = AnalogWaveform.from_array_2d(data, start_index=start_index, sample_count=sample_count)

    assert exc.value.args[0].startswith(expected_message)


###############################################################################
# capacity
###############################################################################
@pytest.mark.parametrize(
    "capacity, expected_data",
    [(3, [1, 2, 3]), (4, [1, 2, 3, 0]), (10, [1, 2, 3, 0, 0, 0, 0, 0, 0, 0])],
)
def test___waveform___set_capacity___resizes_array_and_pads_with_zeros(
    capacity: int, expected_data: list[int]
) -> None:
    data = [1, 2, 3]
    waveform = AnalogWaveform.from_array_1d(data, np.int32)

    waveform.capacity = capacity

    assert waveform.capacity == capacity
    assert waveform.raw_data.tolist() == data
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
    waveform = AnalogWaveform.from_array_1d(data, np.int32)

    with pytest.raises(ValueError) as exc:
        waveform.capacity = capacity

    assert exc.value.args[0].startswith(expected_message)


def test___referenced_array___set_capacity___reference_sees_size_change() -> None:
    data = np.array([1, 2, 3], np.int32)
    waveform = AnalogWaveform.from_array_1d(data, np.int32, copy=False)

    waveform.capacity = 10

    assert len(data) == 10
    assert waveform.capacity == 10
    assert data.tolist() == [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]
    assert waveform.raw_data.tolist() == [1, 2, 3]
    assert waveform._data.tolist() == [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]


def test___array_with_external_buffer___set_capacity___raises_value_error() -> None:
    data = array.array("l", [1, 2, 3])
    waveform = AnalogWaveform.from_array_1d(data, np.int32, copy=False)

    with pytest.raises(ValueError) as exc:
        waveform.capacity = 10

    assert exc.value.args[0].startswith("cannot resize this array: it does not own its data")


###############################################################################
# misc
###############################################################################
def test___waveform___set_channel_name___sets_extended_property() -> None:
    waveform = AnalogWaveform()

    waveform.channel_name = "Dev1/ai0"

    assert waveform.channel_name == "Dev1/ai0"
    assert waveform.extended_properties["NI_ChannelName"] == "Dev1/ai0"


def test___invalid_type___set_channel_name___raises_type_error() -> None:
    waveform = AnalogWaveform()

    with pytest.raises(TypeError) as exc:
        waveform.channel_name = 1  # type: ignore[assignment]

    assert exc.value.args[0].startswith("The channel name must be a str.")


def test___waveform___set_unit_description___sets_extended_property() -> None:
    waveform = AnalogWaveform()

    waveform.unit_description = "Volts"

    assert waveform.unit_description == "Volts"
    assert waveform.extended_properties["NI_UnitDescription"] == "Volts"


def test___invalid_type___set_unit_description___raises_type_error() -> None:
    waveform = AnalogWaveform()

    with pytest.raises(TypeError) as exc:
        waveform.unit_description = None  # type: ignore[assignment]

    assert exc.value.args[0].startswith("The unit description must be a str.")


def test___waveform___set_undefined_property___raises_attribute_error() -> None:
    waveform = AnalogWaveform()

    with pytest.raises(AttributeError):
        waveform.undefined_property = "Whatever"  # type: ignore[attr-defined]


def test___waveform___take_weak_ref___references_waveform() -> None:
    waveform = AnalogWaveform()

    waveform_ref = weakref.ref(waveform)

    assert waveform_ref() is waveform
