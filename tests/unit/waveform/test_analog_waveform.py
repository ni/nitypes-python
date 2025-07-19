from __future__ import annotations

import array
import copy
import datetime as dt
import itertools
import pickle
import sys
import weakref
from collections.abc import Sequence
from typing import Any, SupportsIndex, Union

import hightime as ht
import numpy as np
import numpy.typing as npt
import pytest
from packaging.version import Version
from typing_extensions import assert_type

import nitypes.bintime as bt
import nitypes.waveform.errors as wfmex
import nitypes.waveform.warnings as wfmwarn
from nitypes.waveform import (
    NO_SCALING,
    AnalogWaveform,
    LinearScaleMode,
    NoneScaleMode,
    SampleIntervalMode,
    ScaleMode,
    Timing,
)


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


@pytest.mark.parametrize("dtype", [np.complex128, np.str_, np.void, "i2, i2"])
def test___sample_count_and_unsupported_dtype___create___raises_type_error(
    dtype: npt.DTypeLike,
) -> None:
    with pytest.raises(TypeError) as exc:
        _ = AnalogWaveform(10, dtype)

    assert exc.value.args[0].startswith("The requested data type is not supported.")


def test___dtype_str_with_unsupported_traw_hint___create___mypy_type_var_warning() -> None:
    waveform1: AnalogWaveform[np.complex128] = AnalogWaveform(dtype="int32")  # type: ignore[type-var]
    waveform2: AnalogWaveform[np.str_] = AnalogWaveform(dtype="int32")  # type: ignore[type-var]
    waveform3: AnalogWaveform[np.void] = AnalogWaveform(dtype="int32")  # type: ignore[type-var]
    _ = waveform1, waveform2, waveform3


def test___dtype_str_with_traw_hint___create___narrows_traw() -> None:
    waveform: AnalogWaveform[np.int32] = AnalogWaveform(dtype="int32")

    assert_type(waveform, AnalogWaveform[np.int32])


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
    data = array.array("i", [1, 2, 3, 4, 5])

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

    assert waveform.raw_data.tolist() == data
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
    data = np.zeros(3, np.str_)

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
    data = array.array("i", [1, 2, 3, 4, 5])

    waveform = AnalogWaveform.from_array_1d(data, dtype=np.int32, copy=False)

    assert waveform.raw_data.tolist() == data.tolist()
    data[:] = array.array("i", [5, 4, 3, 2, 1])
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
    "start_index, sample_count, expected_message, exception_type",
    [
        (-2, None, "The start index must be a non-negative integer.", ValueError),
        (-1, None, "The start index must be a non-negative integer.", ValueError),
        (
            6,
            None,
            "The start index must be less than or equal to the input array length.",
            wfmex.StartIndexTooLargeError,
        ),
        (0, -2, "The sample count must be a non-negative integer.", ValueError),
        (0, -1, "The sample count must be a non-negative integer.", ValueError),
        (
            0,
            6,
            "The sum of the start index and sample count must be less than or equal to the input array length.",
            wfmex.StartIndexOrSampleCountTooLargeError,
        ),
        (
            1,
            5,
            "The sum of the start index and sample count must be less than or equal to the input array length.",
            wfmex.StartIndexOrSampleCountTooLargeError,
        ),
        (
            5,
            1,
            "The sum of the start index and sample count must be less than or equal to the input array length.",
            wfmex.StartIndexOrSampleCountTooLargeError,
        ),
    ],
)
def test___invalid_array_subset___from_array_1d___raises_correct_error(
    start_index: SupportsIndex,
    sample_count: SupportsIndex | None,
    expected_message: str,
    exception_type: type[Exception],
) -> None:
    data = np.array([1, 2, 3, 4, 5], np.int32)

    with pytest.raises(exception_type) as exc:
        _ = AnalogWaveform.from_array_1d(data, start_index=start_index, sample_count=sample_count)

    assert exc.value.args[0].startswith(expected_message)


###############################################################################
# from_array_2d
###############################################################################
def test___float64_ndarray___from_array_2d___creates_waveform_with_float64_dtype() -> None:
    data = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], np.float64)

    waveforms = AnalogWaveform.from_array_2d(data)

    assert_type(waveforms, Sequence[AnalogWaveform[np.float64]])
    assert len(waveforms) == 2
    for i in range(len(waveforms)):
        assert waveforms[i].raw_data.tolist() == data[i].tolist()
        assert waveforms[i].dtype == np.float64
        assert_type(waveforms[i], AnalogWaveform[np.float64])


def test___int32_ndarray___from_array_2d___creates_waveform_with_int32_dtype() -> None:
    data = np.array([[1, 2, 3], [4, 5, 6]], np.int32)

    waveforms = AnalogWaveform.from_array_2d(data)

    assert_type(waveforms, Sequence[AnalogWaveform[np.int32]])
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

    assert_type(waveforms, Sequence[AnalogWaveform[np.int32]])
    assert len(waveforms) == 2
    for i in range(len(waveforms)):
        assert waveforms[i].raw_data.tolist() == data[i].tolist()
        assert waveforms[i].dtype == np.int32
        assert_type(waveforms[i], AnalogWaveform[np.int32])


def test___int32_array_list_with_dtype___from_array_2d___creates_waveform_with_specified_dtype() -> (
    None
):
    data = [array.array("i", [1, 2, 3]), array.array("i", [4, 5, 6])]

    waveforms = AnalogWaveform.from_array_2d(data, np.int32)

    assert_type(waveforms, Sequence[AnalogWaveform[np.int32]])
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

    assert_type(waveforms, Sequence[AnalogWaveform[np.int32]])
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

    assert_type(waveforms, Sequence[AnalogWaveform[Any]])  # dtype not inferred from string
    assert len(waveforms) == 2
    for i in range(len(waveforms)):
        assert waveforms[i].raw_data.tolist() == data[i]
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
    data = np.zeros((2, 3), np.str_)

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
    data = [array.array("i", [1, 2, 3]), array.array("i", [4, 5, 6])]

    waveforms = AnalogWaveform.from_array_2d(data, dtype=np.int32, copy=False)

    assert len(waveforms) == 2
    for i in range(len(waveforms)):
        assert waveforms[i].raw_data.tolist() == data[i].tolist()
    data[0][:] = array.array("i", [3, 2, 1])
    data[1][:] = array.array("i", [6, 5, 4])
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
    "start_index, sample_count, expected_message, exception_type",
    [
        (-2, None, "The start index must be a non-negative integer.", ValueError),
        (-1, None, "The start index must be a non-negative integer.", ValueError),
        (
            6,
            None,
            "The start index must be less than or equal to the input array length.",
            wfmex.StartIndexTooLargeError,
        ),
        (0, -2, "The sample count must be a non-negative integer.", ValueError),
        (0, -1, "The sample count must be a non-negative integer.", ValueError),
        (
            0,
            6,
            "The sum of the start index and sample count must be less than or equal to the input array length.",
            wfmex.StartIndexOrSampleCountTooLargeError,
        ),
        (
            1,
            5,
            "The sum of the start index and sample count must be less than or equal to the input array length.",
            wfmex.StartIndexOrSampleCountTooLargeError,
        ),
        (
            5,
            1,
            "The sum of the start index and sample count must be less than or equal to the input array length.",
            wfmex.StartIndexOrSampleCountTooLargeError,
        ),
    ],
)
def test___invalid_array_subset___from_array_2d___raises_correct_error(
    start_index: SupportsIndex,
    sample_count: SupportsIndex | None,
    expected_message: str,
    exception_type: type[Exception],
) -> None:
    data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], np.int32)

    with pytest.raises(exception_type) as exc:
        _ = AnalogWaveform.from_array_2d(data, start_index=start_index, sample_count=sample_count)

    assert exc.value.args[0].startswith(expected_message)


###############################################################################
# raw_data
###############################################################################
def test___int32_waveform___raw_data___returns_int32_data() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2, 3], np.int32)

    raw_data = waveform.raw_data

    assert_type(raw_data, npt.NDArray[np.int32])
    assert isinstance(raw_data, np.ndarray) and raw_data.dtype == np.int32
    assert list(raw_data) == [0, 1, 2, 3]


def test___int32_waveform_with_linear_scale___raw_data___returns_int32_data() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2, 3], np.int32)
    waveform.scale_mode = LinearScaleMode(2.0, 0.5)

    raw_data = waveform.raw_data

    assert_type(raw_data, npt.NDArray[np.int32])
    assert isinstance(raw_data, np.ndarray) and raw_data.dtype == np.int32
    assert list(raw_data) == [0, 1, 2, 3]


###############################################################################
# get_raw_data
###############################################################################
def test___int32_waveform___get_raw_data___returns_raw_data() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2, 3], np.int32)

    scaled_data = waveform.get_raw_data()

    assert_type(scaled_data, npt.NDArray[np.int32])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.int32
    assert list(scaled_data) == [0, 1, 2, 3]


def test___int32_waveform_with_linear_scale___get_raw_data___returns_raw_data() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2, 3], np.int32)
    waveform.scale_mode = LinearScaleMode(2.0, 0.5)

    scaled_data = waveform.get_raw_data()

    assert_type(scaled_data, npt.NDArray[np.int32])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.int32
    assert list(scaled_data) == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "start_index, sample_count, expected_raw_data",
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
def test___array_subset___get_raw_data___returns_array_subset(
    start_index: int, sample_count: int, expected_raw_data: list[int]
) -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2, 3], np.int32)
    waveform.scale_mode = LinearScaleMode(2.0, 0.5)

    scaled_data = waveform.get_raw_data(start_index=start_index, sample_count=sample_count)

    assert_type(scaled_data, npt.NDArray[np.int32])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.int32
    assert list(scaled_data) == expected_raw_data


@pytest.mark.parametrize(
    "start_index, sample_count, expected_message, exception_type",
    [
        (
            5,
            None,
            "The start index must be less than or equal to the number of samples in the waveform.",
            wfmex.StartIndexTooLargeError,
        ),
        (
            0,
            5,
            "The sum of the start index and sample count must be less than or equal to the number of samples in the waveform.",
            wfmex.StartIndexOrSampleCountTooLargeError,
        ),
        (
            4,
            1,
            "The sum of the start index and sample count must be less than or equal to the number of samples in the waveform.",
            wfmex.StartIndexOrSampleCountTooLargeError,
        ),
    ],
)
def test___invalid_array_subset___get_raw_data___returns_array_subset(
    start_index: int, sample_count: int, expected_message: str, exception_type: type[Exception]
) -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2, 3], np.int32)
    waveform.scale_mode = LinearScaleMode(2.0, 0.5)

    with pytest.raises(exception_type) as exc:
        _ = waveform.get_raw_data(start_index=start_index, sample_count=sample_count)

    assert exc.value.args[0].startswith(expected_message)


###############################################################################
# scaled_data
###############################################################################
def test___int32_waveform___scaled_data___converts_to_float64() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2, 3], np.int32)

    scaled_data = waveform.scaled_data

    assert_type(scaled_data, npt.NDArray[np.float64])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.float64
    assert list(scaled_data) == [0, 1, 2, 3]


def test___int32_waveform_with_linear_scale___scaled_data___applies_linear_scale() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2, 3], np.int32)
    waveform.scale_mode = LinearScaleMode(2.0, 0.5)

    scaled_data = waveform.scaled_data

    assert_type(scaled_data, npt.NDArray[np.float64])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.float64
    assert list(scaled_data) == [0.5, 2.5, 4.5, 6.5]


###############################################################################
# get_scaled_data
###############################################################################
def test___int32_waveform___get_scaled_data___converts_to_float64() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2, 3], np.int32)

    scaled_data = waveform.get_scaled_data()

    assert_type(scaled_data, npt.NDArray[np.float64])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.float64
    assert list(scaled_data) == [0, 1, 2, 3]


def test___int32_waveform_with_linear_scale___get_scaled_data___applies_linear_scale() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2, 3], np.int32)
    waveform.scale_mode = LinearScaleMode(2.0, 0.5)

    scaled_data = waveform.get_scaled_data()

    assert_type(scaled_data, npt.NDArray[np.float64])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.float64
    assert list(scaled_data) == [0.5, 2.5, 4.5, 6.5]


def test___float32_dtype___get_scaled_data___converts_to_float32() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2, 3], np.int32)
    waveform.scale_mode = LinearScaleMode(2.0, 0.5)

    scaled_data = waveform.get_scaled_data(np.float32)

    assert_type(scaled_data, npt.NDArray[np.float32])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.float32
    assert list(scaled_data) == [0.5, 2.5, 4.5, 6.5]


@pytest.mark.parametrize(
    "waveform_dtype",
    [
        np.float32,
        np.float64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ],
)
@pytest.mark.parametrize("scaled_dtype", [np.float32, np.float64])
def test___varying_dtype___get_scaled_data___converts_to_requested_dtype(
    waveform_dtype: npt.DTypeLike, scaled_dtype: npt.DTypeLike
) -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2, 3], waveform_dtype)
    waveform.scale_mode = LinearScaleMode(3.0, 4.0)

    scaled_data = waveform.get_scaled_data(scaled_dtype)

    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == scaled_dtype
    assert list(scaled_data) == [4.0, 7.0, 10.0, 13.0]


def test___unsupported_dtype___get_scaled_data___raises_type_error() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2, 3], np.int32)
    waveform.scale_mode = LinearScaleMode(3.0, 4.0)

    with pytest.raises(TypeError) as exc:
        _ = waveform.get_scaled_data(np.int32)

    assert exc.value.args[0].startswith("The requested data type is not supported.")
    assert "Data type: int32" in exc.value.args[0]
    assert "Supported data types: float32, float64" in exc.value.args[0]


def test___array_subset___get_scaled_data___returns_array_subset() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2, 3], np.int32)
    waveform.scale_mode = LinearScaleMode(2.0, 0.5)

    scaled_data = waveform.get_scaled_data(start_index=1, sample_count=2)

    assert_type(scaled_data, npt.NDArray[np.float64])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.float64
    assert list(scaled_data) == [2.5, 4.5]


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
    "capacity, expected_message, exception_type",
    [
        (-2, "The capacity must be a non-negative integer.", ValueError),
        (-1, "The capacity must be a non-negative integer.", ValueError),
        (
            0,
            "The capacity must be equal to or greater than the number of samples in the waveform.",
            wfmex.CapacityTooSmallError,
        ),
        (
            2,
            "The capacity must be equal to or greater than the number of samples in the waveform.",
            wfmex.CapacityTooSmallError,
        ),
    ],
)
def test___invalid_capacity___set_capacity___raises_correct_error(
    capacity: int, expected_message: str, exception_type: type[Exception]
) -> None:
    data = [1, 2, 3]
    waveform = AnalogWaveform.from_array_1d(data, np.int32)

    with pytest.raises(exception_type) as exc:
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
    data = array.array("i", [1, 2, 3])
    waveform = AnalogWaveform.from_array_1d(data, np.int32, copy=False)

    with pytest.raises(ValueError) as exc:
        waveform.capacity = 10

    assert exc.value.args[0].startswith("cannot resize this array: it does not own its data")


###############################################################################
# extended properties
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


###############################################################################
# timing
###############################################################################
def test___waveform___has_empty_timing() -> None:
    waveform = AnalogWaveform()

    assert waveform.timing is Timing.empty


def test___bintime___waveform_with_timing___static_type_erased() -> None:
    sample_interval = bt.TimeDelta(1e-3)
    timestamp = bt.DateTime.now(dt.timezone.utc)
    time_offset = bt.TimeDelta(1e-6)
    waveform = AnalogWaveform(
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
    waveform = AnalogWaveform(
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
    waveform = AnalogWaveform(
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
    waveform = AnalogWaveform(timing=timing)

    assert waveform.timing.sample_interval.total_seconds() == pytest.approx(1e-3)
    assert (
        waveform.timing.timestamp.year,
        waveform.timing.timestamp.month,
        waveform.timing.timestamp.day,
    ) == (2025, 1, 1)
    assert waveform.timing.time_offset.total_seconds() == pytest.approx(1e-6)


###############################################################################
# scale_mode
###############################################################################
def test___waveform___scale_mode___defaults_to_no_scaling() -> None:
    waveform = AnalogWaveform()

    assert_type(waveform.scale_mode, ScaleMode)
    assert isinstance(waveform.scale_mode, NoneScaleMode)
    assert waveform.scale_mode is NO_SCALING


###############################################################################
# append array
###############################################################################
def test___empty_ndarray___append___no_effect() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.int32)
    array = np.array([], np.int32)

    waveform.append(array)

    assert list(waveform.raw_data) == [0, 1, 2]


def test___int32_ndarray___append___appends_array() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.int32)
    array = np.array([3, 4, 5], np.int32)

    waveform.append(array)

    assert list(waveform.raw_data) == [0, 1, 2, 3, 4, 5]


def test___float64_ndarray___append___appends_array() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.float64)
    array = np.array([3, 4, 5], np.float64)

    waveform.append(array)

    assert list(waveform.raw_data) == [0, 1, 2, 3, 4, 5]


def test___ndarray_with_mismatched_dtype___append___raises_correct_error() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.float64)
    array = np.array([3, 4, 5], np.int32)

    with pytest.raises(wfmex.DatatypeMismatchError) as exc:
        waveform.append(array)  # type: ignore[arg-type]

    assert exc.value.args[0].startswith(
        "The data type of the input array must match the waveform data type."
    )


def test___ndarray_2d___append___raises_value_error() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.float64)
    array = np.array([[3, 4, 5], [6, 7, 8]], np.float64)

    with pytest.raises(ValueError) as exc:
        waveform.append(array)

    assert exc.value.args[0].startswith("The input array must be a one-dimensional array.")


def test___irregular_waveform_and_int32_ndarray_with_timestamps___append___appends_array() -> None:
    start_time = dt.datetime.now(dt.timezone.utc)
    waveform_offsets = [dt.timedelta(0), dt.timedelta(1), dt.timedelta(2)]
    waveform_timestamps = [start_time + offset for offset in waveform_offsets]
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.int32)
    waveform.timing = Timing.create_with_irregular_interval(waveform_timestamps)
    array_offsets = [dt.timedelta(3), dt.timedelta(4), dt.timedelta(5)]
    array_timestamps = [start_time + offset for offset in array_offsets]
    array = np.array([3, 4, 5], np.int32)

    waveform.append(array, array_timestamps)

    assert list(waveform.raw_data) == [0, 1, 2, 3, 4, 5]
    assert waveform.timing.sample_interval_mode == SampleIntervalMode.IRREGULAR
    assert waveform.timing._timestamps == waveform_timestamps + array_timestamps


def test___irregular_waveform_and_int32_ndarray_without_timestamps___append___raises_timing_mismatch_error_and_does_not_append() -> (
    None
):
    start_time = dt.datetime.now(dt.timezone.utc)
    waveform_offsets = [dt.timedelta(0), dt.timedelta(1), dt.timedelta(2)]
    waveform_timestamps = [start_time + offset for offset in waveform_offsets]
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.int32)
    waveform.timing = Timing.create_with_irregular_interval(waveform_timestamps)
    array = np.array([3, 4, 5], np.int32)

    with pytest.raises(wfmex.TimingMismatchError) as exc:
        waveform.append(array)

    assert exc.value.args[0].startswith(
        "The timestamps argument is required when appending to a waveform with irregular timing."
    )
    assert list(waveform.raw_data) == [0, 1, 2]
    assert waveform.timing.sample_interval_mode == SampleIntervalMode.IRREGULAR
    assert waveform.timing._timestamps == waveform_timestamps


def test___irregular_waveform_and_int32_ndarray_with_wrong_timestamp_count___append___raises_correct_error_and_does_not_append() -> (
    None
):
    start_time = dt.datetime.now(dt.timezone.utc)
    waveform_offsets = [dt.timedelta(0), dt.timedelta(1), dt.timedelta(2)]
    waveform_timestamps = [start_time + offset for offset in waveform_offsets]
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.int32)
    waveform.timing = Timing.create_with_irregular_interval(waveform_timestamps)
    array_offsets = [dt.timedelta(3), dt.timedelta(4)]
    array_timestamps = [start_time + offset for offset in array_offsets]
    array = np.array([3, 4, 5], np.int32)

    with pytest.raises(wfmex.IrregularTimestampCountMismatchError) as exc:
        waveform.append(array, array_timestamps)

    assert exc.value.args[0].startswith(
        "The number of irregular timestamps must be equal to the input array length."
    )
    assert list(waveform.raw_data) == [0, 1, 2]
    assert waveform.timing.sample_interval_mode == SampleIntervalMode.IRREGULAR
    assert waveform.timing._timestamps == waveform_timestamps


def test___regular_waveform_and_int32_ndarray_with_timestamps___append___raises_value_error_and_does_not_append() -> (
    None
):
    start_time = dt.datetime.now(dt.timezone.utc)
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.int32)
    waveform.timing = Timing.create_with_regular_interval(dt.timedelta(milliseconds=1))
    array_offsets = [dt.timedelta(3), dt.timedelta(4), dt.timedelta(5)]
    array_timestamps = [start_time + offset for offset in array_offsets]
    array = np.array([3, 4, 5], np.int32)

    with pytest.raises(ValueError) as exc:
        waveform.append(array, array_timestamps)

    assert exc.value.args[0].startswith("The timestamps argument is not supported.")
    assert list(waveform.raw_data) == [0, 1, 2]
    assert waveform.timing.sample_interval_mode == SampleIntervalMode.REGULAR
    assert waveform.timing.sample_interval == dt.timedelta(milliseconds=1)


###############################################################################
# append waveform
###############################################################################
def test___empty_waveform___append___no_effect() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.int32)
    other = AnalogWaveform(dtype=np.int32)

    waveform.append(other)

    assert list(waveform.raw_data) == [0, 1, 2]


def test___int32_waveform___append___appends_waveform() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.int32)
    other = AnalogWaveform.from_array_1d([3, 4, 5], np.int32)

    waveform.append(other)

    assert list(waveform.raw_data) == [0, 1, 2, 3, 4, 5]


def test___float64_waveform___append___appends_waveform() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.float64)
    other = AnalogWaveform.from_array_1d([3, 4, 5], np.float64)

    waveform.append(other)

    assert list(waveform.raw_data) == [0, 1, 2, 3, 4, 5]


def test___waveform_with_mismatched_dtype___append___raises_correct_error() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.float64)
    other = AnalogWaveform.from_array_1d([3, 4, 5], np.int32)

    with pytest.raises(wfmex.DatatypeMismatchError) as exc:
        waveform.append(other)  # type: ignore[arg-type]

    assert exc.value.args[0].startswith(
        "The data type of the input waveform must match the waveform data type."
    )


def test___irregular_waveform_and_irregular_waveform___append___appends_waveform() -> None:
    start_time = dt.datetime.now(dt.timezone.utc)
    waveform_offsets = [dt.timedelta(0), dt.timedelta(1), dt.timedelta(2)]
    waveform_timestamps = [start_time + offset for offset in waveform_offsets]
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.int32)
    waveform.timing = Timing.create_with_irregular_interval(waveform_timestamps)
    other_offsets = [dt.timedelta(3), dt.timedelta(4), dt.timedelta(5)]
    other_timestamps = [start_time + offset for offset in other_offsets]
    other = AnalogWaveform.from_array_1d([3, 4, 5], np.int32)
    other.timing = Timing.create_with_irregular_interval(other_timestamps)

    waveform.append(other)

    assert list(waveform.raw_data) == [0, 1, 2, 3, 4, 5]
    assert waveform.timing.sample_interval_mode == SampleIntervalMode.IRREGULAR
    assert waveform.timing._timestamps == waveform_timestamps + other_timestamps


def test___irregular_waveform_and_regular_waveform___append___raises_correct_error() -> None:
    start_time = dt.datetime.now(dt.timezone.utc)
    waveform_offsets = [dt.timedelta(0), dt.timedelta(1), dt.timedelta(2)]
    waveform_timestamps = [start_time + offset for offset in waveform_offsets]
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.int32)
    waveform.timing = Timing.create_with_irregular_interval(waveform_timestamps)
    other = AnalogWaveform.from_array_1d([3, 4, 5], np.int32)

    with pytest.raises(wfmex.SampleIntervalModeMismatchError) as exc:
        waveform.append(other)

    assert exc.value.args[0].startswith(
        "The timing of one or more waveforms does not match the timing of the current waveform."
    )
    assert list(waveform.raw_data) == [0, 1, 2]
    assert waveform.timing.sample_interval_mode == SampleIntervalMode.IRREGULAR
    assert waveform.timing._timestamps == waveform_timestamps


def test___regular_waveform_and_irregular_waveform___append___raises_correct_error() -> None:
    start_time = dt.datetime.now(dt.timezone.utc)
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.int32)
    waveform.timing = Timing.create_with_regular_interval(dt.timedelta(milliseconds=1))
    other_offsets = [dt.timedelta(3), dt.timedelta(4), dt.timedelta(5)]
    other_timestamps = [start_time + offset for offset in other_offsets]
    other = AnalogWaveform.from_array_1d([3, 4, 5], np.int32)
    other.timing = Timing.create_with_irregular_interval(other_timestamps)

    with pytest.raises(wfmex.SampleIntervalModeMismatchError) as exc:
        waveform.append(other)

    assert exc.value.args[0].startswith(
        "The timing of one or more waveforms does not match the timing of the current waveform."
    )
    assert list(waveform.raw_data) == [0, 1, 2]
    assert waveform.timing.sample_interval_mode == SampleIntervalMode.REGULAR
    assert waveform.timing.sample_interval == dt.timedelta(milliseconds=1)


def test___regular_waveform_and_regular_waveform_with_different_sample_interval___append___appends_waveform_with_timing_mismatch_warning() -> (
    None
):
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.int32)
    waveform.timing = Timing.create_with_regular_interval(dt.timedelta(milliseconds=1))
    other = AnalogWaveform.from_array_1d([3, 4, 5], np.int32)
    other.timing = Timing.create_with_regular_interval(dt.timedelta(milliseconds=2))

    with pytest.warns(wfmwarn.TimingMismatchWarning):
        waveform.append(other)

    assert list(waveform.raw_data) == [0, 1, 2, 3, 4, 5]
    assert waveform.timing.sample_interval_mode == SampleIntervalMode.REGULAR
    assert waveform.timing.sample_interval == dt.timedelta(milliseconds=1)


def test___regular_waveform_and_regular_waveform_with_different_extended_properties___append___merges_extended_properties() -> (
    None
):
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.int32)
    waveform.extended_properties["A"] = 1
    waveform.extended_properties["B"] = 2
    other = AnalogWaveform.from_array_1d([3, 4, 5], np.int32)
    other.extended_properties["B"] = 3
    other.extended_properties["C"] = 4

    waveform.append(other)

    assert list(waveform.raw_data) == [0, 1, 2, 3, 4, 5]
    assert waveform.extended_properties == {"A": 1, "B": 2, "C": 4}


def test___regular_waveform_and_regular_waveform_with_different_scale_mode___append___appends_waveform_with_scaling_mismatch_warning() -> (
    None
):
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.int32)
    waveform.scale_mode = LinearScaleMode(1.0, 0.0)
    other = AnalogWaveform.from_array_1d([3, 4, 5], np.int32)
    other.scale_mode = LinearScaleMode(2.0, 0.0)

    with pytest.warns(wfmwarn.ScalingMismatchWarning):
        waveform.append(other)

    assert list(waveform.raw_data) == [0, 1, 2, 3, 4, 5]
    assert waveform.scale_mode == LinearScaleMode(1.0, 0.0)


###############################################################################
# append waveforms
###############################################################################
def test___empty_waveform_list___append___no_effect() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.int32)
    other: list[AnalogWaveform[np.int32]] = []

    waveform.append(other)

    assert list(waveform.raw_data) == [0, 1, 2]


def test___int32_waveform_list___append___appends_waveform() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.int32)
    other = [
        AnalogWaveform.from_array_1d([3, 4, 5], np.int32),
        AnalogWaveform.from_array_1d([6], np.int32),
        AnalogWaveform.from_array_1d([7, 8], np.int32),
    ]

    waveform.append(other)

    assert list(waveform.raw_data) == [0, 1, 2, 3, 4, 5, 6, 7, 8]


def test___float64_waveform_tuple___append___appends_waveform() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.float64)
    other = (
        AnalogWaveform.from_array_1d([3, 4, 5], np.float64),
        AnalogWaveform.from_array_1d([6, 7, 8], np.float64),
    )

    waveform.append(other)

    assert list(waveform.raw_data) == [0, 1, 2, 3, 4, 5, 6, 7, 8]


def test___waveform_list_with_mismatched_dtype___append___raises_correct_error_and_does_not_append() -> (
    None
):
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.float64)
    other = [
        AnalogWaveform.from_array_1d([3, 4, 5], np.float64),
        AnalogWaveform.from_array_1d([6, 7, 8], np.int32),
    ]

    with pytest.raises(wfmex.DatatypeMismatchError) as exc:
        waveform.append(other)  # type: ignore[arg-type]

    assert exc.value.args[0].startswith(
        "The data type of the input waveform must match the waveform data type."
    )
    assert list(waveform.raw_data) == [0, 1, 2]


def test___irregular_waveform_and_irregular_waveform_list___append___appends_waveform() -> None:
    start_time = dt.datetime.now(dt.timezone.utc)
    waveform_offsets = [dt.timedelta(0), dt.timedelta(1), dt.timedelta(2)]
    waveform_timestamps = [start_time + offset for offset in waveform_offsets]
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.int32)
    waveform.timing = Timing.create_with_irregular_interval(waveform_timestamps)
    other1_offsets = [dt.timedelta(3), dt.timedelta(4), dt.timedelta(5)]
    other1_timestamps = [start_time + offset for offset in other1_offsets]
    other1 = AnalogWaveform.from_array_1d([3, 4, 5], np.int32)
    other1.timing = Timing.create_with_irregular_interval(other1_timestamps)
    other2_offsets = [dt.timedelta(6), dt.timedelta(7), dt.timedelta(8)]
    other2_timestamps = [start_time + offset for offset in other2_offsets]
    other2 = AnalogWaveform.from_array_1d([6, 7, 8], np.int32)
    other2.timing = Timing.create_with_irregular_interval(other2_timestamps)
    other = [other1, other2]

    waveform.append(other)

    assert list(waveform.raw_data) == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert waveform.timing.sample_interval_mode == SampleIntervalMode.IRREGULAR
    assert (
        waveform.timing._timestamps == waveform_timestamps + other1_timestamps + other2_timestamps
    )


def test___irregular_waveform_and_regular_waveform_list___append___raises_correct_error_and_does_not_append() -> (
    None
):
    start_time = dt.datetime.now(dt.timezone.utc)
    waveform_offsets = [dt.timedelta(0), dt.timedelta(1), dt.timedelta(2)]
    waveform_timestamps = [start_time + offset for offset in waveform_offsets]
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.int32)
    waveform.timing = Timing.create_with_irregular_interval(waveform_timestamps)
    other1_offsets = [dt.timedelta(3), dt.timedelta(4), dt.timedelta(5)]
    other1_timestamps = [start_time + offset for offset in other1_offsets]
    other1 = AnalogWaveform.from_array_1d([3, 4, 5], np.int32)
    other1.timing = Timing.create_with_irregular_interval(other1_timestamps)
    other2 = AnalogWaveform.from_array_1d([6, 7, 8], np.int32)
    other2.timing = Timing.create_with_regular_interval(dt.timedelta(milliseconds=1))
    other = [other1, other2]

    with pytest.raises(wfmex.SampleIntervalModeMismatchError) as exc:
        waveform.append(other)

    assert exc.value.args[0].startswith(
        "The timing of one or more waveforms does not match the timing of the current waveform."
    )
    assert list(waveform.raw_data) == [0, 1, 2]
    assert waveform.timing.sample_interval_mode == SampleIntervalMode.IRREGULAR
    assert waveform.timing._timestamps == waveform_timestamps


def test___regular_waveform_and_irregular_waveform_list___append___raises_correct_error_and_does_not_append() -> (
    None
):
    start_time = dt.datetime.now(dt.timezone.utc)
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.int32)
    waveform.timing = Timing.create_with_regular_interval(dt.timedelta(milliseconds=1))
    other1 = AnalogWaveform.from_array_1d([3, 4, 5], np.int32)
    other1.timing = Timing.create_with_regular_interval(dt.timedelta(milliseconds=1))
    other2_offsets = [dt.timedelta(3), dt.timedelta(4), dt.timedelta(5)]
    other2_timestamps = [start_time + offset for offset in other2_offsets]
    other2 = AnalogWaveform.from_array_1d([3, 4, 5], np.int32)
    other2.timing = Timing.create_with_irregular_interval(other2_timestamps)
    other = [other1, other2]

    with pytest.raises(wfmex.SampleIntervalModeMismatchError) as exc:
        waveform.append(other)

    assert exc.value.args[0].startswith(
        "The timing of one or more waveforms does not match the timing of the current waveform."
    )
    assert list(waveform.raw_data) == [0, 1, 2]
    assert waveform.timing.sample_interval_mode == SampleIntervalMode.REGULAR
    assert waveform.timing.sample_interval == dt.timedelta(milliseconds=1)


###############################################################################
# load data
###############################################################################
def test___empty_ndarray___load_data___clears_data() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.int32)
    array = np.array([], np.int32)

    waveform.load_data(array)

    assert list(waveform.raw_data) == []


def test___int32_ndarray___load_data___overwrites_data() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.int32)
    array = np.array([3, 4, 5], np.int32)

    waveform.load_data(array)

    assert list(waveform.raw_data) == [3, 4, 5]


def test___float64_ndarray___load_data___overwrites_data() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.float64)
    array = np.array([3, 4, 5], np.float64)

    waveform.load_data(array)

    assert list(waveform.raw_data) == [3, 4, 5]


def test___ndarray_with_mismatched_dtype___load_data___raises_correct_error() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.float64)
    array = np.array([3, 4, 5], np.int32)

    with pytest.raises(wfmex.DatatypeMismatchError) as exc:
        waveform.load_data(array)  # type: ignore[arg-type]

    assert exc.value.args[0].startswith(
        "The data type of the input array must match the waveform data type."
    )


def test___ndarray_2d___load_data___raises_value_error() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.float64)
    array = np.array([[3, 4, 5], [6, 7, 8]], np.float64)

    with pytest.raises(ValueError) as exc:
        waveform.load_data(array)

    assert exc.value.args[0].startswith("The input array must be a one-dimensional array.")


def test___smaller_ndarray___load_data___preserves_capacity() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.int32)
    array = np.array([3], np.int32)

    waveform.load_data(array)

    assert list(waveform.raw_data) == [3]
    assert waveform.capacity == 3


def test___larger_ndarray___load_data___grows_capacity() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.int32)
    array = np.array([3, 4, 5, 6], np.int32)

    waveform.load_data(array)

    assert list(waveform.raw_data) == [3, 4, 5, 6]
    assert waveform.capacity == 4


def test___waveform_with_start_index___load_data___clears_start_index() -> None:
    waveform = AnalogWaveform.from_array_1d(
        np.array([0, 1, 2], np.int32), np.int32, copy=False, start_index=1, sample_count=1
    )
    assert waveform._start_index == 1
    array = np.array([3], np.int32)

    waveform.load_data(array)

    assert list(waveform.raw_data) == [3]
    assert waveform._start_index == 0


def test___ndarray_subset___load_data___overwrites_data() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.int32)
    array = np.array([3, 4, 5], np.int32)

    waveform.load_data(array, start_index=1, sample_count=1)

    assert list(waveform.raw_data) == [4]
    assert waveform._start_index == 0
    assert waveform.capacity == 3


def test___smaller_ndarray_no_copy___load_data___takes_ownership_of_array() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.int32)
    array = np.array([3], np.int32)

    waveform.load_data(array, copy=False)

    assert list(waveform.raw_data) == [3]
    assert waveform._data is array


def test___larger_ndarray_no_copy___load_data___takes_ownership_of_array() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.int32)
    array = np.array([3, 4, 5, 6], np.int32)

    waveform.load_data(array, copy=False)

    assert list(waveform.raw_data) == [3, 4, 5, 6]
    assert waveform._data is array


def test___ndarray_subset_no_copy___load_data___takes_ownership_of_array_subset() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.int32)
    array = np.array([3, 4, 5, 6], np.int32)

    waveform.load_data(array, copy=False, start_index=1, sample_count=2)

    assert list(waveform.raw_data) == [4, 5]
    assert waveform._data is array


def test___irregular_waveform_and_int32_ndarray_with_timestamps___load_data___overwrites_data_but_not_timestamps() -> (
    None
):
    start_time = dt.datetime.now(dt.timezone.utc)
    waveform_offsets = [dt.timedelta(0), dt.timedelta(1), dt.timedelta(2)]
    waveform_timestamps = [start_time + offset for offset in waveform_offsets]
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.int32)
    waveform.timing = Timing.create_with_irregular_interval(waveform_timestamps)
    array = np.array([3, 4, 5], np.int32)

    waveform.load_data(array)

    assert list(waveform.raw_data) == [3, 4, 5]
    assert waveform.timing.sample_interval_mode == SampleIntervalMode.IRREGULAR
    assert waveform.timing._timestamps == waveform_timestamps


def test___irregular_waveform_and_int32_ndarray_with_wrong_sample_count___load_data___raises_correct_error_and_does_not_overwrite_data() -> (
    None
):
    start_time = dt.datetime.now(dt.timezone.utc)
    waveform_offsets = [dt.timedelta(0), dt.timedelta(1), dt.timedelta(2)]
    waveform_timestamps = [start_time + offset for offset in waveform_offsets]
    waveform = AnalogWaveform.from_array_1d([0, 1, 2], np.int32)
    waveform.timing = Timing.create_with_irregular_interval(waveform_timestamps)
    array = np.array([3, 4], np.int32)

    with pytest.raises(wfmex.IrregularTimestampCountMismatchError) as exc:
        waveform.load_data(array)

    assert exc.value.args[0].startswith(
        "The input array length must be equal to the number of irregular timestamps."
    )
    assert list(waveform.raw_data) == [0, 1, 2]
    assert waveform.timing.sample_interval_mode == SampleIntervalMode.IRREGULAR
    assert waveform.timing._timestamps == waveform_timestamps


###############################################################################
# magic methods
###############################################################################
@pytest.mark.parametrize(
    "left, right",
    [
        (AnalogWaveform(), AnalogWaveform()),
        (AnalogWaveform(10), AnalogWaveform(10)),
        (AnalogWaveform(10, np.float64), AnalogWaveform(10, np.float64)),
        (AnalogWaveform(10, np.int32), AnalogWaveform(10, np.int32)),
        (
            AnalogWaveform(10, np.int32, start_index=5, capacity=20),
            AnalogWaveform(10, np.int32, start_index=5, capacity=20),
        ),
        (
            AnalogWaveform.from_array_1d([1, 2, 3], np.float64),
            AnalogWaveform.from_array_1d([1, 2, 3], np.float64),
        ),
        (
            AnalogWaveform.from_array_1d([1, 2, 3], np.int32),
            AnalogWaveform.from_array_1d([1, 2, 3], np.int32),
        ),
        (
            AnalogWaveform(
                timing=Timing.create_with_regular_interval(dt.timedelta(milliseconds=1))
            ),
            AnalogWaveform(
                timing=Timing.create_with_regular_interval(dt.timedelta(milliseconds=1))
            ),
        ),
        (
            AnalogWaveform(
                timing=Timing.create_with_regular_interval(ht.timedelta(milliseconds=1))
            ),
            AnalogWaveform(
                timing=Timing.create_with_regular_interval(ht.timedelta(milliseconds=1))
            ),
        ),
        (
            AnalogWaveform(
                extended_properties={"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Volts"}
            ),
            AnalogWaveform(
                extended_properties={"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Volts"}
            ),
        ),
        (
            AnalogWaveform(scale_mode=LinearScaleMode(2.0, 1.0)),
            AnalogWaveform(scale_mode=LinearScaleMode(2.0, 1.0)),
        ),
        # start_index and capacity may differ as long as raw_data and sample_count are the same.
        (
            AnalogWaveform(10, np.int32, start_index=5, capacity=20),
            AnalogWaveform(10, np.int32, start_index=10, capacity=25),
        ),
        (
            AnalogWaveform.from_array_1d(
                [0, 0, 1, 2, 3, 4, 5, 0], np.int32, start_index=2, sample_count=5
            ),
            AnalogWaveform.from_array_1d(
                [0, 1, 2, 3, 4, 5, 0, 0, 0], np.int32, start_index=1, sample_count=5
            ),
        ),
        # Same value, different time type
        (
            AnalogWaveform(
                timing=Timing.create_with_regular_interval(dt.timedelta(milliseconds=1))
            ),
            AnalogWaveform(
                timing=Timing.create_with_regular_interval(ht.timedelta(milliseconds=1))
            ),
        ),
        (
            AnalogWaveform(
                timing=Timing.create_with_regular_interval(ht.timedelta(milliseconds=1))
            ),
            AnalogWaveform(
                timing=Timing.create_with_regular_interval(dt.timedelta(milliseconds=1))
            ),
        ),
    ],
)
def test___same_value___equality___equal(
    left: AnalogWaveform[Any], right: AnalogWaveform[Any]
) -> None:
    assert left == right
    assert not (left != right)


@pytest.mark.parametrize(
    "left, right",
    [
        (AnalogWaveform(), AnalogWaveform(10)),
        (AnalogWaveform(10), AnalogWaveform(11)),
        (AnalogWaveform(10, np.float64), AnalogWaveform(10, np.int32)),
        (
            AnalogWaveform(15, np.int32, start_index=5, capacity=20),
            AnalogWaveform(10, np.int32, start_index=5, capacity=20),
        ),
        (
            AnalogWaveform.from_array_1d([1, 4, 3], np.float64),
            AnalogWaveform.from_array_1d([1, 2, 3], np.float64),
        ),
        (
            AnalogWaveform.from_array_1d([1, 2, 3], np.int32),
            AnalogWaveform.from_array_1d([1, 2, 3], np.float64),
        ),
        (
            AnalogWaveform(
                timing=Timing.create_with_regular_interval(dt.timedelta(milliseconds=1))
            ),
            AnalogWaveform(
                timing=Timing.create_with_regular_interval(dt.timedelta(milliseconds=2))
            ),
        ),
        (
            AnalogWaveform(
                timing=Timing.create_with_regular_interval(ht.timedelta(milliseconds=1))
            ),
            AnalogWaveform(
                timing=Timing.create_with_regular_interval(ht.timedelta(milliseconds=2))
            ),
        ),
        (
            AnalogWaveform(
                extended_properties={"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Volts"}
            ),
            AnalogWaveform(
                extended_properties={"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Amps"}
            ),
        ),
        (
            AnalogWaveform(scale_mode=LinearScaleMode(2.0, 1.0)),
            AnalogWaveform(scale_mode=LinearScaleMode(2.0, 1.1)),
        ),
        (
            AnalogWaveform(scale_mode=NO_SCALING),
            AnalogWaveform(scale_mode=LinearScaleMode(2.0, 1.0)),
        ),
    ],
)
def test___different_value___equality___not_equal(
    left: AnalogWaveform[Any], right: AnalogWaveform[Any]
) -> None:
    assert not (left == right)
    assert left != right


if Version(np.__version__) >= Version("2.0.0") or sys.platform != "win32":
    _NDARRAY_DTYPE_INT32 = ", dtype=int32"
else:
    _NDARRAY_DTYPE_INT32 = ""


@pytest.mark.parametrize(
    "value, expected_repr",
    [
        (AnalogWaveform(), "nitypes.waveform.AnalogWaveform(0)"),
        (
            AnalogWaveform(5),
            "nitypes.waveform.AnalogWaveform(5, raw_data=array([0., 0., 0., 0., 0.]))",
        ),
        (
            AnalogWaveform(5, np.float64),
            "nitypes.waveform.AnalogWaveform(5, raw_data=array([0., 0., 0., 0., 0.]))",
        ),
        (AnalogWaveform(0, np.int32), "nitypes.waveform.AnalogWaveform(0, int32)"),
        (
            AnalogWaveform(5, np.int32),
            f"nitypes.waveform.AnalogWaveform(5, int32, raw_data=array([0, 0, 0, 0, 0]{_NDARRAY_DTYPE_INT32}))",
        ),
        (
            AnalogWaveform(5, np.int32, start_index=5, capacity=20),
            f"nitypes.waveform.AnalogWaveform(5, int32, raw_data=array([0, 0, 0, 0, 0]{_NDARRAY_DTYPE_INT32}))",
        ),
        (
            AnalogWaveform.from_array_1d([1, 2, 3], np.float64),
            "nitypes.waveform.AnalogWaveform(3, raw_data=array([1., 2., 3.]))",
        ),
        (
            AnalogWaveform.from_array_1d([1, 2, 3], np.int32),
            f"nitypes.waveform.AnalogWaveform(3, int32, raw_data=array([1, 2, 3]{_NDARRAY_DTYPE_INT32}))",
        ),
        (
            AnalogWaveform(
                timing=Timing.create_with_regular_interval(dt.timedelta(milliseconds=1))
            ),
            "nitypes.waveform.AnalogWaveform(0, "
            "timing=nitypes.waveform.Timing(nitypes.waveform.SampleIntervalMode.REGULAR, "
            "sample_interval=datetime.timedelta(microseconds=1000)))",
        ),
        (
            AnalogWaveform(
                timing=Timing.create_with_regular_interval(ht.timedelta(milliseconds=1))
            ),
            "nitypes.waveform.AnalogWaveform(0, "
            "timing=nitypes.waveform.Timing(nitypes.waveform.SampleIntervalMode.REGULAR, "
            "sample_interval=hightime.timedelta(microseconds=1000)))",
        ),
        (
            AnalogWaveform(
                extended_properties={"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Volts"}
            ),
            "nitypes.waveform.AnalogWaveform(0, extended_properties={'NI_ChannelName': 'Dev1/ai0', "
            "'NI_UnitDescription': 'Volts'})",
        ),
        (
            AnalogWaveform(scale_mode=LinearScaleMode(2.0, 1.0)),
            "nitypes.waveform.AnalogWaveform(0, scale_mode=nitypes.waveform.LinearScaleMode(2.0, 1.0))",
        ),
        (
            AnalogWaveform.from_array_1d(
                [1, 2, 3],
                np.int32,
                timing=Timing.create_with_regular_interval(dt.timedelta(milliseconds=1)),
            ),
            f"nitypes.waveform.AnalogWaveform(3, int32, raw_data=array([1, 2, 3]{_NDARRAY_DTYPE_INT32}), "
            "timing=nitypes.waveform.Timing(nitypes.waveform.SampleIntervalMode.REGULAR, "
            "sample_interval=datetime.timedelta(microseconds=1000)))",
        ),
        (
            AnalogWaveform.from_array_1d(
                [1, 2, 3],
                np.int32,
                extended_properties={"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Volts"},
            ),
            f"nitypes.waveform.AnalogWaveform(3, int32, raw_data=array([1, 2, 3]{_NDARRAY_DTYPE_INT32}), "
            "extended_properties={'NI_ChannelName': 'Dev1/ai0', 'NI_UnitDescription': 'Volts'})",
        ),
        (
            AnalogWaveform.from_array_1d([1, 2, 3], np.int32, scale_mode=LinearScaleMode(2.0, 1.0)),
            f"nitypes.waveform.AnalogWaveform(3, int32, raw_data=array([1, 2, 3]{_NDARRAY_DTYPE_INT32}), "
            "scale_mode=nitypes.waveform.LinearScaleMode(2.0, 1.0))",
        ),
    ],
)
def test___various_values___repr___looks_ok(value: AnalogWaveform[Any], expected_repr: str) -> None:
    assert repr(value) == expected_repr


_VARIOUS_VALUES = [
    AnalogWaveform(),
    AnalogWaveform(10),
    AnalogWaveform(10, np.float64),
    AnalogWaveform(10, np.int32),
    AnalogWaveform(10, np.int32, start_index=5, capacity=20),
    AnalogWaveform.from_array_1d([1, 2, 3], np.float64),
    AnalogWaveform.from_array_1d([1, 2, 3], np.int32),
    AnalogWaveform(timing=Timing.create_with_regular_interval(dt.timedelta(milliseconds=1))),
    AnalogWaveform(timing=Timing.create_with_regular_interval(ht.timedelta(milliseconds=1))),
    AnalogWaveform(
        extended_properties={"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Volts"}
    ),
    AnalogWaveform(scale_mode=LinearScaleMode(2.0, 1.0)),
    AnalogWaveform(10, np.int32, start_index=5, capacity=20),
    AnalogWaveform.from_array_1d([0, 0, 1, 2, 3, 4, 5, 0], np.int32, start_index=2, sample_count=5),
]


@pytest.mark.parametrize("value", _VARIOUS_VALUES)
def test___various_values___copy___makes_shallow_copy(value: AnalogWaveform[Any]) -> None:
    new_value = copy.copy(value)

    _assert_shallow_copy(new_value, value)


def _assert_shallow_copy(value: AnalogWaveform[Any], other: AnalogWaveform[Any]) -> None:
    assert value == other
    assert value is not other
    # _data may be a view of the original array.
    assert value._data is other._data or value._data.base is other._data
    assert value._extended_properties is other._extended_properties
    assert value._timing is other._timing
    assert value._scale_mode is other._scale_mode


@pytest.mark.parametrize("value", _VARIOUS_VALUES)
def test___various_values___deepcopy___makes_shallow_copy(value: AnalogWaveform[Any]) -> None:
    new_value = copy.deepcopy(value)

    _assert_deep_copy(new_value, value)


def _assert_deep_copy(value: AnalogWaveform[Any], other: AnalogWaveform[Any]) -> None:
    assert value == other
    assert value is not other
    assert value._data is not other._data and value._data.base is not other._data
    assert value._extended_properties is not other._extended_properties
    if other._timing is not Timing.empty:
        assert value._timing is not other._timing
    if other._scale_mode is not NO_SCALING:
        assert value._scale_mode is not other._scale_mode


@pytest.mark.parametrize("value", _VARIOUS_VALUES)
def test___various_values___pickle_unpickle___makes_deep_copy(
    value: AnalogWaveform[Any],
) -> None:
    new_value = pickle.loads(pickle.dumps(value))

    _assert_deep_copy(new_value, value)


def test___waveform___pickle___references_public_modules() -> None:
    value = AnalogWaveform(
        raw_data=np.array([1, 2, 3], np.float64),
        extended_properties={"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Volts"},
        timing=Timing.create_with_regular_interval(dt.timedelta(milliseconds=1)),
        scale_mode=LinearScaleMode(2.0, 1.0),
    )

    value_bytes = pickle.dumps(value)

    assert b"nitypes.waveform" in value_bytes
    assert b"nitypes.waveform._analog" not in value_bytes
    assert b"nitypes.waveform._extended_properties" not in value_bytes
    assert b"nitypes.waveform._numeric" not in value_bytes
    assert b"nitypes.waveform._timing" not in value_bytes
    assert b"nitypes.waveform._scaling" not in value_bytes
