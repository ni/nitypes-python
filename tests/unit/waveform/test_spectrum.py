from __future__ import annotations

import array
import copy
import itertools
import pickle
import sys
import weakref
from typing import Any, SupportsIndex

import numpy as np
import numpy.typing as npt
import pytest
from packaging.version import Version
from typing_extensions import assert_type

import nitypes.waveform.errors as wfmex
from nitypes.waveform import Spectrum


###############################################################################
# create
###############################################################################
def test___no_args___create___creates_empty_spectrum_with_default_dtype() -> None:
    spectrum = Spectrum()

    assert spectrum.sample_count == spectrum.capacity == len(spectrum.data) == 0
    assert spectrum.dtype == np.float64
    assert_type(spectrum, Spectrum[np.float64])


def test___sample_count___create___creates_spectrum_with_sample_count_and_default_dtype() -> None:
    spectrum = Spectrum(10)

    assert spectrum.sample_count == spectrum.capacity == len(spectrum.data) == 10
    assert spectrum.dtype == np.float64
    assert_type(spectrum, Spectrum[np.float64])


def test___sample_count_and_dtype___create___creates_spectrum_with_sample_count_and_dtype() -> None:
    spectrum = Spectrum(10, np.int32)

    assert spectrum.sample_count == spectrum.capacity == len(spectrum.data) == 10
    assert spectrum.dtype == np.int32
    assert_type(spectrum, Spectrum[np.int32])


def test___sample_count_and_dtype_str___create___creates_spectrum_with_sample_count_and_dtype() -> (
    None
):
    spectrum = Spectrum(10, "i4")

    assert spectrum.sample_count == spectrum.capacity == len(spectrum.data) == 10
    assert spectrum.dtype == np.int32
    assert_type(spectrum, Spectrum[Any])  # dtype not inferred from string


def test___sample_count_and_dtype_any___create___creates_spectrum_with_sample_count_and_dtype() -> (
    None
):
    dtype: np.dtype[Any] = np.dtype(np.int32)
    spectrum = Spectrum(10, dtype)

    assert spectrum.sample_count == spectrum.capacity == len(spectrum.data) == 10
    assert spectrum.dtype == np.int32
    assert_type(spectrum, Spectrum[Any])  # dtype not inferred from np.dtype[Any]


def test___sample_count_dtype_and_capacity___create___creates_spectrum_with_sample_count_dtype_and_capacity() -> (
    None
):
    spectrum = Spectrum(10, np.int32, capacity=20)

    assert spectrum.sample_count == len(spectrum.data) == 10
    assert spectrum.capacity == 20
    assert spectrum.dtype == np.int32
    assert_type(spectrum, Spectrum[np.int32])


@pytest.mark.parametrize("dtype", [np.complex128, np.str_, np.void, "i2, i2"])
def test___sample_count_and_unsupported_dtype___create___raises_type_error(
    dtype: npt.DTypeLike,
) -> None:
    with pytest.raises(TypeError) as exc:
        _ = Spectrum(10, dtype)

    assert exc.value.args[0].startswith("The requested data type is not supported.")


def test___dtype_str_with_unsupported_traw_hint___create___mypy_type_var_warning() -> None:
    spectrum1: Spectrum[np.complex128] = Spectrum(dtype="int32")  # type: ignore[type-var]
    spectrum2: Spectrum[np.str_] = Spectrum(dtype="int32")  # type: ignore[type-var]
    spectrum3: Spectrum[np.void] = Spectrum(dtype="int32")  # type: ignore[type-var]
    _ = spectrum1, spectrum2, spectrum3


def test___dtype_str_with_traw_hint___create___narrows_traw() -> None:
    spectrum: Spectrum[np.int32] = Spectrum(dtype="int32")

    assert_type(spectrum, Spectrum[np.int32])


###############################################################################
# from_array_1d
###############################################################################
def test___float64_ndarray___from_array_1d___creates_spectrum_with_float64_dtype() -> None:
    data = np.array([1.1, 2.2, 3.3, 4.4, 5.5], np.float64)

    spectrum = Spectrum.from_array_1d(data)

    assert spectrum.data.tolist() == data.tolist()
    assert spectrum.dtype == np.float64
    assert_type(spectrum, Spectrum[np.float64])


def test___int32_ndarray___from_array_1d___creates_spectrum_with_int32_dtype() -> None:
    data = np.array([1, 2, 3, 4, 5], np.int32)

    spectrum = Spectrum.from_array_1d(data)

    assert spectrum.data.tolist() == data.tolist()
    assert spectrum.dtype == np.int32
    assert_type(spectrum, Spectrum[np.int32])


def test___int32_array_with_dtype___from_array_1d___creates_spectrum_with_specified_dtype() -> None:
    data = array.array("i", [1, 2, 3, 4, 5])

    spectrum = Spectrum.from_array_1d(data, np.int32)

    assert spectrum.data.tolist() == data.tolist()
    assert spectrum.dtype == np.int32
    assert_type(spectrum, Spectrum[np.int32])


def test___int16_ndarray_with_mismatched_dtype___from_array_1d___creates_spectrum_with_specified_dtype() -> (
    None
):
    data = np.array([1, 2, 3, 4, 5], np.int16)

    spectrum = Spectrum.from_array_1d(data, np.int32)

    assert spectrum.data.tolist() == data.tolist()
    assert spectrum.dtype == np.int32
    assert_type(spectrum, Spectrum[np.int32])


def test___int_list_with_dtype___from_array_1d___creates_spectrum_with_specified_dtype() -> None:
    data = [1, 2, 3, 4, 5]

    spectrum = Spectrum.from_array_1d(data, np.int32)

    assert spectrum.data.tolist() == data
    assert spectrum.dtype == np.int32
    assert_type(spectrum, Spectrum[np.int32])


def test___int_list_with_dtype_str___from_array_1d___creates_spectrum_with_specified_dtype() -> (
    None
):
    data = [1, 2, 3, 4, 5]

    spectrum = Spectrum.from_array_1d(data, "int32")

    assert spectrum.data.tolist() == data
    assert spectrum.dtype == np.int32
    assert_type(spectrum, Spectrum[Any])  # dtype not inferred from string


def test___int32_ndarray_2d___from_array_1d___raises_value_error() -> None:
    data = np.array([[1, 2, 3], [4, 5, 6]], np.int32)

    with pytest.raises(ValueError) as exc:
        _ = Spectrum.from_array_1d(data)

    assert exc.value.args[0].startswith(
        "The input array must be a one-dimensional array or sequence."
    )


def test___int_list_without_dtype___from_array_1d___raises_value_error() -> None:
    data = [1, 2, 3, 4, 5]

    with pytest.raises(ValueError) as exc:
        _ = Spectrum.from_array_1d(data)

    assert exc.value.args[0].startswith(
        "You must specify a dtype when the input array is a sequence."
    )


def test___bytes___from_array_1d___raises_value_error() -> None:
    data = b"\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00"

    with pytest.raises(ValueError) as exc:
        _ = Spectrum.from_array_1d(data, np.int32)

    assert exc.value.args[0].startswith("invalid literal for int() with base 10:")


def test___iterable___from_array_1d___raises_type_error() -> None:
    data = itertools.repeat(3)

    with pytest.raises(TypeError) as exc:
        _ = Spectrum.from_array_1d(data, np.int32)  # type: ignore[call-overload]

    assert exc.value.args[0].startswith(
        "The input array must be a one-dimensional array or sequence."
    )


def test___ndarray_with_unsupported_dtype___from_array_1d___raises_type_error() -> None:
    data = np.zeros(3, np.str_)

    with pytest.raises(TypeError) as exc:
        _ = Spectrum.from_array_1d(data)

    assert exc.value.args[0].startswith("The requested data type is not supported.")


def test___copy___from_array_1d___creates_spectrum_linked_to_different_buffer() -> None:
    data = np.array([1, 2, 3, 4, 5], np.int32)

    spectrum = Spectrum.from_array_1d(data, copy=True)

    assert spectrum._data is not data
    assert spectrum.data.tolist() == data.tolist()
    data[:] = [5, 4, 3, 2, 1]
    assert spectrum.data.tolist() != data.tolist()


def test___int32_ndarray_no_copy___from_array_1d___creates_spectrum_linked_to_same_buffer() -> None:
    data = np.array([1, 2, 3, 4, 5], np.int32)

    spectrum = Spectrum.from_array_1d(data, copy=False)

    assert spectrum._data is data
    assert spectrum.data.tolist() == data.tolist()
    data[:] = [5, 4, 3, 2, 1]
    assert spectrum.data.tolist() == data.tolist()


def test___int32_array_no_copy___from_array_1d___creates_spectrum_linked_to_same_buffer() -> None:
    data = array.array("i", [1, 2, 3, 4, 5])

    spectrum = Spectrum.from_array_1d(data, dtype=np.int32, copy=False)

    assert spectrum.data.tolist() == data.tolist()
    data[:] = array.array("i", [5, 4, 3, 2, 1])
    assert spectrum.data.tolist() == data.tolist()


def test___int_list_no_copy___from_array_1d___raises_value_error() -> None:
    data = [1, 2, 3, 4, 5]

    with pytest.raises(ValueError) as exc:
        _ = Spectrum.from_array_1d(data, np.int32, copy=False)

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
def test___array_subset___from_array_1d___creates_spectrum_with_array_subset(
    start_index: SupportsIndex, sample_count: SupportsIndex | None, expected_data: list[int]
) -> None:
    data = np.array([1, 2, 3, 4, 5], np.int32)

    spectrum = Spectrum.from_array_1d(data, start_index=start_index, sample_count=sample_count)

    assert spectrum.data.tolist() == expected_data


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
        _ = Spectrum.from_array_1d(data, start_index=start_index, sample_count=sample_count)

    assert exc.value.args[0].startswith(expected_message)


###############################################################################
# from_array_2d
###############################################################################
def test___float64_ndarray___from_array_2d___creates_spectrum_with_float64_dtype() -> None:
    data = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], np.float64)

    spectrums = Spectrum.from_array_2d(data)

    assert len(spectrums) == 2
    for i in range(len(spectrums)):
        assert spectrums[i].data.tolist() == data[i].tolist()
        assert spectrums[i].dtype == np.float64
        assert_type(spectrums[i], Spectrum[np.float64])


def test___int32_ndarray___from_array_2d___creates_spectrum_with_int32_dtype() -> None:
    data = np.array([[1, 2, 3], [4, 5, 6]], np.int32)

    spectrums = Spectrum.from_array_2d(data)

    assert len(spectrums) == 2
    for i in range(len(spectrums)):
        assert spectrums[i].data.tolist() == data[i].tolist()
        assert spectrums[i].dtype == np.int32
        assert_type(spectrums[i], Spectrum[np.int32])


def test___int16_ndarray_with_mismatched_dtype___from_array_2d___creates_spectrum_with_specified_dtype() -> (
    None
):
    data = np.array([[1, 2, 3], [4, 5, 6]], np.int16)

    spectrums = Spectrum.from_array_2d(data, np.int32)

    assert len(spectrums) == 2
    for i in range(len(spectrums)):
        assert spectrums[i].data.tolist() == data[i].tolist()
        assert spectrums[i].dtype == np.int32
        assert_type(spectrums[i], Spectrum[np.int32])


def test___int32_array_list_with_dtype___from_array_2d___creates_spectrum_with_specified_dtype() -> (
    None
):
    data = [array.array("i", [1, 2, 3]), array.array("i", [4, 5, 6])]

    spectrums = Spectrum.from_array_2d(data, np.int32)

    assert len(spectrums) == 2
    for i in range(len(spectrums)):
        assert spectrums[i].data.tolist() == data[i].tolist()
        assert spectrums[i].dtype == np.int32
        assert_type(spectrums[i], Spectrum[np.int32])


def test___int_list_list_with_dtype___from_array_2d___creates_spectrum_with_specified_dtype() -> (
    None
):
    data = [[1, 2, 3], [4, 5, 6]]

    spectrums = Spectrum.from_array_2d(data, np.int32)

    assert len(spectrums) == 2
    for i in range(len(spectrums)):
        assert spectrums[i].data.tolist() == data[i]
        assert spectrums[i].dtype == np.int32
        assert_type(spectrums[i], Spectrum[np.int32])


def test___int_list_list_with_dtype_str___from_array_2d___creates_spectrum_with_specified_dtype() -> (
    None
):
    data = [[1, 2, 3], [4, 5, 6]]

    spectrums = Spectrum.from_array_2d(data, "int32")

    assert len(spectrums) == 2
    for i in range(len(spectrums)):
        assert spectrums[i].data.tolist() == data[i]
        assert spectrums[i].dtype == np.int32
        assert_type(spectrums[i], Spectrum[Any])  # dtype not inferred from string


def test___int32_ndarray_1d___from_array_2d___raises_value_error() -> None:
    data = np.array([1, 2, 3, 4, 5], np.int32)

    with pytest.raises(ValueError) as exc:
        _ = Spectrum.from_array_2d(data)

    assert exc.value.args[0].startswith(
        "The input array must be a two-dimensional array or nested sequence."
    )


def test___int_list_list_without_dtype___from_array_2d___raises_value_error() -> None:
    data = [[1, 2, 3], [4, 5, 6]]

    with pytest.raises(ValueError) as exc:
        _ = Spectrum.from_array_2d(data)

    assert exc.value.args[0].startswith(
        "You must specify a dtype when the input array is a sequence."
    )


def test___bytes_list___from_array_2d___raises_value_error() -> None:
    data = [
        b"\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00",
        b"\x04\x00\x00\x00\x05\x00\x00\x00\x06\x00\x00\x00",
    ]

    with pytest.raises(ValueError) as exc:
        _ = Spectrum.from_array_2d(data, np.int32)

    assert exc.value.args[0].startswith("invalid literal for int() with base 10:")


def test___list_iterable___from_array_2d___raises_type_error() -> None:
    data = itertools.repeat([3])

    with pytest.raises(TypeError) as exc:
        _ = Spectrum.from_array_2d(data, np.int32)  # type: ignore[call-overload]

    assert exc.value.args[0].startswith(
        "The input array must be a two-dimensional array or nested sequence."
    )


def test___iterable_list___from_array_2d___raises_type_error() -> None:
    data = [itertools.repeat(3), itertools.repeat(4)]

    with pytest.raises(TypeError) as exc:
        _ = Spectrum.from_array_2d(data, np.int32)  # type: ignore[arg-type]

    assert exc.value.args[0].startswith("int() argument must be")


def test___ndarray_with_unsupported_dtype___from_array_2d___raises_type_error() -> None:
    data = np.zeros((2, 3), np.str_)

    with pytest.raises(TypeError) as exc:
        _ = Spectrum.from_array_2d(data)

    assert exc.value.args[0].startswith("The requested data type is not supported.")


def test___copy___from_array_2d___creates_spectrum_linked_to_different_buffer() -> None:
    data = np.array([[1, 2, 3], [4, 5, 6]], np.int32)

    spectrums = Spectrum.from_array_2d(data, copy=True)

    assert len(spectrums) == 2
    for i in range(len(spectrums)):
        assert spectrums[i].data.tolist() == data[i].tolist()
    data[0][:] = [3, 2, 1]
    data[1][:] = [6, 5, 4]
    for i in range(len(spectrums)):
        assert spectrums[i].data.tolist() != data[i].tolist()


def test___int32_ndarray_no_copy___from_array_2d___creates_spectrum_linked_to_same_buffer() -> None:
    data = np.array([[1, 2, 3], [4, 5, 6]], np.int32)

    spectrums = Spectrum.from_array_2d(data, copy=False)

    assert len(spectrums) == 2
    for i in range(len(spectrums)):
        assert spectrums[i].data.tolist() == data[i].tolist()
    data[0][:] = [3, 2, 1]
    data[1][:] = [6, 5, 4]
    for i in range(len(spectrums)):
        assert spectrums[i].data.tolist() == data[i].tolist()


def test___int32_array_list_no_copy___from_array_2d___creates_spectrum_linked_to_same_buffer() -> (
    None
):
    data = [array.array("i", [1, 2, 3]), array.array("i", [4, 5, 6])]

    spectrums = Spectrum.from_array_2d(data, dtype=np.int32, copy=False)

    assert len(spectrums) == 2
    for i in range(len(spectrums)):
        assert spectrums[i].data.tolist() == data[i].tolist()
    data[0][:] = array.array("i", [3, 2, 1])
    data[1][:] = array.array("i", [6, 5, 4])
    for i in range(len(spectrums)):
        assert spectrums[i].data.tolist() == data[i].tolist()


def test___int_list_list_no_copy___from_array_2d___raises_value_error() -> None:
    data = [[1, 2, 3], [4, 5, 6]]

    with pytest.raises(ValueError) as exc:
        _ = Spectrum.from_array_2d(data, np.int32, copy=False)

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
def test___array_subset___from_array_2d___creates_spectrum_with_array_subset(
    start_index: SupportsIndex, sample_count: SupportsIndex | None, expected_data: list[list[int]]
) -> None:
    data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], np.int32)

    spectrums = Spectrum.from_array_2d(data, start_index=start_index, sample_count=sample_count)

    assert len(spectrums) == 2
    for i in range(len(spectrums)):
        assert spectrums[i].data.tolist() == expected_data[i]


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
        _ = Spectrum.from_array_2d(data, start_index=start_index, sample_count=sample_count)

    assert exc.value.args[0].startswith(expected_message)


###############################################################################
# data
###############################################################################
def test___int32_spectrum___data___returns_int32_data() -> None:
    spectrum = Spectrum.from_array_1d([0, 1, 2, 3], np.int32)

    data = spectrum.data

    assert_type(data, npt.NDArray[np.int32])
    assert isinstance(data, np.ndarray) and data.dtype == np.int32
    assert list(data) == [0, 1, 2, 3]


###############################################################################
# get_data
###############################################################################
def test___int32_spectrum___get_data___returns_data() -> None:
    spectrum = Spectrum.from_array_1d([0, 1, 2, 3], np.int32)

    scaled_data = spectrum.get_data()

    assert_type(scaled_data, npt.NDArray[np.int32])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.int32
    assert list(scaled_data) == [0, 1, 2, 3]


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
    spectrum = Spectrum.from_array_1d([0, 1, 2, 3], np.int32)

    scaled_data = spectrum.get_data(start_index=start_index, sample_count=sample_count)

    assert_type(scaled_data, npt.NDArray[np.int32])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.int32
    assert list(scaled_data) == expected_data


@pytest.mark.parametrize(
    "start_index, sample_count, expected_message",
    [
        (
            5,
            None,
            "The start index must be less than or equal to the number of samples in the spectrum.",
        ),
        (
            0,
            5,
            "The sum of the start index and sample count must be less than or equal to the number of samples in the spectrum.",
        ),
        (
            4,
            1,
            "The sum of the start index and sample count must be less than or equal to the number of samples in the spectrum.",
        ),
    ],
)
def test___invalid_array_subset___get_data___returns_array_subset(
    start_index: int, sample_count: int, expected_message: str
) -> None:
    spectrum = Spectrum.from_array_1d([0, 1, 2, 3], np.int32)

    with pytest.raises(
        (wfmex.StartIndexTooLargeError, wfmex.StartIndexOrSampleCountTooLargeError)
    ) as exc:
        _ = spectrum.get_data(start_index=start_index, sample_count=sample_count)

    assert exc.value.args[0].startswith(expected_message)


###############################################################################
# capacity
###############################################################################
@pytest.mark.parametrize(
    "capacity, expected_data",
    [(3, [1, 2, 3]), (4, [1, 2, 3, 0]), (10, [1, 2, 3, 0, 0, 0, 0, 0, 0, 0])],
)
def test___spectrum___set_capacity___resizes_array_and_pads_with_zeros(
    capacity: int, expected_data: list[int]
) -> None:
    data = [1, 2, 3]
    spectrum = Spectrum.from_array_1d(data, np.int32)

    spectrum.capacity = capacity

    assert spectrum.capacity == capacity
    assert spectrum.data.tolist() == data
    assert spectrum._data.tolist() == expected_data


@pytest.mark.parametrize(
    "capacity, expected_message, exception_type",
    [
        (-2, "The capacity must be a non-negative integer.", ValueError),
        (-1, "The capacity must be a non-negative integer.", ValueError),
        (
            0,
            "The capacity must be equal to or greater than the number of samples in the spectrum.",
            wfmex.CapacityTooSmallError,
        ),
        (
            2,
            "The capacity must be equal to or greater than the number of samples in the spectrum.",
            wfmex.CapacityTooSmallError,
        ),
    ],
)
def test___invalid_capacity___set_capacity___raises_correct_error(
    capacity: int, expected_message: str, exception_type: type[Exception]
) -> None:
    data = [1, 2, 3]
    spectrum = Spectrum.from_array_1d(data, np.int32)

    with pytest.raises(exception_type) as exc:
        spectrum.capacity = capacity

    assert exc.value.args[0].startswith(expected_message)


def test___referenced_array___set_capacity___reference_sees_size_change() -> None:
    data = np.array([1, 2, 3], np.int32)
    spectrum = Spectrum.from_array_1d(data, np.int32, copy=False)

    spectrum.capacity = 10

    assert len(data) == 10
    assert spectrum.capacity == 10
    assert data.tolist() == [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]
    assert spectrum.data.tolist() == [1, 2, 3]
    assert spectrum._data.tolist() == [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]


def test___array_with_external_buffer___set_capacity___raises_value_error() -> None:
    data = array.array("i", [1, 2, 3])
    spectrum = Spectrum.from_array_1d(data, np.int32, copy=False)

    with pytest.raises(ValueError) as exc:
        spectrum.capacity = 10

    assert exc.value.args[0].startswith("cannot resize this array: it does not own its data")


###############################################################################
# extended properties
###############################################################################
def test___spectrum___set_channel_name___sets_extended_property() -> None:
    spectrum = Spectrum()

    spectrum.channel_name = "Dev1/ai0"

    assert spectrum.channel_name == "Dev1/ai0"
    assert spectrum.extended_properties["NI_ChannelName"] == "Dev1/ai0"


def test___invalid_type___set_channel_name___raises_type_error() -> None:
    spectrum = Spectrum()

    with pytest.raises(TypeError) as exc:
        spectrum.channel_name = 1  # type: ignore[assignment]

    assert exc.value.args[0].startswith("The channel name must be a str.")


def test___spectrum___set_units___sets_extended_property() -> None:
    spectrum = Spectrum()

    spectrum.units = "Volts"

    assert spectrum.units == "Volts"
    assert spectrum.extended_properties["NI_UnitDescription"] == "Volts"


def test___invalid_type___set_units___raises_type_error() -> None:
    spectrum = Spectrum()

    with pytest.raises(TypeError) as exc:
        spectrum.units = None  # type: ignore[assignment]

    assert exc.value.args[0].startswith("The units must be a str.")


def test___spectrum___set_undefined_property___raises_attribute_error() -> None:
    spectrum = Spectrum()

    with pytest.raises(AttributeError):
        spectrum.undefined_property = "Whatever"  # type: ignore[attr-defined]


def test___spectrum___take_weak_ref___references_spectrum() -> None:
    spectrum = Spectrum()

    spectrum_ref = weakref.ref(spectrum)

    assert spectrum_ref() is spectrum


###############################################################################
# frequency range
###############################################################################
def test___spectrum___has_default_frequency_range() -> None:
    spectrum = Spectrum()

    assert spectrum.start_frequency == 0.0
    assert spectrum.frequency_increment == 0.0


def test___spectrum_with_frequencies___has_specified_frequency_range() -> None:
    spectrum = Spectrum(start_frequency=123.456, frequency_increment=0.1)

    assert spectrum.start_frequency == 123.456
    assert spectrum.frequency_increment == 0.1


def test___spectrum_with_frequencies___set_frequencies___has_set_frequency_range() -> None:
    spectrum = Spectrum(start_frequency=123.456, frequency_increment=0.1)

    spectrum.start_frequency = 234.567
    spectrum.frequency_increment = 0.2

    assert spectrum.start_frequency == 234.567
    assert spectrum.frequency_increment == 0.2


###############################################################################
# append array
###############################################################################
def test___empty_ndarray___append___no_effect() -> None:
    spectrum = Spectrum.from_array_1d([0, 1, 2], np.int32)
    array = np.array([], np.int32)

    spectrum.append(array)

    assert list(spectrum.data) == [0, 1, 2]


def test___int32_ndarray___append___appends_array() -> None:
    spectrum = Spectrum.from_array_1d([0, 1, 2], np.int32)
    array = np.array([3, 4, 5], np.int32)

    spectrum.append(array)

    assert list(spectrum.data) == [0, 1, 2, 3, 4, 5]


def test___float64_ndarray___append___appends_array() -> None:
    spectrum = Spectrum.from_array_1d([0, 1, 2], np.float64)
    array = np.array([3, 4, 5], np.float64)

    spectrum.append(array)

    assert list(spectrum.data) == [0, 1, 2, 3, 4, 5]


def test___ndarray_with_mismatched_dtype___append___raises_correct_error() -> None:
    spectrum = Spectrum.from_array_1d([0, 1, 2], np.float64)
    array = np.array([3, 4, 5], np.int32)

    with pytest.raises(wfmex.DatatypeMismatchError) as exc:
        spectrum.append(array)  # type: ignore[arg-type]

    assert exc.value.args[0].startswith(
        "The data type of the input array must match the spectrum data type."
    )


def test___ndarray_2d___append___raises_value_error() -> None:
    spectrum = Spectrum.from_array_1d([0, 1, 2], np.float64)
    array = np.array([[3, 4, 5], [6, 7, 8]], np.float64)

    with pytest.raises(ValueError) as exc:
        spectrum.append(array)

    assert exc.value.args[0].startswith("The input array must be a one-dimensional array.")


###############################################################################
# append spectrum
###############################################################################
def test___empty_spectrum___append___no_effect() -> None:
    spectrum = Spectrum.from_array_1d([0, 1, 2], np.int32)
    other = Spectrum(dtype=np.int32)

    spectrum.append(other)

    assert list(spectrum.data) == [0, 1, 2]


def test___int32_spectrum___append___appends_spectrum() -> None:
    spectrum = Spectrum.from_array_1d([0, 1, 2], np.int32)
    other = Spectrum.from_array_1d([3, 4, 5], np.int32)

    spectrum.append(other)

    assert list(spectrum.data) == [0, 1, 2, 3, 4, 5]


def test___float64_spectrum___append___appends_spectrum() -> None:
    spectrum = Spectrum.from_array_1d([0, 1, 2], np.float64)
    other = Spectrum.from_array_1d([3, 4, 5], np.float64)

    spectrum.append(other)

    assert list(spectrum.data) == [0, 1, 2, 3, 4, 5]


def test___spectrum_with_mismatched_dtype___append___raises_correct_error() -> None:
    spectrum = Spectrum.from_array_1d([0, 1, 2], np.float64)
    other = Spectrum.from_array_1d([3, 4, 5], np.int32)

    with pytest.raises(wfmex.DatatypeMismatchError) as exc:
        spectrum.append(other)  # type: ignore[arg-type]

    assert exc.value.args[0].startswith(
        "The data type of the input spectrum must match the spectrum data type."
    )


###############################################################################
# append spectrums
###############################################################################
def test___empty_spectrum_list___append___no_effect() -> None:
    spectrum = Spectrum.from_array_1d([0, 1, 2], np.int32)
    other: list[Spectrum[np.int32]] = []

    spectrum.append(other)

    assert list(spectrum.data) == [0, 1, 2]


def test___int32_spectrum_list___append___appends_spectrum() -> None:
    spectrum = Spectrum.from_array_1d([0, 1, 2], np.int32)
    other = [
        Spectrum.from_array_1d([3, 4, 5], np.int32),
        Spectrum.from_array_1d([6], np.int32),
        Spectrum.from_array_1d([7, 8], np.int32),
    ]

    spectrum.append(other)

    assert list(spectrum.data) == [0, 1, 2, 3, 4, 5, 6, 7, 8]


def test___float64_spectrum_tuple___append___appends_spectrum() -> None:
    spectrum = Spectrum.from_array_1d([0, 1, 2], np.float64)
    other = (
        Spectrum.from_array_1d([3, 4, 5], np.float64),
        Spectrum.from_array_1d([6, 7, 8], np.float64),
    )

    spectrum.append(other)

    assert list(spectrum.data) == [0, 1, 2, 3, 4, 5, 6, 7, 8]


def test___spectrum_list_with_mismatched_dtype___append___raises_correct_error_and_does_not_append() -> (
    None
):
    spectrum = Spectrum.from_array_1d([0, 1, 2], np.float64)
    other = [
        Spectrum.from_array_1d([3, 4, 5], np.float64),
        Spectrum.from_array_1d([6, 7, 8], np.int32),
    ]

    with pytest.raises(wfmex.DatatypeMismatchError) as exc:
        spectrum.append(other)  # type: ignore[arg-type]

    assert exc.value.args[0].startswith(
        "The data type of the input spectrum must match the spectrum data type."
    )
    assert list(spectrum.data) == [0, 1, 2]


###############################################################################
# load data
###############################################################################
def test___empty_ndarray___load_data___clears_data() -> None:
    spectrum = Spectrum.from_array_1d([0, 1, 2], np.int32)
    array = np.array([], np.int32)

    spectrum.load_data(array)

    assert list(spectrum.data) == []


def test___int32_ndarray___load_data___overwrites_data() -> None:
    spectrum = Spectrum.from_array_1d([0, 1, 2], np.int32)
    array = np.array([3, 4, 5], np.int32)

    spectrum.load_data(array)

    assert list(spectrum.data) == [3, 4, 5]


def test___float64_ndarray___load_data___overwrites_data() -> None:
    spectrum = Spectrum.from_array_1d([0, 1, 2], np.float64)
    array = np.array([3, 4, 5], np.float64)

    spectrum.load_data(array)

    assert list(spectrum.data) == [3, 4, 5]


def test___ndarray_with_mismatched_dtype___load_data___raises_correct_error() -> None:
    spectrum = Spectrum.from_array_1d([0, 1, 2], np.float64)
    array = np.array([3, 4, 5], np.int32)

    with pytest.raises(wfmex.DatatypeMismatchError) as exc:
        spectrum.load_data(array)  # type: ignore[arg-type]

    assert exc.value.args[0].startswith(
        "The data type of the input array must match the spectrum data type."
    )


def test___ndarray_2d___load_data___raises_value_error() -> None:
    spectrum = Spectrum.from_array_1d([0, 1, 2], np.float64)
    array = np.array([[3, 4, 5], [6, 7, 8]], np.float64)

    with pytest.raises(ValueError) as exc:
        spectrum.load_data(array)

    assert exc.value.args[0].startswith("The input array must be a one-dimensional array.")


def test___smaller_ndarray___load_data___preserves_capacity() -> None:
    spectrum = Spectrum.from_array_1d([0, 1, 2], np.int32)
    array = np.array([3], np.int32)

    spectrum.load_data(array)

    assert list(spectrum.data) == [3]
    assert spectrum.capacity == 3


def test___larger_ndarray___load_data___grows_capacity() -> None:
    spectrum = Spectrum.from_array_1d([0, 1, 2], np.int32)
    array = np.array([3, 4, 5, 6], np.int32)

    spectrum.load_data(array)

    assert list(spectrum.data) == [3, 4, 5, 6]
    assert spectrum.capacity == 4


def test___spectrum_with_start_index___load_data___clears_start_index() -> None:
    spectrum = Spectrum.from_array_1d(
        np.array([0, 1, 2], np.int32), np.int32, copy=False, start_index=1, sample_count=1
    )
    assert spectrum.start_index == 1
    array = np.array([3], np.int32)

    spectrum.load_data(array)

    assert list(spectrum.data) == [3]
    assert spectrum.start_index == 0


def test___ndarray_subset___load_data___overwrites_data() -> None:
    spectrum = Spectrum.from_array_1d([0, 1, 2], np.int32)
    array = np.array([3, 4, 5], np.int32)

    spectrum.load_data(array, start_index=1, sample_count=1)

    assert list(spectrum.data) == [4]
    assert spectrum.start_index == 0
    assert spectrum.capacity == 3


def test___smaller_ndarray_no_copy___load_data___takes_ownership_of_array() -> None:
    spectrum = Spectrum.from_array_1d([0, 1, 2], np.int32)
    array = np.array([3], np.int32)

    spectrum.load_data(array, copy=False)

    assert list(spectrum.data) == [3]
    assert spectrum._data is array


def test___larger_ndarray_no_copy___load_data___takes_ownership_of_array() -> None:
    spectrum = Spectrum.from_array_1d([0, 1, 2], np.int32)
    array = np.array([3, 4, 5, 6], np.int32)

    spectrum.load_data(array, copy=False)

    assert list(spectrum.data) == [3, 4, 5, 6]
    assert spectrum._data is array


def test___ndarray_subset_no_copy___load_data___takes_ownership_of_array_subset() -> None:
    spectrum = Spectrum.from_array_1d([0, 1, 2], np.int32)
    array = np.array([3, 4, 5, 6], np.int32)

    spectrum.load_data(array, copy=False, start_index=1, sample_count=2)

    assert list(spectrum.data) == [4, 5]
    assert spectrum._data is array


###############################################################################
# magic methods
###############################################################################
@pytest.mark.parametrize(
    "left, right",
    [
        (Spectrum(), Spectrum()),
        (Spectrum(10), Spectrum(10)),
        (Spectrum(10, np.float64), Spectrum(10, np.float64)),
        (Spectrum(10, np.int32), Spectrum(10, np.int32)),
        (
            Spectrum(10, np.int32, start_index=5, capacity=20),
            Spectrum(10, np.int32, start_index=5, capacity=20),
        ),
        (
            Spectrum.from_array_1d([1, 2, 3], np.float64),
            Spectrum.from_array_1d([1, 2, 3], np.float64),
        ),
        (
            Spectrum.from_array_1d([1, 2, 3], np.int32),
            Spectrum.from_array_1d([1, 2, 3], np.int32),
        ),
        (
            Spectrum(
                extended_properties={"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Volts"}
            ),
            Spectrum(
                extended_properties={"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Volts"}
            ),
        ),
        # start_index and capacity may differ as long as data and sample_count are the same.
        (
            Spectrum(10, np.int32, start_index=5, capacity=20),
            Spectrum(10, np.int32, start_index=10, capacity=25),
        ),
        (
            Spectrum.from_array_1d(
                [0, 0, 1, 2, 3, 4, 5, 0], np.int32, start_index=2, sample_count=5
            ),
            Spectrum.from_array_1d(
                [0, 1, 2, 3, 4, 5, 0, 0, 0], np.int32, start_index=1, sample_count=5
            ),
        ),
    ],
)
def test___same_value___equality___equal(left: Spectrum[Any], right: Spectrum[Any]) -> None:
    assert left == right
    assert not (left != right)


@pytest.mark.parametrize(
    "left, right",
    [
        (Spectrum(), Spectrum(10)),
        (Spectrum(10), Spectrum(11)),
        (Spectrum(10, np.float64), Spectrum(10, np.int32)),
        (
            Spectrum(15, np.int32, start_index=5, capacity=20),
            Spectrum(10, np.int32, start_index=5, capacity=20),
        ),
        (
            Spectrum.from_array_1d([1, 4, 3], np.float64),
            Spectrum.from_array_1d([1, 2, 3], np.float64),
        ),
        (
            Spectrum.from_array_1d([1, 2, 3], np.int32),
            Spectrum.from_array_1d([1, 2, 3], np.float64),
        ),
        (
            Spectrum(
                extended_properties={"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Volts"}
            ),
            Spectrum(
                extended_properties={"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Amps"}
            ),
        ),
    ],
)
def test___different_value___equality___not_equal(
    left: Spectrum[Any], right: Spectrum[Any]
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
        (Spectrum(), "nitypes.waveform.Spectrum(0)"),
        (
            Spectrum(5),
            "nitypes.waveform.Spectrum(5, data=array([0., 0., 0., 0., 0.]))",
        ),
        (
            Spectrum(5, np.float64),
            "nitypes.waveform.Spectrum(5, data=array([0., 0., 0., 0., 0.]))",
        ),
        (Spectrum(0, np.int32), "nitypes.waveform.Spectrum(0, int32)"),
        (
            Spectrum(5, np.int32),
            f"nitypes.waveform.Spectrum(5, int32, data=array([0, 0, 0, 0, 0]{_NDARRAY_DTYPE_INT32}))",
        ),
        (
            Spectrum(5, np.int32, start_index=5, capacity=20),
            f"nitypes.waveform.Spectrum(5, int32, data=array([0, 0, 0, 0, 0]{_NDARRAY_DTYPE_INT32}))",
        ),
        (
            Spectrum.from_array_1d([1, 2, 3], np.float64),
            "nitypes.waveform.Spectrum(3, data=array([1., 2., 3.]))",
        ),
        (
            Spectrum.from_array_1d([1, 2, 3], np.int32),
            f"nitypes.waveform.Spectrum(3, int32, data=array([1, 2, 3]{_NDARRAY_DTYPE_INT32}))",
        ),
        (
            Spectrum(
                extended_properties={"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Volts"}
            ),
            "nitypes.waveform.Spectrum(0, extended_properties={'NI_ChannelName': 'Dev1/ai0', "
            "'NI_UnitDescription': 'Volts'})",
        ),
        (
            Spectrum.from_array_1d(
                [1, 2, 3],
                np.int32,
                start_frequency=123.456,
                frequency_increment=0.1,
            ),
            f"nitypes.waveform.Spectrum(3, int32, data=array([1, 2, 3]{_NDARRAY_DTYPE_INT32}), "
            "start_frequency=123.456, frequency_increment=0.1)",
        ),
        (
            Spectrum.from_array_1d(
                [1, 2, 3],
                np.int32,
                extended_properties={"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Volts"},
            ),
            f"nitypes.waveform.Spectrum(3, int32, data=array([1, 2, 3]{_NDARRAY_DTYPE_INT32}), "
            "extended_properties={'NI_ChannelName': 'Dev1/ai0', 'NI_UnitDescription': 'Volts'})",
        ),
    ],
)
def test___various_values___repr___looks_ok(value: Spectrum[Any], expected_repr: str) -> None:
    assert repr(value) == expected_repr


_VARIOUS_VALUES = [
    Spectrum(),
    Spectrum(10),
    Spectrum(10, np.float64),
    Spectrum(10, np.int32),
    Spectrum(10, np.int32, start_index=5, capacity=20),
    Spectrum.from_array_1d([1, 2, 3], np.float64),
    Spectrum.from_array_1d([1, 2, 3], np.int32),
    Spectrum(start_frequency=123.456, frequency_increment=0.1),
    Spectrum(extended_properties={"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Volts"}),
    Spectrum(10, np.int32, start_index=5, capacity=20),
    Spectrum.from_array_1d([0, 0, 1, 2, 3, 4, 5, 0], np.int32, start_index=2, sample_count=5),
]


@pytest.mark.parametrize("value", _VARIOUS_VALUES)
def test___various_values___copy___makes_shallow_copy(value: Spectrum[Any]) -> None:
    new_value = copy.copy(value)

    _assert_shallow_copy(new_value, value)


def _assert_shallow_copy(value: Spectrum[Any], other: Spectrum[Any]) -> None:
    assert value == other
    assert value is not other
    # _data may be a view of the original array.
    assert value._data is other._data or value._data.base is other._data
    assert value._start_frequency == other._start_frequency
    assert value._frequency_increment == other._frequency_increment
    assert value._extended_properties is other._extended_properties


@pytest.mark.parametrize("value", _VARIOUS_VALUES)
def test___various_values___deepcopy___makes_shallow_copy(value: Spectrum[Any]) -> None:
    new_value = copy.deepcopy(value)

    _assert_deep_copy(new_value, value)


def _assert_deep_copy(value: Spectrum[Any], other: Spectrum[Any]) -> None:
    assert value == other
    assert value is not other
    assert value._data is not other._data and value._data.base is not other._data
    assert value._start_frequency == other._start_frequency
    assert value._frequency_increment == other._frequency_increment
    assert value._extended_properties is not other._extended_properties


@pytest.mark.parametrize("value", _VARIOUS_VALUES)
def test___various_values___pickle_unpickle___makes_deep_copy(
    value: Spectrum[Any],
) -> None:
    new_value = pickle.loads(pickle.dumps(value))

    _assert_deep_copy(new_value, value)


def test___spectrum___pickle___references_public_modules() -> None:
    value = Spectrum(
        data=np.array([1, 2, 3], np.float64),
        start_frequency=123.456,
        frequency_increment=0.1,
        extended_properties={"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Volts"},
    )

    value_bytes = pickle.dumps(value)

    assert b"nitypes.waveform" in value_bytes
    assert b"nitypes.waveform._extended_properties" not in value_bytes
    assert b"nitypes.waveform._spectrum" not in value_bytes


@pytest.mark.parametrize(
    "pickled_value, expected",
    [
        # nitypes 1.0.0
        (
            b"\x80\x04\x95\xdf\x01\x00\x00\x00\x00\x00\x00\x8c\x08builtins\x94\x8c\x07getattr\x94\x93\x94\x8c\x10nitypes.waveform\x94\x8c\x08Spectrum\x94\x93\x94\x8c\t_unpickle\x94\x86\x94R\x94K\x03\x8c\x05numpy\x94\x8c\x05dtype\x94\x93\x94\x8c\x02f8\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01<\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00t\x94b\x86\x94}\x94(\x8c\x04data\x94\x8c\x16numpy._core.multiarray\x94\x8c\x0c_reconstruct\x94\x93\x94h\t\x8c\x07ndarray\x94\x93\x94K\x00\x85\x94C\x01b\x94\x87\x94R\x94(K\x01K\x03\x85\x94h\x0e\x89C\x18\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x08@\x94t\x94b\x8c\x0fstart_frequency\x94G@^\xdd/\x1a\x9f\xbew\x8c\x13frequency_increment\x94G?\xb9\x99\x99\x99\x99\x99\x9a\x8c\x13extended_properties\x94h\x03\x8c\x1aExtendedPropertyDictionary\x94\x93\x94)\x81\x94N}\x94\x8c\x0b_properties\x94}\x94(\x8c\x0eNI_ChannelName\x94\x8c\x08Dev1/ai0\x94\x8c\x12NI_UnitDescription\x94\x8c\x05Volts\x94us\x86\x94b\x8c\x18copy_extended_properties\x94\x89u\x86\x94R\x94.",
            Spectrum(
                data=np.array([1, 2, 3], np.float64),
                start_frequency=123.456,
                frequency_increment=0.1,
                extended_properties={"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Volts"},
            ),
        ),
    ],
)
def test___pickled_value___unpickle___is_compatible(
    pickled_value: bytes, expected: Spectrum[Any]
) -> None:
    new_value = pickle.loads(pickled_value)
    assert new_value == expected
