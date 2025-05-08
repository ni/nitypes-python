from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from typing_extensions import assert_type

from nitypes.complex import ComplexInt32Base, ComplexInt32DType
from nitypes.waveform import AnalogWaveform, ComplexWaveform, LinearScaleMode


###############################################################################
# create
###############################################################################
def test___sample_count_and_complexint32_dtype___create___creates_waveform_with_sample_count_and_dtype() -> (
    None
):
    waveform = AnalogWaveform(10, ComplexInt32DType)

    assert waveform.sample_count == waveform.capacity == len(waveform.raw_data) == 10
    assert waveform.dtype == ComplexInt32DType
    assert_type(waveform, ComplexWaveform[ComplexInt32Base])


def test___sample_count_and_complex64_dtype___create___creates_waveform_with_sample_count_and_dtype() -> (
    None
):
    waveform = AnalogWaveform(10, np.complex64)

    assert waveform.sample_count == waveform.capacity == len(waveform.raw_data) == 10
    assert waveform.dtype == np.complex64
    assert_type(waveform, ComplexWaveform[np.complex64])


def test___sample_count_and_complex128_dtype___create___creates_waveform_with_sample_count_and_dtype() -> (
    None
):
    waveform = AnalogWaveform(10, np.complex128)

    assert waveform.sample_count == waveform.capacity == len(waveform.raw_data) == 10
    assert waveform.dtype == np.complex128
    assert_type(waveform, ComplexWaveform[np.complex128])


def test___sample_count_and_unknown_structured_dtype___create___raises_type_error() -> None:
    dtype = np.dtype([("a", np.int16), ("b", np.int32)])

    with pytest.raises(TypeError) as exc:
        waveform = AnalogWaveform(10, dtype)

        # AnalogWaveform currently cannot distinguish between ComplexInt32DType and other structured
        # data types at type-checking time.
        assert_type(waveform, ComplexWaveform[ComplexInt32Base])

    assert exc.value.args[0].startswith("The requested data type is not supported.")
    assert "Data type: [('a', '<i2'), ('b', '<i4')]" in exc.value.args[0]


###############################################################################
# from_array_1d
###############################################################################
def test___complexint32_ndarray___from_array_1d___creates_waveform_with_complexint32_dtype() -> (
    None
):
    data = np.array([(1, 2), (3, -4)], ComplexInt32DType)

    waveform = AnalogWaveform.from_array_1d(data)

    assert waveform.raw_data.tolist() == data.tolist()
    assert waveform.dtype == ComplexInt32DType
    assert_type(waveform, ComplexWaveform[ComplexInt32Base])


def test___complex64_ndarray___from_array_1d___creates_waveform_with_complex64_dtype() -> None:
    data = np.array([1.1 + 2.2j, 3.3 - 4.4j], np.complex64)

    waveform = AnalogWaveform.from_array_1d(data)

    assert waveform.raw_data.tolist() == data.tolist()
    assert waveform.dtype == np.complex64
    assert_type(waveform, ComplexWaveform[np.complex64])


def test___complex128_ndarray___from_array_1d___creates_waveform_with_complex128_dtype() -> None:
    data = np.array([1.1 + 2.2j, 3.3 - 4.4j], np.complex128)

    waveform = AnalogWaveform.from_array_1d(data)

    assert waveform.raw_data.tolist() == data.tolist()
    assert waveform.dtype == np.complex128
    assert_type(waveform, ComplexWaveform[np.complex128])


def test___complex_list_with_dtype___from_array_1d___creates_waveform_with_specified_dtype() -> (
    None
):
    data = [1.1 + 2.2j, 3.3 - 4.4j]

    waveform = AnalogWaveform.from_array_1d(data, np.complex64)

    assert waveform.raw_data.tolist() == pytest.approx(data)
    assert waveform.dtype == np.complex64
    assert_type(waveform, ComplexWaveform[np.complex64])


def test___complex_list_with_dtype_str___from_array_1d___creates_waveform_with_specified_dtype() -> (
    None
):
    data = [1.1 + 2.2j, 3.3 - 4.4j]

    waveform = AnalogWaveform.from_array_1d(data, "complex64")

    assert waveform.raw_data.tolist() == pytest.approx(data)
    assert waveform.dtype == np.complex64
    assert_type(waveform, AnalogWaveform[Any, Any])  # dtype not inferred from string


###############################################################################
# from_array_2d
###############################################################################
def test___complexint32_ndarray___from_array_2d___creates_waveform_with_complexint32_dtype() -> (
    None
):
    data = np.array([[(1, 2), (3, -4), (-5, 6)], [(-7 - 8), (9, 10), (11, 12)]], ComplexInt32DType)

    waveforms = AnalogWaveform.from_array_2d(data)

    assert len(waveforms) == 2
    for i in range(len(waveforms)):
        assert waveforms[i].raw_data.tolist() == data[i].tolist()
        assert waveforms[i].dtype == ComplexInt32DType
        assert_type(waveforms[i], ComplexWaveform[ComplexInt32Base])


def test___complex64_ndarray___from_array_2d___creates_waveform_with_complex64_dtype() -> None:
    data = np.array([[1 + 2j, 3 - 4j, -5 + 6j], [-7 - 8j, 9 + 10j, 11 + 12j]], np.complex64)

    waveforms = AnalogWaveform.from_array_2d(data)

    assert len(waveforms) == 2
    for i in range(len(waveforms)):
        assert waveforms[i].raw_data.tolist() == data[i].tolist()
        assert waveforms[i].dtype == np.complex64
        assert_type(waveforms[i], ComplexWaveform[np.complex64])


def test___complex128_ndarray___from_array_2d___creates_waveform_with_complex128_dtype() -> None:
    data = np.array([[1 + 2j, 3 - 4j, -5 + 6j], [-7 - 8j, 9 + 10j, 11 + 12j]], np.complex128)

    waveforms = AnalogWaveform.from_array_2d(data)

    assert len(waveforms) == 2
    for i in range(len(waveforms)):
        assert waveforms[i].raw_data.tolist() == data[i].tolist()
        assert waveforms[i].dtype == np.complex128
        assert_type(waveforms[i], ComplexWaveform[np.complex128])


def test___complex_list_list_with_dtype___from_array_2d___creates_waveform_with_specified_dtype() -> (
    None
):
    data = [[1 + 2j, 3 - 4j, -5 + 6j], [-7 - 8j, 9 + 10j, 11 + 12j]]

    waveforms = AnalogWaveform.from_array_2d(data, np.complex64)

    assert len(waveforms) == 2
    for i in range(len(waveforms)):
        assert waveforms[i].raw_data.tolist() == data[i]
        assert waveforms[i].dtype == np.complex64
        assert_type(waveforms[i], ComplexWaveform[np.complex64])


def test___int_list_list_with_dtype_str___from_array_2d___creates_waveform_with_specified_dtype() -> (
    None
):
    data = [[1 + 2j, 3 - 4j, -5 + 6j], [-7 - 8j, 9 + 10j, 11 + 12j]]

    waveforms = AnalogWaveform.from_array_2d(data, "complex64")

    assert len(waveforms) == 2
    for i in range(len(waveforms)):
        assert waveforms[i].raw_data.tolist() == data[i]  # type: ignore[comparison-overlap]
        assert waveforms[i].dtype == np.complex64
        assert_type(waveforms[i], AnalogWaveform[Any, Any])  # dtype not inferred from string


###############################################################################
# raw_data
###############################################################################
def test___complexint32_waveform___raw_data___returns_complexint32_data() -> None:
    waveform = AnalogWaveform.from_array_1d([(1, 2), (3, -4)], ComplexInt32DType)

    raw_data = waveform.raw_data

    assert_type(raw_data, npt.NDArray[ComplexInt32Base])
    assert isinstance(raw_data, np.ndarray) and raw_data.dtype == ComplexInt32DType
    assert [x.item() for x in raw_data] == [(1, 2), (3, -4)]


def test___complex64_waveform___raw_data___returns_complex64_data() -> None:
    waveform = AnalogWaveform.from_array_1d([1 + 2j, 3 - 4j], np.complex64)

    raw_data = waveform.raw_data

    assert_type(raw_data, npt.NDArray[np.complex64])
    assert isinstance(raw_data, np.ndarray) and raw_data.dtype == np.complex64
    assert list(raw_data) == [1 + 2j, 3 - 4j]


###############################################################################
# scaled_data
###############################################################################
def test___complexint32_waveform___scaled_data___converts_to_complex128() -> None:
    waveform = AnalogWaveform.from_array_1d([(1, 2), (3, -4)], ComplexInt32DType)

    scaled_data = waveform.scaled_data

    assert_type(scaled_data, npt.NDArray[np.complex128])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.complex128
    assert list(scaled_data) == [1 + 2j, 3 - 4j]


def test___complex64_waveform___scaled_data___converts_to_complex128() -> None:
    waveform = AnalogWaveform.from_array_1d([1 + 2j, 3 - 4j], np.complex64)

    scaled_data = waveform.scaled_data

    assert_type(scaled_data, npt.NDArray[np.complex128])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.complex128
    assert list(scaled_data) == [1 + 2j, 3 - 4j]


def test___complexint32_waveform_with_linear_scale___scaled_data___converts_to_complex128() -> None:
    waveform = AnalogWaveform.from_array_1d([(1, 2), (3, -4)], ComplexInt32DType)
    waveform.scale_mode = LinearScaleMode(2.0, 0.5)

    scaled_data = waveform.scaled_data

    assert_type(scaled_data, npt.NDArray[np.complex128])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.complex128
    assert list(scaled_data) == [2.5 + 4j, 6.5 - 8j]


def test___complex64_waveform_with_linear_scale___scaled_data___converts_to_complex128() -> None:
    waveform = AnalogWaveform.from_array_1d([1 + 2j, 3 - 4j], np.complex64)
    waveform.scale_mode = LinearScaleMode(2.0, 0.5)

    scaled_data = waveform.scaled_data

    assert_type(scaled_data, npt.NDArray[np.complex128])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.complex128
    assert list(scaled_data) == [2.5 + 4j, 6.5 - 8j]


###############################################################################
# get_scaled_data
###############################################################################
def test___complexint32_waveform_with_complex64_dtype___get_scaled_data___converts_to_complex64() -> (
    None
):
    waveform = AnalogWaveform.from_array_1d([(1, 2), (3, -4)], ComplexInt32DType)

    scaled_data = waveform.get_scaled_data(np.complex64)

    assert_type(scaled_data, npt.NDArray[np.complex64])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.complex64
    assert list(scaled_data) == [1 + 2j, 3 - 4j]


def test___complex64_waveform_with_complex64_dtype___get_scaled_data___does_not_convert() -> None:
    waveform = AnalogWaveform.from_array_1d([1 + 2j, 3 - 4j], np.complex64)

    scaled_data = waveform.get_scaled_data(np.complex64)

    assert_type(scaled_data, npt.NDArray[np.complex64])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.complex64
    assert list(scaled_data) == [1 + 2j, 3 - 4j]


def test___complexint32_waveform_with_unknown_structured_dtype___get_scaled_data___raises_type_error() -> (
    None
):
    waveform = AnalogWaveform.from_array_1d([(1, 2), (3, -4)], ComplexInt32DType)
    dtype = np.dtype([("a", np.int16), ("b", np.int16)])

    with pytest.raises(TypeError) as exc:
        _ = waveform.get_scaled_data(dtype)

    assert exc.value.args[0].startswith("The requested data type is not supported.")
    assert "Data type: [('a', '<i2'), ('b', '<i2')]" in exc.value.args[0]
    assert "Supported data types: complex64, complex128" in exc.value.args[0]
