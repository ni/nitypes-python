from __future__ import annotations

import copy
import datetime as dt
import pickle
from typing import Any

import hightime as ht
import numpy as np
import numpy.typing as npt
import pytest
from typing_extensions import assert_type

from nitypes.complex import ComplexInt32Base, ComplexInt32DType
from nitypes.waveform import (
    NO_SCALING,
    ComplexWaveform,
    LinearScaleMode,
    PrecisionTiming,
    Timing,
)


###############################################################################
# create
###############################################################################
def test___sample_count_and_complexint32_dtype___create___creates_waveform_with_sample_count_and_dtype() -> (
    None
):
    waveform = ComplexWaveform(10, ComplexInt32DType)

    assert waveform.sample_count == waveform.capacity == len(waveform.raw_data) == 10
    assert waveform.dtype == ComplexInt32DType
    assert_type(waveform, ComplexWaveform[ComplexInt32Base])


def test___sample_count_and_complex64_dtype___create___creates_waveform_with_sample_count_and_dtype() -> (
    None
):
    waveform = ComplexWaveform(10, np.complex64)

    assert waveform.sample_count == waveform.capacity == len(waveform.raw_data) == 10
    assert waveform.dtype == np.complex64
    assert_type(waveform, ComplexWaveform[np.complex64])


def test___sample_count_and_complex128_dtype___create___creates_waveform_with_sample_count_and_dtype() -> (
    None
):
    waveform = ComplexWaveform(10, np.complex128)

    assert waveform.sample_count == waveform.capacity == len(waveform.raw_data) == 10
    assert waveform.dtype == np.complex128
    assert_type(waveform, ComplexWaveform[np.complex128])


def test___sample_count_and_unknown_structured_dtype___create___raises_type_error() -> None:
    dtype = np.dtype([("a", np.int16), ("b", np.int32)])

    with pytest.raises(TypeError) as exc:
        waveform = ComplexWaveform(10, dtype)

        # ComplexWaveform currently cannot distinguish between ComplexInt32DType and other
        # structured data types at type-checking time.
        assert_type(waveform, ComplexWaveform[ComplexInt32Base])

    assert exc.value.args[0].startswith("The requested data type is not supported.")
    assert "Data type: [('a', '<i2'), ('b', '<i4')]" in exc.value.args[0]


def test___sample_count_and_structured_dtype_str___create___raises_type_error() -> None:
    with pytest.raises(TypeError) as exc:
        _ = ComplexWaveform(10, "i2, i2")

    assert exc.value.args[0].startswith("The requested data type is not supported.")
    assert "Data type: [('f0', '<i2'), ('f1', '<i2')]" in exc.value.args[0]


@pytest.mark.parametrize("dtype", [np.int32, np.float64, np.str_])
def test___sample_count_and_unsupported_dtype___create___raises_type_error(
    dtype: npt.DTypeLike,
) -> None:
    with pytest.raises(TypeError) as exc:
        _ = ComplexWaveform(10, dtype)

    assert exc.value.args[0].startswith("The requested data type is not supported.")


def test___dtype_str_with_unsupported_traw_hint___create___mypy_type_var_warning() -> None:
    waveform1: ComplexWaveform[np.int32] = ComplexWaveform(dtype="complex64")  # type: ignore[type-var]
    waveform2: ComplexWaveform[np.float64] = ComplexWaveform(dtype="complex64")  # type: ignore[type-var]
    waveform3: ComplexWaveform[np.str_] = ComplexWaveform(dtype="complex64")  # type: ignore[type-var]
    _ = waveform1, waveform2, waveform3


def test___dtype_str_with_traw_hint___create___narrows_traw() -> None:
    waveform: ComplexWaveform[np.complex64] = ComplexWaveform(dtype="complex64")

    assert_type(waveform, ComplexWaveform[np.complex64])


###############################################################################
# from_array_1d
###############################################################################
def test___complexint32_ndarray___from_array_1d___creates_waveform_with_complexint32_dtype() -> (
    None
):
    data = np.array([(1, 2), (3, -4)], ComplexInt32DType)

    waveform = ComplexWaveform.from_array_1d(data)

    assert waveform.raw_data.tolist() == data.tolist()
    assert waveform.dtype == ComplexInt32DType
    assert_type(waveform, ComplexWaveform[ComplexInt32Base])


def test___complex64_ndarray___from_array_1d___creates_waveform_with_complex64_dtype() -> None:
    data = np.array([1.1 + 2.2j, 3.3 - 4.4j], np.complex64)

    waveform = ComplexWaveform.from_array_1d(data)

    assert waveform.raw_data.tolist() == data.tolist()
    assert waveform.dtype == np.complex64
    assert_type(waveform, ComplexWaveform[np.complex64])


def test___complex128_ndarray___from_array_1d___creates_waveform_with_complex128_dtype() -> None:
    data = np.array([1.1 + 2.2j, 3.3 - 4.4j], np.complex128)

    waveform = ComplexWaveform.from_array_1d(data)

    assert waveform.raw_data.tolist() == data.tolist()
    assert waveform.dtype == np.complex128
    assert_type(waveform, ComplexWaveform[np.complex128])


def test___complex_list_with_dtype___from_array_1d___creates_waveform_with_specified_dtype() -> (
    None
):
    data = [1.1 + 2.2j, 3.3 - 4.4j]

    waveform = ComplexWaveform.from_array_1d(data, np.complex64)

    assert waveform.raw_data.tolist() == pytest.approx(data)
    assert waveform.dtype == np.complex64
    assert_type(waveform, ComplexWaveform[np.complex64])


def test___complex_list_with_dtype_str___from_array_1d___creates_waveform_with_specified_dtype() -> (
    None
):
    data = [1.1 + 2.2j, 3.3 - 4.4j]

    waveform = ComplexWaveform.from_array_1d(data, "complex64")

    assert waveform.raw_data.tolist() == pytest.approx(data)
    assert waveform.dtype == np.complex64
    assert_type(waveform, ComplexWaveform[Any])  # dtype not inferred from string


###############################################################################
# from_array_2d
###############################################################################
def test___complexint32_ndarray___from_array_2d___creates_waveform_with_complexint32_dtype() -> (
    None
):
    data = np.array([[(1, 2), (3, -4), (-5, 6)], [(-7 - 8), (9, 10), (11, 12)]], ComplexInt32DType)

    waveforms = ComplexWaveform.from_array_2d(data)

    assert len(waveforms) == 2
    for i in range(len(waveforms)):
        assert waveforms[i].raw_data.tolist() == data[i].tolist()
        assert waveforms[i].dtype == ComplexInt32DType
        assert_type(waveforms[i], ComplexWaveform[ComplexInt32Base])


def test___complex64_ndarray___from_array_2d___creates_waveform_with_complex64_dtype() -> None:
    data = np.array([[1 + 2j, 3 - 4j, -5 + 6j], [-7 - 8j, 9 + 10j, 11 + 12j]], np.complex64)

    waveforms = ComplexWaveform.from_array_2d(data)

    assert len(waveforms) == 2
    for i in range(len(waveforms)):
        assert waveforms[i].raw_data.tolist() == data[i].tolist()
        assert waveforms[i].dtype == np.complex64
        assert_type(waveforms[i], ComplexWaveform[np.complex64])


def test___complex128_ndarray___from_array_2d___creates_waveform_with_complex128_dtype() -> None:
    data = np.array([[1 + 2j, 3 - 4j, -5 + 6j], [-7 - 8j, 9 + 10j, 11 + 12j]], np.complex128)

    waveforms = ComplexWaveform.from_array_2d(data)

    assert len(waveforms) == 2
    for i in range(len(waveforms)):
        assert waveforms[i].raw_data.tolist() == data[i].tolist()
        assert waveforms[i].dtype == np.complex128
        assert_type(waveforms[i], ComplexWaveform[np.complex128])


def test___complex_list_list_with_dtype___from_array_2d___creates_waveform_with_specified_dtype() -> (
    None
):
    data = [[1 + 2j, 3 - 4j, -5 + 6j], [-7 - 8j, 9 + 10j, 11 + 12j]]

    waveforms = ComplexWaveform.from_array_2d(data, np.complex64)

    assert len(waveforms) == 2
    for i in range(len(waveforms)):
        assert waveforms[i].raw_data.tolist() == data[i]
        assert waveforms[i].dtype == np.complex64
        assert_type(waveforms[i], ComplexWaveform[np.complex64])


def test___int_list_list_with_dtype_str___from_array_2d___creates_waveform_with_specified_dtype() -> (
    None
):
    data = [[1 + 2j, 3 - 4j, -5 + 6j], [-7 - 8j, 9 + 10j, 11 + 12j]]

    waveforms = ComplexWaveform.from_array_2d(data, "complex64")

    assert len(waveforms) == 2
    for i in range(len(waveforms)):
        assert waveforms[i].raw_data.tolist() == data[i]
        assert waveforms[i].dtype == np.complex64
        assert_type(waveforms[i], ComplexWaveform[Any])  # dtype not inferred from string


###############################################################################
# raw_data
###############################################################################
def test___complexint32_waveform___raw_data___returns_complexint32_data() -> None:
    waveform = ComplexWaveform.from_array_1d([(1, 2), (3, -4)], ComplexInt32DType)

    raw_data = waveform.raw_data

    assert_type(raw_data, npt.NDArray[ComplexInt32Base])
    assert isinstance(raw_data, np.ndarray) and raw_data.dtype == ComplexInt32DType
    assert [x.item() for x in raw_data] == [(1, 2), (3, -4)]


def test___complex64_waveform___raw_data___returns_complex64_data() -> None:
    waveform = ComplexWaveform.from_array_1d([1 + 2j, 3 - 4j], np.complex64)

    raw_data = waveform.raw_data

    assert_type(raw_data, npt.NDArray[np.complex64])
    assert isinstance(raw_data, np.ndarray) and raw_data.dtype == np.complex64
    assert list(raw_data) == [1 + 2j, 3 - 4j]


###############################################################################
# scaled_data
###############################################################################
def test___complexint32_waveform___scaled_data___converts_to_complex128() -> None:
    waveform = ComplexWaveform.from_array_1d([(1, 2), (3, -4)], ComplexInt32DType)

    scaled_data = waveform.scaled_data

    assert_type(scaled_data, npt.NDArray[np.complex128])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.complex128
    assert list(scaled_data) == [1 + 2j, 3 - 4j]


def test___complex64_waveform___scaled_data___converts_to_complex128() -> None:
    waveform = ComplexWaveform.from_array_1d([1 + 2j, 3 - 4j], np.complex64)

    scaled_data = waveform.scaled_data

    assert_type(scaled_data, npt.NDArray[np.complex128])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.complex128
    assert list(scaled_data) == [1 + 2j, 3 - 4j]


def test___complexint32_waveform_with_linear_scale___scaled_data___converts_to_complex128() -> None:
    waveform = ComplexWaveform.from_array_1d([(1, 2), (3, -4)], ComplexInt32DType)
    waveform.scale_mode = LinearScaleMode(2.0, 0.5)

    scaled_data = waveform.scaled_data

    assert_type(scaled_data, npt.NDArray[np.complex128])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.complex128
    assert list(scaled_data) == [2.5 + 4j, 6.5 - 8j]


def test___complex64_waveform_with_linear_scale___scaled_data___converts_to_complex128() -> None:
    waveform = ComplexWaveform.from_array_1d([1 + 2j, 3 - 4j], np.complex64)
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
    waveform = ComplexWaveform.from_array_1d([(1, 2), (3, -4)], ComplexInt32DType)

    scaled_data = waveform.get_scaled_data(np.complex64)

    assert_type(scaled_data, npt.NDArray[np.complex64])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.complex64
    assert list(scaled_data) == [1 + 2j, 3 - 4j]


def test___complex64_waveform_with_complex64_dtype___get_scaled_data___does_not_convert() -> None:
    waveform = ComplexWaveform.from_array_1d([1 + 2j, 3 - 4j], np.complex64)

    scaled_data = waveform.get_scaled_data(np.complex64)

    assert_type(scaled_data, npt.NDArray[np.complex64])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.complex64
    assert list(scaled_data) == [1 + 2j, 3 - 4j]


def test___complexint32_waveform_with_unknown_structured_dtype___get_scaled_data___raises_type_error() -> (
    None
):
    waveform = ComplexWaveform.from_array_1d([(1, 2), (3, -4)], ComplexInt32DType)
    dtype = np.dtype([("a", np.int16), ("b", np.int16)])

    with pytest.raises(TypeError) as exc:
        _ = waveform.get_scaled_data(dtype)

    assert exc.value.args[0].startswith("The requested data type is not supported.")
    assert "Data type: [('a', '<i2'), ('b', '<i2')]" in exc.value.args[0]
    assert "Supported data types: complex64, complex128" in exc.value.args[0]


###############################################################################
# magic methods
###############################################################################
@pytest.mark.parametrize(
    "left, right",
    [
        (ComplexWaveform(), ComplexWaveform()),
        (ComplexWaveform(10), ComplexWaveform(10)),
        (ComplexWaveform(10, np.complex128), ComplexWaveform(10, np.complex128)),
        (ComplexWaveform(10, ComplexInt32DType), ComplexWaveform(10, ComplexInt32DType)),
        (
            ComplexWaveform(10, ComplexInt32DType, start_index=5, capacity=20),
            ComplexWaveform(10, ComplexInt32DType, start_index=5, capacity=20),
        ),
        (
            ComplexWaveform.from_array_1d([1 + 2j, 3 + 4j, 5 + 6j], np.complex128),
            ComplexWaveform.from_array_1d([1 + 2j, 3 + 4j, 5 + 6j], np.complex128),
        ),
        (
            ComplexWaveform.from_array_1d([(1, 2), (3, 4), (5, 6)], ComplexInt32DType),
            ComplexWaveform.from_array_1d([(1, 2), (3, 4), (5, 6)], ComplexInt32DType),
        ),
        (
            ComplexWaveform(
                timing=Timing.create_with_regular_interval(dt.timedelta(milliseconds=1))
            ),
            ComplexWaveform(
                timing=Timing.create_with_regular_interval(dt.timedelta(milliseconds=1))
            ),
        ),
        (
            ComplexWaveform(
                timing=PrecisionTiming.create_with_regular_interval(ht.timedelta(milliseconds=1))
            ),
            ComplexWaveform(
                timing=PrecisionTiming.create_with_regular_interval(ht.timedelta(milliseconds=1))
            ),
        ),
        (
            ComplexWaveform(
                extended_properties={"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Volts"}
            ),
            ComplexWaveform(
                extended_properties={"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Volts"}
            ),
        ),
        (
            ComplexWaveform(scale_mode=LinearScaleMode(2.0, 1.0)),
            ComplexWaveform(scale_mode=LinearScaleMode(2.0, 1.0)),
        ),
        # start_index and capacity may differ as long as raw_data and sample_count are the same.
        (
            ComplexWaveform(10, ComplexInt32DType, start_index=5, capacity=20),
            ComplexWaveform(10, ComplexInt32DType, start_index=10, capacity=25),
        ),
        (
            ComplexWaveform.from_array_1d(
                [0, 0, 1, 2, 3, 4, 5, 0], ComplexInt32DType, start_index=2, sample_count=5
            ),
            ComplexWaveform.from_array_1d(
                [0, 1, 2, 3, 4, 5, 0, 0, 0], ComplexInt32DType, start_index=1, sample_count=5
            ),
        ),
    ],
)
def test___same_value___equality___equal(
    left: ComplexWaveform[Any], right: ComplexWaveform[Any]
) -> None:
    assert left == right
    assert not (left != right)


@pytest.mark.parametrize(
    "left, right",
    [
        (ComplexWaveform(), ComplexWaveform(10)),
        (ComplexWaveform(10), ComplexWaveform(11)),
        (ComplexWaveform(10, np.complex128), ComplexWaveform(10, ComplexInt32DType)),
        (
            ComplexWaveform(15, ComplexInt32DType, start_index=5, capacity=20),
            ComplexWaveform(10, ComplexInt32DType, start_index=5, capacity=20),
        ),
        (
            ComplexWaveform.from_array_1d([1 + 2j, 3 + 5j, 5 + 6j], np.complex128),
            ComplexWaveform.from_array_1d([1 + 2j, 3 + 4j, 5 + 6j], np.complex128),
        ),
        (
            ComplexWaveform.from_array_1d([(1, 2), (3, 4), (5, 6)], ComplexInt32DType),
            ComplexWaveform.from_array_1d([1 + 2j, 3 + 4j, 5 + 6j], np.complex128),
        ),
        (
            ComplexWaveform(
                timing=Timing.create_with_regular_interval(dt.timedelta(milliseconds=1))
            ),
            ComplexWaveform(
                timing=Timing.create_with_regular_interval(dt.timedelta(milliseconds=2))
            ),
        ),
        (
            ComplexWaveform(
                timing=PrecisionTiming.create_with_regular_interval(ht.timedelta(milliseconds=1))
            ),
            ComplexWaveform(
                timing=PrecisionTiming.create_with_regular_interval(ht.timedelta(milliseconds=2))
            ),
        ),
        (
            ComplexWaveform(
                extended_properties={"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Volts"}
            ),
            ComplexWaveform(
                extended_properties={"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Amps"}
            ),
        ),
        (
            ComplexWaveform(scale_mode=LinearScaleMode(2.0, 1.0)),
            ComplexWaveform(scale_mode=LinearScaleMode(2.0, 1.1)),
        ),
        (
            ComplexWaveform(scale_mode=NO_SCALING),
            ComplexWaveform(scale_mode=LinearScaleMode(2.0, 1.0)),
        ),
        # __eq__ does not convert timing, even if the values are equivalent.
        (
            ComplexWaveform(
                timing=Timing.create_with_regular_interval(dt.timedelta(milliseconds=1))
            ),
            ComplexWaveform(
                timing=PrecisionTiming.create_with_regular_interval(ht.timedelta(milliseconds=1))
            ),
        ),
        (
            ComplexWaveform(
                timing=PrecisionTiming.create_with_regular_interval(ht.timedelta(milliseconds=1))
            ),
            ComplexWaveform(
                timing=Timing.create_with_regular_interval(dt.timedelta(milliseconds=1))
            ),
        ),
    ],
)
def test___different_value___equality___not_equal(
    left: ComplexWaveform[Any], right: ComplexWaveform[Any]
) -> None:
    assert not (left == right)
    assert left != right


@pytest.mark.parametrize(
    "value, expected_repr",
    [
        (ComplexWaveform(), "nitypes.waveform.ComplexWaveform(0)"),
        (
            ComplexWaveform(5),
            "nitypes.waveform.ComplexWaveform(5, raw_data=array([0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]))",
        ),
        (
            ComplexWaveform(5, np.complex128),
            "nitypes.waveform.ComplexWaveform(5, raw_data=array([0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]))",
        ),
        (ComplexWaveform(0, ComplexInt32DType), "nitypes.waveform.ComplexWaveform(0, void32)"),
        (
            ComplexWaveform(5, ComplexInt32DType),
            "nitypes.waveform.ComplexWaveform(5, void32, raw_data=array([(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],\n"
            "      dtype=[('real', '<i2'), ('imag', '<i2')]))",
        ),
        (
            ComplexWaveform(5, ComplexInt32DType, start_index=5, capacity=20),
            "nitypes.waveform.ComplexWaveform(5, void32, raw_data=array([(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],\n"
            "      dtype=[('real', '<i2'), ('imag', '<i2')]))",
        ),
        (
            ComplexWaveform.from_array_1d([1.23 + 3.45j, 6.78 - 9.01j], np.complex128),
            "nitypes.waveform.ComplexWaveform(2, raw_data=array([1.23+3.45j, 6.78-9.01j]))",
        ),
        (
            ComplexWaveform.from_array_1d([(1, 2), (3, 4), (5, 6)], ComplexInt32DType),
            "nitypes.waveform.ComplexWaveform(3, void32, raw_data=array([(1, 2), (3, 4), (5, 6)], dtype=[('real', '<i2'), ('imag', '<i2')]))",
        ),
        (
            ComplexWaveform(
                timing=Timing.create_with_regular_interval(dt.timedelta(milliseconds=1))
            ),
            "nitypes.waveform.ComplexWaveform(0, timing=nitypes.waveform.Timing(nitypes.waveform.SampleIntervalMode.REGULAR, sample_interval=datetime.timedelta(microseconds=1000)))",
        ),
        (
            ComplexWaveform(
                timing=PrecisionTiming.create_with_regular_interval(ht.timedelta(milliseconds=1))
            ),
            "nitypes.waveform.ComplexWaveform(0, timing=nitypes.waveform.PrecisionTiming(nitypes.waveform.SampleIntervalMode.REGULAR, sample_interval=hightime.timedelta(microseconds=1000)))",
        ),
        (
            ComplexWaveform(
                extended_properties={"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Volts"}
            ),
            "nitypes.waveform.ComplexWaveform(0, extended_properties={'NI_ChannelName': 'Dev1/ai0', 'NI_UnitDescription': 'Volts'})",
        ),
        (
            ComplexWaveform(scale_mode=LinearScaleMode(2.0, 1.0)),
            "nitypes.waveform.ComplexWaveform(0, scale_mode=nitypes.waveform.LinearScaleMode(2.0, 1.0))",
        ),
        (
            ComplexWaveform.from_array_1d(
                [(1, 2), (3, 4), (5, 6)],
                ComplexInt32DType,
                timing=Timing.create_with_regular_interval(dt.timedelta(milliseconds=1)),
            ),
            "nitypes.waveform.ComplexWaveform(3, void32, raw_data=array([(1, 2), (3, 4), (5, 6)], dtype=[('real', '<i2'), ('imag', '<i2')]), timing=nitypes.waveform.Timing(nitypes.waveform.SampleIntervalMode.REGULAR, sample_interval=datetime.timedelta(microseconds=1000)))",
        ),
        (
            ComplexWaveform.from_array_1d(
                [(1, 2), (3, 4), (5, 6)],
                ComplexInt32DType,
                extended_properties={"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Volts"},
            ),
            "nitypes.waveform.ComplexWaveform(3, void32, raw_data=array([(1, 2), (3, 4), (5, 6)], dtype=[('real', '<i2'), ('imag', '<i2')]), extended_properties={'NI_ChannelName': 'Dev1/ai0', 'NI_UnitDescription': 'Volts'})",
        ),
        (
            ComplexWaveform.from_array_1d(
                [(1, 2), (3, 4), (5, 6)], ComplexInt32DType, scale_mode=LinearScaleMode(2.0, 1.0)
            ),
            "nitypes.waveform.ComplexWaveform(3, void32, raw_data=array([(1, 2), (3, 4), (5, 6)], dtype=[('real', '<i2'), ('imag', '<i2')]), scale_mode=nitypes.waveform.LinearScaleMode(2.0, 1.0))",
        ),
    ],
)
def test___various_values___repr___looks_ok(
    value: ComplexWaveform[Any], expected_repr: str
) -> None:
    assert repr(value) == expected_repr


_VARIOUS_VALUES = [
    ComplexWaveform(),
    ComplexWaveform(10),
    ComplexWaveform(10, np.complex128),
    ComplexWaveform(10, ComplexInt32DType),
    ComplexWaveform(10, ComplexInt32DType, start_index=5, capacity=20),
    ComplexWaveform.from_array_1d([123 + 3.45j, 6.78 - 9.01j], np.complex128),
    ComplexWaveform.from_array_1d([(1, 2), (3, 4), (5, 6)], ComplexInt32DType),
    ComplexWaveform(timing=Timing.create_with_regular_interval(dt.timedelta(milliseconds=1))),
    ComplexWaveform(
        timing=PrecisionTiming.create_with_regular_interval(ht.timedelta(milliseconds=1))
    ),
    ComplexWaveform(
        extended_properties={"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Volts"}
    ),
    ComplexWaveform(scale_mode=LinearScaleMode(2.0, 1.0)),
    ComplexWaveform(10, ComplexInt32DType, start_index=5, capacity=20),
    ComplexWaveform.from_array_1d(
        [(0, 0), (0, 0), (1, 1), (2, -2), (3, 33), (4, -44), (5, 50), (0, 0)],
        ComplexInt32DType,
        start_index=2,
        sample_count=5,
    ),
]


@pytest.mark.parametrize("value", _VARIOUS_VALUES)
def test___various_values___copy___makes_shallow_copy(value: ComplexWaveform[Any]) -> None:
    new_value = copy.copy(value)

    _assert_shallow_copy(new_value, value)


def _assert_shallow_copy(value: ComplexWaveform[Any], other: ComplexWaveform[Any]) -> None:
    assert value == other
    assert value is not other
    # _data may be a view of the original array.
    assert value._data is other._data or value._data.base is other._data
    assert value._extended_properties is other._extended_properties
    assert value._timing is other._timing
    assert value._scale_mode is other._scale_mode


@pytest.mark.parametrize("value", _VARIOUS_VALUES)
def test___various_values___deepcopy___makes_shallow_copy(value: ComplexWaveform[Any]) -> None:
    new_value = copy.deepcopy(value)

    _assert_deep_copy(new_value, value)


def _assert_deep_copy(value: ComplexWaveform[Any], other: ComplexWaveform[Any]) -> None:
    assert value == other
    assert value is not other
    assert value._data is not other._data and value._data.base is not other._data
    assert value._extended_properties is not other._extended_properties
    if other._timing is not Timing.empty and other._timing is not PrecisionTiming.empty:
        assert value._timing is not other._timing
    if other._scale_mode is not NO_SCALING:
        assert value._scale_mode is not other._scale_mode


@pytest.mark.parametrize("value", _VARIOUS_VALUES)
def test___various_values___pickle_unpickle___makes_deep_copy(
    value: ComplexWaveform[Any],
) -> None:
    new_value = pickle.loads(pickle.dumps(value))

    _assert_deep_copy(new_value, value)


def test___waveform___pickle___references_public_modules() -> None:
    value = ComplexWaveform(
        raw_data=np.array([1, 2, 3], np.complex128),
        extended_properties={"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Volts"},
        timing=Timing.create_with_regular_interval(dt.timedelta(milliseconds=1)),
        scale_mode=LinearScaleMode(2.0, 1.0),
    )

    value_bytes = pickle.dumps(value)

    assert b"nitypes.waveform" in value_bytes
    assert b"nitypes.waveform._complex" not in value_bytes
    assert b"nitypes.waveform._extended_properties" not in value_bytes
    assert b"nitypes.waveform._numeric" not in value_bytes
    assert b"nitypes.waveform._timing" not in value_bytes
    assert b"nitypes.waveform._scaling" not in value_bytes
