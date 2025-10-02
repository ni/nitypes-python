from __future__ import annotations

import array
import copy
import itertools
import pickle
from typing import Any

import numpy as np
import pytest
from typing_extensions import assert_type

from nitypes.waveform._extended_properties import ExtendedPropertyDictionary
from nitypes.xy_data import _TData, _UNIT_DESCRIPTION_X, _UNIT_DESCRIPTION_Y, XYData


###############################################################################
# create
###############################################################################
def test___data_and_dtype___create___creates_xydata_with_data_and_dtype() -> None:
    data = np.array([1, 2, 3, 4, 5], np.int32)
    xydata = XYData(data, data)

    assert xydata.dtype == np.int32
    assert_type(xydata, XYData[np.int32])


def test___mismatched_dtypes___create___raises_type_error() -> None:
    data = np.array([1, 2, 3, 4, 5], np.int32)
    data2 = np.array([1, 2, 3, 4, 5], np.float64)

    with pytest.raises(TypeError) as exc:
        _ = XYData(data, data2)

    assert exc.value.args[0].startswith("x_values and y_values must have the same type.")


# TODO: Fix this test to pass in arrays of the invalid types.
# @pytest.mark.parametrize("dtype", [np.complex128, np.str_, np.void, "i2, i2"])
# def test___unsupported_dtype___create___raises_type_error(
#     dtype: npt.DTypeLike,
# ) -> None:
#     data = np.array([1, 2, 3, 4, 5], dtype)
#     with pytest.raises(TypeError) as exc:
#         _ = XYData(data, data)

#     assert exc.value.args[0].startswith("The requested data type is not supported.")


###############################################################################
# from_arrays_1d
###############################################################################
def test___float64_ndarray___from_array_1d___creates_spectrum_with_float64_dtype() -> None:
    data = np.array([1.1, 2.2, 3.3, 4.4, 5.5], np.float64)

    xydata = XYData.from_arrays_1d(data, data)

    assert xydata.x_data.tolist() == data.tolist()
    assert xydata.y_data.tolist() == data.tolist()
    assert xydata.dtype == np.float64
    assert_type(xydata, XYData[np.float64])


def test___int32_ndarray___from_array_1d___creates_spectrum_with_int32_dtype() -> None:
    data = np.array([1, 2, 3, 4, 5], np.int32)

    xydata = XYData.from_arrays_1d(data, data)

    assert xydata.x_data.tolist() == data.tolist()
    assert xydata.y_data.tolist() == data.tolist()
    assert xydata.dtype == np.int32
    assert_type(xydata, XYData[np.int32])


def test___int32_array_with_dtype___from_array_1d___creates_spectrum_with_specified_dtype() -> None:
    data = array.array("i", [1, 2, 3, 4, 5])

    xydata = XYData.from_arrays_1d(data, data, np.int32)

    assert xydata.x_data.tolist() == data.tolist()
    assert xydata.y_data.tolist() == data.tolist()
    assert xydata.dtype == np.int32
    assert_type(xydata, XYData[np.int32])


def test___int16_ndarray_with_mismatched_dtype___from_array_1d___creates_spectrum_with_specified_dtype() -> (
    None
):
    data = np.array([1, 2, 3, 4, 5], np.int16)

    xydata = XYData.from_arrays_1d(data, data, np.int32)

    assert xydata.x_data.tolist() == data.tolist()
    assert xydata.y_data.tolist() == data.tolist()
    assert xydata.dtype == np.int32
    assert_type(xydata, XYData[np.int32])


def test___int_list_with_dtype___from_array_1d___creates_spectrum_with_specified_dtype() -> None:
    data = [1, 2, 3, 4, 5]

    xydata = XYData.from_arrays_1d(data, data, np.int32)

    assert xydata.x_data.tolist() == data
    assert xydata.y_data.tolist() == data
    assert xydata.dtype == np.int32
    assert_type(xydata, XYData[np.int32])


def test___int_list_with_dtype_str___from_array_1d___creates_spectrum_with_specified_dtype() -> (
    None
):
    data = [1, 2, 3, 4, 5]

    xydata = XYData.from_arrays_1d(data, data, "int32")

    assert xydata.x_data.tolist() == data
    assert xydata.y_data.tolist() == data
    assert xydata.dtype == np.int32
    assert_type(xydata, XYData[Any])  # dtype not inferred from string


def test___int32_ndarray_2d___from_array_1d___raises_value_error() -> None:
    data = np.array([[1, 2, 3], [4, 5, 6]], np.int32)

    with pytest.raises(ValueError) as exc:
        _ = XYData.from_arrays_1d(data, data)

    assert exc.value.args[0].startswith(
        "The input array must be a one-dimensional array or sequence."
    )


def test___int_list_without_dtype___from_array_1d___raises_value_error() -> None:
    data = [1, 2, 3, 4, 5]

    with pytest.raises(ValueError) as exc:
        _ = XYData.from_arrays_1d(data, data)

    assert exc.value.args[0].startswith(
        "You must specify a dtype when the input array is a sequence."
    )


def test___bytes___from_array_1d___raises_value_error() -> None:
    data = b"\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00"

    with pytest.raises(ValueError) as exc:
        _ = XYData.from_arrays_1d(data, data, np.int32)

    assert exc.value.args[0].startswith("invalid literal for int() with base 10:")


def test___iterable___from_array_1d___raises_type_error() -> None:
    data = itertools.repeat(3)

    with pytest.raises(TypeError) as exc:
        _ = XYData.from_arrays_1d(data, data, np.int32)  # type: ignore[call-overload]

    assert exc.value.args[0].startswith(
        "The input array must be a one-dimensional array or sequence."
    )


def test___ndarray_with_unsupported_dtype___from_array_1d___raises_type_error() -> None:
    data = np.zeros(3, np.str_)

    with pytest.raises(TypeError) as exc:
        _ = XYData.from_arrays_1d(data, data)

    assert exc.value.args[0].startswith("The requested data type is not supported.")


def test___copy___from_array_1d___creates_spectrum_linked_to_different_buffer() -> None:
    x_data = np.array([1, 2, 3, 4, 5], np.int32)
    y_data = np.array([6, 7, 8, 9, 10], np.int32)

    xydata = XYData.from_arrays_1d(x_data, y_data, copy=True)

    assert xydata.x_data is not x_data
    assert xydata.x_data.tolist() == x_data.tolist()
    x_data[:] = [5, 4, 3, 2, 1]
    assert xydata.x_data.tolist() != x_data.tolist()

    assert xydata.y_data is not y_data
    assert xydata.y_data.tolist() == y_data.tolist()
    y_data[:] = [5, 4, 3, 2, 1]
    assert xydata.y_data.tolist() != y_data.tolist()


def test___int32_ndarray_no_copy___from_array_1d___creates_spectrum_linked_to_same_buffer() -> None:
    x_data = np.array([1, 2, 3, 4, 5], np.int32)
    y_data = np.array([6, 7, 8, 9, 10], np.int32)

    xydata = XYData.from_arrays_1d(x_data, y_data, copy=False)

    assert xydata._x_data is x_data
    assert xydata.x_data.tolist() == x_data.tolist()
    x_data[:] = [5, 4, 3, 2, 1]
    assert xydata.x_data.tolist() == x_data.tolist()

    assert xydata._y_data is y_data
    assert xydata.y_data.tolist() == y_data.tolist()
    y_data[:] = [5, 4, 3, 2, 1]
    assert xydata.y_data.tolist() == y_data.tolist()


def test___int_list_no_copy___from_array_1d___raises_value_error() -> None:
    x_data = [1, 2, 3, 4, 5]
    y_data = [6, 7, 8, 9, 10]

    with pytest.raises(ValueError) as exc:
        _ = XYData.from_arrays_1d(x_data, y_data, np.int32, copy=False)

    assert exc.value.args[0].startswith(
        "Unable to avoid copy while creating an array as requested."
    )


###############################################################################
# compare
###############################################################################
@pytest.mark.parametrize(
    "left, right",
    [
        (
            XYData.from_arrays_1d([1, 2], [3, 4], np.int32),
            XYData.from_arrays_1d([1, 2], [3, 4], np.int32),
        ),
        (
            XYData.from_arrays_1d([1.0, 2.0], [3.0, 4.0], np.int32),
            XYData.from_arrays_1d([1.0, 2.0], [3.0, 4.0], np.int32),
        ),
    ],
)
def test___same_value___comparison___equal(left: XYData[_TData], right: XYData[_TData]) -> None:
    assert left == right


@pytest.mark.parametrize(
    "left, right",
    [
        (
            XYData.from_arrays_1d([1, 2], [5, 6], np.int32),
            XYData.from_arrays_1d([1, 2], [3, 4], np.int32),
        ),
        (
            XYData.from_arrays_1d([1.0, 2.0], [5.0, 6.0], np.int32),
            XYData.from_arrays_1d([1.0, 2.0], [3.0, 4.0], np.int32),
        ),
    ],
)
def test___different_values___comparison___not_equal(
    left: XYData[_TData], right: XYData[_TData]
) -> None:
    assert left != right


def test___different_units___comparison___not_equal() -> None:
    left = XYData.from_arrays_1d([0], [0], np.int32, x_units="volts", y_units="seconds")
    right = XYData.from_arrays_1d([0], [0], np.int32, x_units="amps", y_units="seconds")

    assert left != right


###############################################################################
# other operators
###############################################################################
@pytest.mark.parametrize(
    "value, expected_repr",
    [
        (
            XYData.from_arrays_1d([10], [20], np.int32),
            "nitypes.xy_data.XYData(x_data=array([10], dtype=int32), y_data=array([20], dtype=int32), x_units='', y_units='')",
        ),
        (
            XYData.from_arrays_1d([1.0, 1.1], [1.2, 1.3], np.float64),
            "nitypes.xy_data.XYData(x_data=array([1. , 1.1]), y_data=array([1.2, 1.3]), x_units='', y_units='')",
        ),
        (
            XYData.from_arrays_1d([10], [20], np.int32, x_units="volts", y_units="s"),
            "nitypes.xy_data.XYData(x_data=array([10], dtype=int32), y_data=array([20], dtype=int32), x_units='volts', y_units='s')",
        ),
    ],
)
def test___various_values___repr___looks_ok(value: XYData[Any], expected_repr: str) -> None:
    assert repr(value) == expected_repr


@pytest.mark.parametrize(
    "value, expected_str",
    [
        (
            XYData.from_arrays_1d([], [], np.int32),
            "[[], []]",
        ),
        (
            XYData.from_arrays_1d([], [], np.int32, x_units="volts", y_units="s"),
            "[[], []]",
        ),
        (
            XYData.from_arrays_1d([10, 20], [30, 40], np.int32),
            "[[10, 20], [30, 40]]",
        ),
        (
            XYData.from_arrays_1d([10.0, 20.0], [30.0, 40.0], np.float64),
            "[[10.0, 20.0], [30.0, 40.0]]",
        ),
        (
            XYData.from_arrays_1d([10], [20], np.int32, x_units="volts", y_units="s"),
            "[[10 volts], [20 s]]",
        ),
        (
            XYData.from_arrays_1d([1, 2], [3, 4], np.int32, x_units="miles", y_units="hr"),
            "[[1 miles, 2 miles], [3 hr, 4 hr]]",
        ),
    ],
)
def test___various_values___str___looks_ok(value: XYData[Any], expected_str: str) -> None:
    assert str(value) == expected_str


###############################################################################
# other properties
###############################################################################
def test___xy_data_with_units___get_extended_properties___returns_correct_dictionary() -> None:
    value = XYData.from_arrays_1d([20.0], [40.0], np.float64, x_units="watts", y_units="hr")

    prop_dict = value.extended_properties

    assert isinstance(prop_dict, ExtendedPropertyDictionary)
    assert prop_dict.get(_UNIT_DESCRIPTION_X) == "watts"
    assert prop_dict.get(_UNIT_DESCRIPTION_Y) == "hr"


def test___xy_data_with_units___set_units___units_updated_correctly() -> None:
    value = XYData.from_arrays_1d([20.0], [40.0], np.float64, x_units="watts", y_units="hr")

    value.x_units = "volts"
    value.y_units = "s"

    assert value.x_units == "volts"
    assert value.y_units == "s"


@pytest.mark.parametrize(
    "value",
    [
        XYData.from_arrays_1d([10, 20], [30, 40], np.int32),
        XYData.from_arrays_1d([20.0, 20.1], [20.3, 20.4], np.float64),
        XYData.from_arrays_1d([10, 20], [30, 40], np.int32, x_units="A", y_units="B"),
        XYData.from_arrays_1d([20.0, 20.1], [20.3, 20.4], np.float64, x_units="C", y_units="D"),
    ],
)
def test___various_values___copy___makes_copy(value: XYData[_TData]) -> None:
    new_value = copy.copy(value)
    assert new_value is not value
    assert new_value == value


@pytest.mark.parametrize(
    "value",
    [
        XYData.from_arrays_1d([10, 20], [30, 40], np.int32),
        XYData.from_arrays_1d([20.0, 20.1], [20.3, 20.4], np.float64),
        XYData.from_arrays_1d([10, 20], [30, 40], np.int32, x_units="A", y_units="B"),
        XYData.from_arrays_1d([20.0, 20.1], [20.3, 20.4], np.float64, x_units="C", y_units="D"),
    ],
)
def test___various_values___pickle_unpickle___makes_copy(value: XYData[_TData]) -> None:
    new_value = pickle.loads(pickle.dumps(value))
    assert new_value is not value
    assert new_value == value


def test___xy_data___pickle___references_public_modules() -> None:
    value = XYData.from_arrays_1d([10, 20], [30, 40], np.int32)
    value_bytes = pickle.dumps(value)

    assert b"nitypes.xy_data" in value_bytes
    assert b"nitypes.xy_data._xy_data" not in value_bytes


def test___various_units_values___change_units___updates_units_correctly() -> None:
    data = XYData.from_arrays_1d([1], [2], np.int32)

    # Because x and y units are stored as a single string in the ExtendedPropertiesDictionary,
    # I want to test an assortment of unit assignments (blank strings, order, etc.) to make sure
    # the code that updates/reads the units from the single string is correct.
    data.x_units = "volts"

    assert data.x_units == "volts"
    assert data.y_units == ""

    data.y_units = "seconds"

    assert data.x_units == "volts"
    assert data.y_units == "seconds"

    data.x_units = ""
    data.y_units = "hours"

    assert data.x_units == ""
    assert data.y_units == "hours"

    data.y_units = ""

    assert data.x_units == data.y_units == ""

    data.y_units = "A"
    data.x_units = "B"

    assert data.x_units == "B"
    assert data.y_units == "A"
