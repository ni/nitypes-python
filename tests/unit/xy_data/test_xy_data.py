from __future__ import annotations

import copy
import pickle
from typing import Any

import pytest
from typing_extensions import assert_type

from nitypes.waveform._extended_properties import (
    UNIT_DESCRIPTION,
    ExtendedPropertyDictionary,
)
from nitypes.xy_data import TNumeric, XYData


###############################################################################
# create
###############################################################################
def test___no_data_values_no_type___create___raises_type_error() -> None:
    with pytest.raises(TypeError) as exc:
        _ = XYData([], [])

    assert exc.value.args[0].startswith(
        "You must specify x_values and y_values as non-empty or specify value_type."
    )


def test___no_data_values_int_type___create___creates_with_int_type() -> None:
    data = XYData([], [], value_type=int)

    assert_type(data._values, list[list[int]])
    assert data.x_data == []
    assert data.y_data == []
    assert data.x_units == data.y_units == ""
    assert data._value_type == int


def test___int_data_values___create___creates_with_int_data_and_default_units() -> None:
    data = XYData([10, 20, 30], [40, 50, 60])

    assert_type(data._values[0][0], int)
    assert data.x_data == [10, 20, 30]
    assert data.y_data == [40, 50, 60]
    assert data.x_units == data.y_units == ""


def test___float_data_value___create___creates_with_float_data_and_default_units() -> None:
    data = XYData([20.2, 30.3, 40.4], [50.5, 60.6, 70.7])

    assert_type(data._values[0][0], float)
    assert data.x_data == [20.2, 30.3, 40.4]
    assert data.y_data == [50.5, 60.6, 70.7]
    assert data.x_units == data.y_units == ""


def test___float_data_value_and_units___create___creates_with_float_data_and_units() -> None:
    expected_x_data = [1.1, 2.2, 3.3]
    expected_x_units = "volts"
    expected_y_data = [4.4, 5.5, 6.6]
    expected_y_units = "seconds"

    data = XYData(expected_x_data, expected_y_data, expected_x_units, expected_y_units)

    assert data.x_data == expected_x_data
    assert data.y_data == expected_y_data
    assert data.x_units == expected_x_units
    assert data.y_units == expected_y_units


@pytest.mark.parametrize("data_value", [[[1.0, 2.0]], [{"key", "value"}]])
def test___invalid_data_value___create___raises_type_error(data_value: Any) -> None:
    with pytest.raises(TypeError) as exc:
        _ = XYData(data_value, data_value)

    assert exc.value.args[0].startswith("The XYData input data must be an int or float.")


def test___mixed_data_values___create___raises_type_error() -> None:
    with pytest.raises(TypeError) as exc:
        mixed_data = [1.0, 2, 3]
        _ = XYData(mixed_data, mixed_data)

    assert exc.value.args[0].startswith("Input data does not match expected type.")


def test___int_data_values_length_mismatch___create___raises_value_error() -> None:
    with pytest.raises(ValueError) as exc:
        _ = XYData([1, 2], [3, 4, 5])

    assert exc.value.args[0].startswith("x_values and y_values must be the same length.")


###############################################################################
# append
###############################################################################
def test___xy_data___append_same_type___values_appended() -> None:
    data = XYData([1, 2, 3], [7, 6, 5])

    data.append(4, 4)

    assert data.x_data == [1, 2, 3, 4]
    assert data.y_data == [7, 6, 5, 4]


def test___xy_data___append_multiple_times___values_appended() -> None:
    data = XYData([1, 2, 3], [7, 6, 5])

    data.append(4, 4)
    data.append(20, 21)

    assert data.x_data == [1, 2, 3, 4, 20]
    assert data.y_data == [7, 6, 5, 4, 21]


def test___xy_data___append_different_type___raises_type_error() -> None:
    data = XYData([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])

    with pytest.raises(TypeError) as exc:
        data.append(True, False)

    assert exc.value.args[0].startswith("Input data does not match expected type.")


def test___empty_xy_data___append___appends_data() -> None:
    data = XYData([], [], value_type=int)
    data.append(1, 2)

    assert data.x_data == [1]
    assert data.y_data == [2]


###############################################################################
# extend
###############################################################################
def test___xy_data___extend_same_type___values_extended() -> None:
    data = XYData([1, 2, 3], [4, 5, 6])

    data.extend([4, 5], [7, 8])

    assert data.x_data == [1, 2, 3, 4, 5]
    assert data.y_data == [4, 5, 6, 7, 8]


def test___xy_data___extend_multiple_times___values_extended() -> None:
    data = XYData([1, 2, 3], [4, 5, 6])

    data.extend([4, 5], [7, 8])
    data.extend([10], [11])

    assert data.x_data == [1, 2, 3, 4, 5, 10]
    assert data.y_data == [4, 5, 6, 7, 8, 11]


def test___xy_data___extend_different_type___raises_type_error() -> None:
    data = XYData([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])

    with pytest.raises(TypeError) as exc:
        data.extend([1, 2], [3, 4])

    assert exc.value.args[0].startswith("Input data does not match expected type.")


def test___xy_data___extend_mixed_lengths___raises_value_error() -> None:
    data = XYData([1, 2, 3], [4, 5, 6])

    with pytest.raises(ValueError) as exc:
        data.extend([1, 2, 3], [4, 5])

    assert exc.value.args[0].startswith("X and Y sequences to extend must be the same length.")


def test___empty_xy_data___extend___values_extended() -> None:
    data = XYData([], [], value_type=float)
    data.extend([1.0, 2.0], [3.0, 4.0])

    assert data.x_data == [1.0, 2.0]
    assert data.y_data == [3.0, 4.0]


def test___xy_data___extend_with_empty_list___values_unchanged() -> None:
    data = XYData([1, 2, 3], [4, 5, 6])

    data.extend([], [])

    assert data.x_data == [1, 2, 3]
    assert data.y_data == [4, 5, 6]


###############################################################################
# length
###############################################################################
def test___xy_data___check_length___length_correct() -> None:
    xy_data = XYData([1, 2], [3, 4])

    assert len(xy_data) == 2


###############################################################################
# compare
###############################################################################
@pytest.mark.parametrize(
    "left, right",
    [
        (XYData([1, 2], [3, 4]), XYData([1, 2], [3, 4])),
        (XYData([1.0, 2.0], [3.0, 4.0]), XYData([1.0, 2.0], [3.0, 4.0])),
    ],
)
def test___same_value___comparison___equal(left: XYData[TNumeric], right: XYData[TNumeric]) -> None:
    assert left == right


@pytest.mark.parametrize(
    "left, right",
    [
        (XYData([1, 2], [5, 6]), XYData([1, 2], [3, 4])),
        (XYData([1.0, 2.0], [5.0, 6.0]), XYData([1.0, 2.0], [3.0, 4.0])),
    ],
)
def test___different_values___comparison___not_equal(
    left: XYData[TNumeric], right: XYData[TNumeric]
) -> None:
    assert left != right


def test___different_units___comparison___not_equal() -> None:
    left = XYData([0], [0], "volts", "seconds")
    right = XYData([0], [0], "amps", "seconds")

    assert left != right


###############################################################################
# other operators
###############################################################################
@pytest.mark.parametrize(
    "value, expected_repr",
    [
        (
            XYData([10], [20]),
            "nitypes.xy_data.XYData(x_data=[10], y_data=[20], x_units='', y_units='')",
        ),
        (
            XYData([1.0, 1.1], [1.2, 1.3]),
            "nitypes.xy_data.XYData(x_data=[1.0, 1.1], y_data=[1.2, 1.3], x_units='', y_units='')",
        ),
        (
            XYData([10], [20], "volts", "s"),
            "nitypes.xy_data.XYData(x_data=[10], y_data=[20], x_units='volts', y_units='s')",
        ),
    ],
)
def test___various_values___repr___looks_ok(value: XYData[Any], expected_repr: str) -> None:
    assert repr(value) == expected_repr


@pytest.mark.parametrize(
    "value, expected_str",
    [
        (XYData([], [], value_type=int), "[[], []]"),
        (XYData([], [], "volts", "s", value_type=int), "[[], []]"),
        (XYData([10, 20], [30, 40]), "[[10, 20], [30, 40]]"),
        (XYData([10.0, 20.0], [30.0, 40.0]), "[[10.0, 20.0], [30.0, 40.0]]"),
        (XYData([10], [20], "volts", "s"), "[[10 volts], [20 s]]"),
        (XYData([1, 2], [3, 4], "miles", "hr"), "[[1 miles, 2 miles], [3 hr, 4 hr]]"),
    ],
)
def test___various_values___str___looks_ok(value: XYData[Any], expected_str: str) -> None:
    assert str(value) == expected_str


###############################################################################
# other properties
###############################################################################
def test___xy_data_with_units___get_extended_properties___returns_correct_dictionary() -> None:
    value = XYData([20.0], [40.0], "watts", "hr")

    prop_dict = value.extended_properties

    assert isinstance(prop_dict, ExtendedPropertyDictionary)
    assert prop_dict.get(UNIT_DESCRIPTION) == "watts,hr"


def test___xy_data_with_units___set_units___units_updated_correctly() -> None:
    value = XYData([20.0], [40.0], "watts", "hr")

    value.x_units = "volts"
    value.y_units = "s"

    assert value.x_units == "volts"
    assert value.y_units == "s"


@pytest.mark.parametrize(
    "value",
    [
        XYData([10, 20], [30, 40]),
        XYData([20.0, 20.1], [20.3, 20.4]),
        XYData([10, 20], [30, 40], "A", "B"),
        XYData([20.0, 20.1], [20.3, 20.4], "C", "D"),
    ],
)
def test___various_values___copy___makes_copy(value: XYData[TNumeric]) -> None:
    new_value = copy.copy(value)
    assert new_value is not value
    assert new_value == value


@pytest.mark.parametrize(
    "value",
    [
        XYData([10, 20], [30, 40]),
        XYData([20.0, 20.1], [20.3, 20.4]),
        XYData([10, 20], [30, 40], "A", "B"),
        XYData([20.0, 20.1], [20.3, 20.4], "C", "D"),
    ],
)
def test___various_values___pickle_unpickle___makes_copy(value: XYData[TNumeric]) -> None:
    new_value = pickle.loads(pickle.dumps(value))
    assert new_value is not value
    assert new_value == value


def test___xy_data___pickle___references_public_modules() -> None:
    value = XYData([10, 20], [30, 40])
    value_bytes = pickle.dumps(value)

    assert b"nitypes.xy_data" in value_bytes
    assert b"nitypes.xy_data._xy_data" not in value_bytes


def test___various_units_values___change_units___updates_units_correctly() -> None:
    data = XYData([1], [2])

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
