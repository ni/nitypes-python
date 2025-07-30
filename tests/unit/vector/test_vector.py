from __future__ import annotations

import copy
import pickle
from typing import Any

import pytest
from typing_extensions import assert_type

from nitypes.vector import Vector
from nitypes.vector import VectorType
from nitypes.waveform._extended_properties import (
    UNIT_DESCRIPTION,
    ExtendedPropertyDictionary,
)

OUT_OF_RANGE_INT = 0x7FFFFFFF


###############################################################################
# create
###############################################################################
def test___no_data_values_no_type___create___raises_type_error() -> None:
    with pytest.raises(TypeError) as exc:
        _ = Vector([])

    assert exc.value.args[0].startswith(
        "You must specify values as non-empty or specify value_type."
    )


def test___no_data_values_bool_type___create___creates_with_bool_type() -> None:
    data = Vector([], value_type=bool)

    assert_type(data._values, list[bool])
    assert data._values == []
    assert data.units == ""
    assert data._value_type == bool


def test___bool_data_values___create___creates_with_bool_data_and_default_units() -> None:
    data = Vector([True, False])

    assert_type(data._values[0], bool)
    assert data._values == [True, False]
    assert data.units == ""


def test___int_data_values___create___creates_with_int_data_and_default_units() -> None:
    data = Vector([10, 20, 30])

    assert_type(data._values[0], int)
    assert data._values == [10, 20, 30]
    assert data.units == ""


def test___float_data_value___create___creates_scalar_data_with_data_and_default_units() -> None:
    data = Vector([20.2, 30.3, 40.4])

    assert_type(data._values[0], float)
    assert data._values == [20.2, 30.3, 40.4]
    assert data.units == ""


def test___str_data_value___create___creates_scalar_data_with_data_and_default_units() -> None:
    data = Vector(["one", "two"])

    assert_type(data._values[0], str)
    assert data._values == ["one", "two"]
    assert data.units == ""


@pytest.mark.parametrize("data_value", [True, 10, 20.0, "value"])
@pytest.mark.parametrize("units", ["volts"])
def test___data_value_and_units___create___creates_scalar_data_with_data_and_units(
    data_value: Any, units: str
) -> None:
    expected_data = [data_value] * 5
    data = Vector(expected_data, units)

    assert data._values == expected_data
    assert data.units == units


@pytest.mark.parametrize("data_value", [[[1.0, 2.0]], [{"key", "value"}]])
def test___invalid_data_value___create___raises_type_error(data_value: Any) -> None:
    with pytest.raises(TypeError) as exc:
        _ = Vector(data_value)

    assert exc.value.args[0].startswith("The vector input data must be a bool, int, float, or str.")


def test___mixed_data_values___create___raises_type_error() -> None:
    with pytest.raises(TypeError) as exc:
        _ = Vector([True, "string", 1.0])

    assert exc.value.args[0].startswith("All values in the values input must be of the same type.")


###############################################################################
# get_item
###############################################################################
def test___vector_with_data___get_item_at_index___returns_correct_value() -> None:
    vector = Vector([1, 2, 3], "volts")

    assert vector[0] == 1
    assert vector[1] == 2
    assert vector[2] == 3


def test___vector_with_data___get_item_at_slice___returns_correct_values() -> None:
    vector = Vector([1, 2, 3], "volts")

    assert vector[0:2] == [1, 2]
    assert vector[1:3] == [2, 3]
    assert vector[0:] == [1, 2, 3]
    assert vector[:1] == [1]


###############################################################################
# set_item
###############################################################################
def test___vector_with_data___set_item_at_index___value_set_correctly() -> None:
    vector = Vector([1, 2, 3], "volts")

    vector[1] = 4

    assert vector._values == [1, 4, 3]


def test___vector_with_data___set_item_at_slice___values_set_correctly() -> None:
    vector = Vector([1, 2, 3], "volts")

    vector[0:2] = [6, 7]
    assert vector._values == [6, 7, 3]


def test___vector_with_int_data___set_out_of_range_int_at_index___raises_value_error() -> None:
    vector = Vector([1, 2, 3], "volts")

    with pytest.raises(ValueError) as exc:
        vector[1] = OUT_OF_RANGE_INT

    assert exc.value.args[0].startswith(
        "The integer vector value must be a within the range of Int32."
    )


def test___vector_with_int_data___set_out_of_range_int_at_slice___raises_value_error() -> None:
    vector = Vector([1, 2, 3], "volts")

    with pytest.raises(ValueError) as exc:
        vector[0:2] = [OUT_OF_RANGE_INT, 7]

    assert exc.value.args[0].startswith(
        "The integer vector value must be a within the range of Int32."
    )


###############################################################################
# append
###############################################################################
def test___vector_with_data___append_same_type___values_appended() -> None:
    vector = Vector([1, 2, 3], "volts")

    vector.append(4)

    assert vector._values == [1, 2, 3, 4]


def test___vector_with_data___append_different_type___raises_type_error() -> None:
    vector = Vector([1.0, 2.0, 3.0], "volts")

    with pytest.raises(TypeError) as exc:
        vector.append(True)

    assert exc.value.args[0].startswith("Input type does not match existing type.")


def test___vector_with_int_data___append_value_out_of_range___raises_value_error() -> None:
    vector = Vector([1, 2, 3], "volts")

    with pytest.raises(ValueError) as exc:
        vector.append(OUT_OF_RANGE_INT)

    assert exc.value.args[0].startswith(
        "The integer vector value must be a within the range of Int32."
    )


def test___no_data_values_bool_type___append___appends_bool_data() -> None:
    data = Vector([], value_type=bool)
    data.append(True)

    assert data._values == [True]


###############################################################################
# extend
###############################################################################
def test___vector_with_data___extend_same_type___values_extended() -> None:
    vector = Vector([1, 2, 3], "volts")

    vector.extend([4, 5])

    assert vector._values == [1, 2, 3, 4, 5]


def test___vector_with_data___extend_different_type___raises_type_error() -> None:
    vector = Vector([1.0, 2.0, 3.0], "volts")

    with pytest.raises(TypeError) as exc:
        vector.extend([True, False])

    assert exc.value.args[0].startswith("Input type does not match existing type.")


def test___vector_with_data___extend_mixed_type___raises_type_error() -> None:
    vector = Vector([1.0, 2.0, 3.0], "volts")

    with pytest.raises(TypeError) as exc:
        vector.extend([4.0, False, 5.0])

    assert exc.value.args[0].startswith("Input type does not match existing type.")


def test___vector_with_int_data___extend_value_out_of_range___raises_value_error() -> None:
    vector = Vector([1, 2, 3], "volts")

    with pytest.raises(ValueError) as exc:
        vector.extend([5, 6, OUT_OF_RANGE_INT])

    assert exc.value.args[0].startswith(
        "The integer vector value must be a within the range of Int32."
    )


def test___no_data_values_bool_type___extend___extends_bool_data() -> None:
    data = Vector([], value_type=bool)
    data.extend([True, False])

    assert data._values == [True, False]


###############################################################################
# delete
###############################################################################
def test___vector_with_data___delete_at_index___value_deleted() -> None:
    vector = Vector([1, 2, 3], "volts")

    del vector[1]

    assert vector._values == [1, 3]


def test___vector_with_data___delete_at_slice___values_deleted() -> None:
    vector = Vector([1, 2, 3], "volts")

    del vector[1:3]

    assert vector._values == [1]


###############################################################################
# remove
###############################################################################
def test___vector_with_data___remove_value___value_removed() -> None:
    vector = Vector([1, 2, 3], "volts")

    vector.remove(2)

    assert vector._values == [1, 3]


###############################################################################
# length
###############################################################################
def test___vector_with_data___check_length___length_correct() -> None:
    vector = Vector([1, 2, 3], "volts")

    assert len(vector) == 3


###############################################################################
# compare
###############################################################################
@pytest.mark.parametrize(
    "left, right",
    [
        (Vector([False]), Vector([False])),
        (Vector([1]), Vector([1])),
        (Vector([10.0]), Vector([10.0])),
        (Vector(["value"]), Vector(["value"])),
    ],
)
def test___same_value___comparison___equal(
    left: Vector[VectorType], right: Vector[VectorType]
) -> None:
    assert left == right


@pytest.mark.parametrize(
    "left, right",
    [
        (Vector([False]), Vector([True])),
        (Vector([1]), Vector([2])),
        (Vector([10.0]), Vector([20.0])),
        (Vector(["value"]), Vector(["other"])),
    ],
)
def test___different_values___comparison___not_equal(
    left: Vector[VectorType], right: Vector[VectorType]
) -> None:
    assert left != right


def test___different_units___comparison___not_equal() -> None:
    left = Vector([0], "volts")
    right = Vector([0], "amps")

    assert left != right


###############################################################################
# other operators
###############################################################################
@pytest.mark.parametrize(
    "value, expected_repr",
    [
        (Vector([False, True]), "nitypes.vector.Vector(values=[False, True], units='')"),
        (Vector([10, 20]), "nitypes.vector.Vector(values=[10, 20], units='')"),
        (Vector([20.0, 20.1]), "nitypes.vector.Vector(values=[20.0, 20.1], units='')"),
        (Vector(["a", "b"]), "nitypes.vector.Vector(values=['a', 'b'], units='')"),
        (Vector([False, True], "f"), "nitypes.vector.Vector(values=[False, True], units='f')"),
        (Vector([10, 20], "volts"), "nitypes.vector.Vector(values=[10, 20], units='volts')"),
        (Vector([20.0, 20.1], "w"), "nitypes.vector.Vector(values=[20.0, 20.1], units='w')"),
        (Vector(["a", "b"], ""), "nitypes.vector.Vector(values=['a', 'b'], units='')"),
    ],
)
def test___various_values___repr___looks_ok(value: Vector[Any], expected_repr: str) -> None:
    assert repr(value) == expected_repr


@pytest.mark.parametrize(
    "value, expected_str",
    [
        (Vector([False, True]), "False, True"),
        (Vector([10, 20]), "10, 20"),
        (Vector([20.0, 20.1]), "20.0, 20.1"),
        (Vector(["a", "b"]), "a, b"),
        (Vector([False, True], "amps"), "False amps, True amps"),
        (Vector([10, 20], "volts"), "10 volts, 20 volts"),
        (Vector([20.0, 20.1], "watts"), "20.0 watts, 20.1 watts"),
        (Vector(["a", "b"], ""), "a, b"),
    ],
)
def test___various_values___str___looks_ok(value: Vector[Any], expected_str: str) -> None:
    assert str(value) == expected_str


###############################################################################
# other properties
###############################################################################
def test___vector_with_units___get_extended_properties___returns_correct_dictionary() -> None:
    value = Vector([20.0], "watts")

    prop_dict = value.extended_properties

    assert isinstance(prop_dict, ExtendedPropertyDictionary)
    assert prop_dict.get(UNIT_DESCRIPTION) == "watts"


def test___vector_with_units___set_units___units_updated_correctly() -> None:
    value = Vector([20.0], "watts")

    value.units = "volts"

    assert value.units == "volts"


@pytest.mark.parametrize(
    "value",
    [
        Vector([False, True]),
        Vector([10, 20]),
        Vector([20.0, 20.1]),
        Vector(["a", "b"]),
        Vector([False, True], "amps"),
        Vector([10, 20], "volts"),
        Vector([20.0, 20.1], "watts"),
        Vector(["a", "b"], ""),
    ],
)
def test___various_values___copy___makes_copy(value: Vector[VectorType]) -> None:
    new_value = copy.copy(value)
    assert new_value is not value
    assert new_value == value


@pytest.mark.parametrize(
    "value",
    [
        Vector([False, True]),
        Vector([10, 20]),
        Vector([20.0, 20.1]),
        Vector(["a", "b"]),
        Vector([False, True], "amps"),
        Vector([10, 20], "volts"),
        Vector([20.0, 20.1], "watts"),
        Vector(["a", "b"], ""),
    ],
)
def test___various_values___pickle_unpickle___makes_copy(value: Vector[VectorType]) -> None:
    new_value = pickle.loads(pickle.dumps(value))
    assert new_value is not value
    assert new_value == value


def test___vector___pickle___references_public_modules() -> None:
    value = Vector([123, 234])
    value_bytes = pickle.dumps(value)

    assert b"nitypes.vector" in value_bytes
    assert b"nitypes.vector._vector" not in value_bytes
