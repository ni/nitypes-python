from __future__ import annotations

import copy
import pickle
from typing import Any

import pytest
from typing_extensions import assert_type

from nitypes.vector import Vector
from nitypes.vector._vector import _VectorType
from nitypes.waveform._extended_properties import (
    UNIT_DESCRIPTION,
    ExtendedPropertyDictionary,
)


###############################################################################
# create
###############################################################################
def test___bool_data_values___create___creates_with_bool_data_and_default_units() -> None:
    data = Vector([True, False, True])

    assert_type(data.values[0], bool)
    assert data.values == [True, False, True]
    assert data.units == ""


def test___int_data_values___create___creates_with_int_data_and_default_units() -> None:
    data = Vector([10, 20, 30])

    assert_type(data.values[0], int)
    assert data.values == [10, 20, 30]
    assert data.units == ""


def test___float_data_value___create___creates_scalar_data_with_data_and_default_units() -> None:
    data = Vector([20.2, 30.3, 40.4])

    assert_type(data.values[0], float)
    assert data.values == [20.2, 30.3, 40.4]
    assert data.units == ""


def test___str_data_value___create___creates_scalar_data_with_data_and_default_units() -> None:
    data = Vector(["one", "two"])

    assert_type(data.values[0], str)
    assert data.values == ["one", "two"]
    assert data.units == ""


@pytest.mark.parametrize("data_value", [True, 10, 20.0, "value"])
@pytest.mark.parametrize("units", ["volts"])
def test___data_value_and_units___create___creates_scalar_data_with_data_and_units(
    data_value: Any, units: str
) -> None:
    expected_data = [data_value] * 5
    data = Vector(expected_data, units)

    assert data.values == expected_data
    assert data.units == units


@pytest.mark.parametrize("data_value", [[[1.0, 2.0]], [{"key", "value"}]])
def test___invalid_data_value___create___raises_type_error(data_value: Any) -> None:
    with pytest.raises(TypeError) as exc:
        _ = Vector(data_value)

    assert exc.value.args[0].startswith("The vector input data must be a bool, int, float, or str.")


###############################################################################
# append
###############################################################################
def test___vector_with_data___append_same_type___values_appended() -> None:
    vector = Vector([1, 2, 3], "volts")

    vector.append(4)

    assert vector.values == [1, 2, 3, 4]


def test___vector_with_data___append_different_type___raises_type_error() -> None:
    vector = Vector([1.0, 2.0, 3.0], "volts")

    with pytest.raises(TypeError) as exc:
        vector.append(True)

    assert exc.value.args[0].startswith(
        "The datatype of the appended value must match the type of the existing vector values"
    )


###############################################################################
# extend
###############################################################################
def test___vector_with_data___extend_same_type___values_extended() -> None:
    vector = Vector([1, 2, 3], "volts")

    vector.extend([4, 5])

    assert vector.values == [1, 2, 3, 4, 5]


def test___vector_with_data___extend_different_type___raises_type_error() -> None:
    vector = Vector([1.0, 2.0, 3.0], "volts")

    with pytest.raises(TypeError) as exc:
        vector.extend([True, False])

    assert exc.value.args[0].startswith(
        "The datatype of the extended values must match the type of the existing vector values"
    )


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
    left: Vector[_VectorType], right: Vector[_VectorType]
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
    left: Vector[_VectorType], right: Vector[_VectorType]
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
def test___various_values___copy___makes_copy(value: Vector[_VectorType]) -> None:
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
def test___various_values___pickle_unpickle___makes_copy(value: Vector[_VectorType]) -> None:
    new_value = pickle.loads(pickle.dumps(value))
    assert new_value is not value
    assert new_value == value


def test___vector___pickle___references_public_modules() -> None:
    value = Vector([123, 234])
    value_bytes = pickle.dumps(value)

    assert b"nitypes.vector" in value_bytes
    assert b"nitypes.vector._vector" not in value_bytes
