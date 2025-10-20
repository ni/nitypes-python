from __future__ import annotations

import copy
import pickle
from typing import Any

import pytest
from typing_extensions import assert_type

from nitypes.scalar import Scalar, TScalar_co
from nitypes.waveform._extended_properties import (
    UNIT_DESCRIPTION,
    ExtendedPropertyDictionary,
)


###############################################################################
# create
###############################################################################
def test___bool_data_value___create___creates_scalar_data_with_data_and_default_units() -> None:
    data = Scalar(True)

    assert_type(data.value, bool)
    assert data.value is True
    assert data.units == ""


def test___int_data_value___create___creates_scalar_data_with_data_and_default_units() -> None:
    data = Scalar(10)

    assert_type(data.value, int)
    assert data.value == 10
    assert data.units == ""


def test___float_data_value___create___creates_scalar_data_with_data_and_default_units() -> None:
    data = Scalar(20.0)

    assert_type(data.value, float)
    assert data.value == 20.0
    assert data.units == ""


def test___str_data_value___create___creates_scalar_data_with_data_and_default_units() -> None:
    data = Scalar("value")

    assert_type(data.value, str)
    assert data.value == "value"
    assert data.units == ""


@pytest.mark.parametrize("data_value", [True, 10, 20.0, "value"])
@pytest.mark.parametrize("units", ["volts"])
def test___data_value_and_units___create___creates_scalar_data_with_data_and_units(
    data_value: Any, units: str
) -> None:
    data = Scalar(data_value, units)

    assert data.value == data_value
    assert data.units == units


@pytest.mark.parametrize("data_value", [[1.0, 2.0], {"key", "value"}])
def test___invalid_data_value___create___raises_type_error(data_value: Any) -> None:
    with pytest.raises(TypeError) as exc:
        _ = Scalar(data_value)

    assert exc.value.args[0].startswith("The scalar input data must be a bool, int, float, or str.")


###############################################################################
# compare
###############################################################################
@pytest.mark.parametrize(
    "left, right",
    [
        (Scalar(False), Scalar(False)),
        (Scalar(1), Scalar(1)),
        (Scalar(10.0), Scalar(10.0)),
        (Scalar("value"), Scalar("value")),
    ],
)
def test___same_value___comparison___equal(
    left: Scalar[TScalar_co], right: Scalar[TScalar_co]
) -> None:
    assert not (left < right)
    assert left <= right
    assert left == right
    assert not (left != right)
    assert not (left > right)
    assert left >= right


@pytest.mark.parametrize(
    "left, right",
    [
        (Scalar(False), Scalar(True)),
        (Scalar(0), Scalar(1)),
        (Scalar(10.0), Scalar(20.0)),
        (Scalar("aaa"), Scalar("zzz")),
    ],
)
def test___lesser_value___comparison___lesser(
    left: Scalar[TScalar_co], right: Scalar[TScalar_co]
) -> None:
    assert left < right
    assert left <= right
    assert not (left == right)
    assert left != right
    assert not (left > right)
    assert not (left >= right)


@pytest.mark.parametrize(
    "left, right",
    [
        (Scalar(False), Scalar(10)),
        (Scalar(False), Scalar(10.0)),
        (Scalar(0), Scalar(1.0)),
    ],
)
def test___mixed_numeric_types___comparison___lesser(
    left: Scalar[TScalar_co], right: Scalar[TScalar_co]
) -> None:
    assert left < right
    assert left <= right
    assert not (left > right)
    assert not (left >= right)


@pytest.mark.parametrize(
    "left, right",
    [
        (Scalar(False), Scalar("value")),
        (Scalar(10), Scalar("value")),
        (Scalar(10.0), Scalar("value")),
    ],
)
def test___numeric_and_string___comparison___throws_exception(
    left: Scalar[TScalar_co], right: Scalar[TScalar_co]
) -> None:
    expected_message = "Comparing Scalar objects of numeric and string types is not permitted"

    with pytest.raises(TypeError) as exc1:
        _ = left < right
    with pytest.raises(TypeError) as exc2:
        _ = left <= right
    with pytest.raises(TypeError) as exc3:
        _ = left > right
    with pytest.raises(TypeError) as exc4:
        _ = left >= right

    assert exc1.value.args[0].startswith(expected_message)
    assert exc2.value.args[0].startswith(expected_message)
    assert exc3.value.args[0].startswith(expected_message)
    assert exc4.value.args[0].startswith(expected_message)


def test___different_units___comparison___throws_exception() -> None:
    left = Scalar(0, "volts")
    right = Scalar(0, "amps")
    expected_message = "Comparing Scalar objects with different units is not permitted."

    with pytest.raises(ValueError) as exc1:
        _ = left < right
    with pytest.raises(ValueError) as exc2:
        _ = left <= right
    with pytest.raises(ValueError) as exc3:
        _ = left > right
    with pytest.raises(ValueError) as exc4:
        _ = left >= right

    assert exc1.value.args[0].startswith(expected_message)
    assert exc2.value.args[0].startswith(expected_message)
    assert exc3.value.args[0].startswith(expected_message)
    assert exc4.value.args[0].startswith(expected_message)


###############################################################################
# other operators
###############################################################################
@pytest.mark.parametrize(
    "value, expected_repr",
    [
        (
            Scalar(False),
            "nitypes.scalar.Scalar(value=False, extended_properties="
            "nitypes.waveform.ExtendedPropertyDictionary({'NI_UnitDescription': ''}))",
        ),
        (
            Scalar(10),
            "nitypes.scalar.Scalar(value=10, extended_properties="
            "nitypes.waveform.ExtendedPropertyDictionary({'NI_UnitDescription': ''}))",
        ),
        (
            Scalar(20.0),
            "nitypes.scalar.Scalar(value=20.0, extended_properties="
            "nitypes.waveform.ExtendedPropertyDictionary({'NI_UnitDescription': ''}))",
        ),
        (
            Scalar("value"),
            "nitypes.scalar.Scalar(value='value', extended_properties="
            "nitypes.waveform.ExtendedPropertyDictionary({'NI_UnitDescription': ''}))",
        ),
        (
            Scalar(False, "amps"),
            "nitypes.scalar.Scalar(value=False, extended_properties="
            "nitypes.waveform.ExtendedPropertyDictionary({'NI_UnitDescription': 'amps'}))",
        ),
        (
            Scalar(10, "volts"),
            "nitypes.scalar.Scalar(value=10, extended_properties="
            "nitypes.waveform.ExtendedPropertyDictionary({'NI_UnitDescription': 'volts'}))",
        ),
        (
            Scalar(20.0, "watts"),
            "nitypes.scalar.Scalar(value=20.0, extended_properties="
            "nitypes.waveform.ExtendedPropertyDictionary({'NI_UnitDescription': 'watts'}))",
        ),
        (
            Scalar("value", ""),
            "nitypes.scalar.Scalar(value='value', ""extended_properties="
            "nitypes.waveform.ExtendedPropertyDictionary({'NI_UnitDescription': ''}))",
        ),
    ],
)
def test___various_values___repr___looks_ok(value: Scalar[Any], expected_repr: str) -> None:
    assert repr(value) == expected_repr


@pytest.mark.parametrize(
    "value, expected_str",
    [
        (Scalar(False), "False"),
        (Scalar(10), "10"),
        (Scalar(20.0), "20.0"),
        (Scalar("value"), "value"),
        (Scalar(False, "amps"), "False amps"),
        (Scalar(10, "volts"), "10 volts"),
        (Scalar(20.0, "watts"), "20.0 watts"),
        (Scalar("value", ""), "value"),
    ],
)
def test___various_values___str___looks_ok(value: Scalar[Any], expected_str: str) -> None:
    assert str(value) == expected_str


###############################################################################
# other properties
###############################################################################
def test___scalar_with_units___get_extended_properties___returns_correct_dictionary() -> None:
    value = Scalar(20.0, "watts")

    prop_dict = value.extended_properties

    assert isinstance(prop_dict, ExtendedPropertyDictionary)
    assert prop_dict.get(UNIT_DESCRIPTION) == "watts"


def test___scalar_with_units___set_units___units_updated_correctly() -> None:
    value = Scalar(20.0, "watts")

    value.units = "volts"

    assert value.units == "volts"


@pytest.mark.parametrize(
    "value",
    [
        Scalar(False),
        Scalar(10),
        Scalar(20.0),
        Scalar("value"),
        Scalar(False, "amps"),
        Scalar(10, "volts"),
        Scalar(20.0, "watts"),
        Scalar("value", ""),
        Scalar(10, "Volts", extended_properties={"one": 1})
    ],
)
def test___various_values___copy___makes_copy(value: Scalar[TScalar_co]) -> None:
    new_value = copy.copy(value)
    assert new_value is not value
    assert new_value == value
    assert new_value.extended_properties == value.extended_properties


@pytest.mark.parametrize(
    "value",
    [
        Scalar(False),
        Scalar(10),
        Scalar(20.0),
        Scalar("value"),
        Scalar(False, "amps"),
        Scalar(10, "volts"),
        Scalar(20.0, "watts"),
        Scalar("value", ""),
        Scalar(10, "Volts", extended_properties={"one": 1})
    ],
)
def test___various_values___pickle_unpickle___makes_copy(value: Scalar[TScalar_co]) -> None:
    new_value = pickle.loads(pickle.dumps(value))
    assert isinstance(new_value, Scalar)
    assert new_value is not value
    assert new_value == value
    assert new_value.extended_properties == value.extended_properties


def test___scalar___pickle___references_public_modules() -> None:
    value = Scalar(123)
    value_bytes = pickle.dumps(value)

    assert b"nitypes.scalar" in value_bytes
    assert b"nitypes.scalar._scalar" not in value_bytes
