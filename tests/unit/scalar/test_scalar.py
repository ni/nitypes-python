from __future__ import annotations

from typing import Any

import pytest
from typing_extensions import assert_type

from nitypes.scalar import Scalar
from nitypes.scalar._scalar import _ScalarType_co


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
    left: Scalar[_ScalarType_co], right: Scalar[_ScalarType_co]
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
    left: Scalar[_ScalarType_co], right: Scalar[_ScalarType_co]
) -> None:
    assert left < right
    assert left <= right
    assert not (left == right)
    assert left != right
    assert not (left > right)
    assert not (left >= right)


def test___different_units___comparison___throws_exception() -> None:
    left = Scalar(0, "volts")
    right = Scalar(0, "amps")
    expected_message = "Comparing Scalar objects with different units is not permitted."

    with pytest.raises(ValueError) as exc:
        _ = left < right

    assert exc.value.args[0].startswith(expected_message)

    with pytest.raises(ValueError) as exc:
        _ = left <= right

    assert exc.value.args[0].startswith(expected_message)

    with pytest.raises(ValueError) as exc:
        _ = left > right

    assert exc.value.args[0].startswith(expected_message)

    with pytest.raises(ValueError) as exc:
        _ = left >= right

    assert exc.value.args[0].startswith(expected_message)


###############################################################################
# other operators
###############################################################################
def test___repr___returns_correct_string() -> None:
    data = Scalar(10, "volts")
    repr_str = data.__repr__()
    expected_str = "nitypes.scalar.Scalar(value=10, units=volts)"
    assert repr_str == expected_str


def test___no_units___str___returns_correct_string() -> None:
    data = Scalar(10)
    assert str(data) == "10"


def test___with_units___str___returns_correct_string() -> None:
    data = Scalar(10, "amps")
    assert str(data) == "10 amps"
