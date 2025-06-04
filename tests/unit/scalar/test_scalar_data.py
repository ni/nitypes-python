from __future__ import annotations

from typing import Any

import pytest

from nitypes.scalar import ScalarData
from nitypes.scalar._scalar_data import _ScalarType


###############################################################################
# create
###############################################################################
@pytest.mark.parametrize("data_value", [True, 10, 20.0, "value"])
def test___data_value___create___creates_scalar_data_with_data_and_default_units(
    data_value: Any,
) -> None:
    data = ScalarData(data_value)

    assert data.value == data_value
    assert data.units == ""


@pytest.mark.parametrize("data_value", [True, 10, 20.0, "value"])
@pytest.mark.parametrize("units", ["volts"])
def test___data_value_and_units___create___creates_scalar_data_with_data_and_units(
    data_value: Any, units: str
) -> None:
    data = ScalarData(data_value, units)

    assert data.value == data_value
    assert data.units == units


@pytest.mark.parametrize("data_value", [[1.0, 2.0], {"key", "value"}])
def test___invalid_data_value___create___raises_type_error(data_value: Any) -> None:
    with pytest.raises(TypeError) as exc:
        _ = ScalarData(data_value)

    assert exc.value.args[0].startswith("The scalar input data must be a bool, int, float, or str.")


###############################################################################
# compare
###############################################################################
@pytest.mark.parametrize(
    "left, right",
    [
        (ScalarData(False), ScalarData(False)),
        (ScalarData(1), ScalarData(1)),
        (ScalarData(10.0), ScalarData(10.0)),
        (ScalarData("value"), ScalarData("value")),
    ],
)
def test___same_value___comparison___equal(
    left: ScalarData[_ScalarType], right: ScalarData[_ScalarType]
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
        (ScalarData(False), ScalarData(True)),
        (ScalarData(0), ScalarData(1)),
        (ScalarData(10.0), ScalarData(20.0)),
        (ScalarData("aaa"), ScalarData("zzz")),
    ],
)
def test___lesser_value___comparison___lesser(
    left: ScalarData[_ScalarType], right: ScalarData[_ScalarType]
) -> None:
    assert left < right
    assert left <= right
    assert not (left == right)
    assert left != right
    assert not (left > right)
    assert not (left >= right)


def test___different_units___comparison___throws_exception() -> None:
    left = ScalarData(0, "volts")
    right = ScalarData(0, "amps")
    expected_message = "Comparing ScalarData objects with different units is not permitted."

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
