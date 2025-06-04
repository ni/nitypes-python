from __future__ import annotations

from typing import Any

import pytest

from nitypes.scalar import ScalarData


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
def test__identical_scalar_datas_default_units__compare__returns_equal() -> None:
    data1 = ScalarData(5)
    data2 = ScalarData(5)
    assert data1 == data2


def test__identical_scalar_datas_custom_units__compare__returns_equal() -> None:
    data1 = ScalarData(5, "amps")
    data2 = ScalarData(5, "amps")
    assert data1 == data2


def test__different_scalar_data_values__compare__returns_not_equal() -> None:
    data1 = ScalarData(5)
    data2 = ScalarData(10)
    assert data1 != data2


def test__different_scalar_data_units__compare__returns_not_equal() -> None:
    data1 = ScalarData(5, "volts")
    data2 = ScalarData(5, "amps")
    assert data1 != data2
