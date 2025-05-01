from __future__ import annotations

import pytest

from nitypes._typing import assert_type
from nitypes.complex import ComplexInt


def test___construct___sets_fields() -> None:
    value = ComplexInt(1, 2)

    assert_type(value, ComplexInt)
    assert isinstance(value, ComplexInt)
    assert value.real == 1
    assert value.imag == 2


def test___complex() -> None:
    value = complex(ComplexInt(1, 2))

    assert_type(value, complex)
    assert isinstance(value, complex)
    assert value == complex(1, 2)


def test___add() -> None:
    value = ComplexInt(1, 2) + ComplexInt(3, 4)

    assert_type(value, ComplexInt)
    assert isinstance(value, ComplexInt)
    assert value == ComplexInt(4, 6)


def test___neg() -> None:
    value = -ComplexInt(1, 2)

    assert_type(value, ComplexInt)
    assert isinstance(value, ComplexInt)
    assert value == ComplexInt(-1, -2)


def test___pos() -> None:
    value = +ComplexInt(1, 2)

    assert_type(value, ComplexInt)
    assert isinstance(value, ComplexInt)
    assert value == ComplexInt(1, 2)


def test___sub() -> None:
    value = ComplexInt(1, 2) - ComplexInt(3, 4)

    assert_type(value, ComplexInt)
    assert isinstance(value, ComplexInt)
    assert value == ComplexInt(-2, -2)


def test___mul() -> None:
    value = ComplexInt(1, 2) * ComplexInt(3, 4)

    assert_type(value, ComplexInt)
    assert isinstance(value, ComplexInt)
    assert value == ComplexInt(-5, 10)


def test___truediv() -> None:
    value = ComplexInt(1, 2) / ComplexInt(3, 4)

    assert_type(value, complex)
    assert isinstance(value, complex)
    assert value == complex(1, 2) / complex(3, 4)


def test___floordiv() -> None:
    value = ComplexInt(3, 4) // ComplexInt(1, 2)

    assert_type(value, ComplexInt)
    assert isinstance(value, ComplexInt)
    assert value == ComplexInt(2, -1)


def test___pow() -> None:
    value = pow(ComplexInt(1, 2), ComplexInt(3, 4))

    assert_type(value, complex)
    assert isinstance(value, complex)
    assert value == pow(complex(1, 2), complex(3, 4))


def test___abs() -> None:
    value = abs(ComplexInt(3, 4))

    assert_type(value, float)
    assert isinstance(value, float)
    assert value == 5.0


def test___conjugate() -> None:
    value = ComplexInt(3, 4).conjugate()

    assert_type(value, ComplexInt)
    assert isinstance(value, ComplexInt)
    assert value == ComplexInt(3, -4)


@pytest.mark.parametrize(
    "left, right", [(ComplexInt(1, 2), ComplexInt(1, 2)), (ComplexInt(-3, 4), ComplexInt(-3, 4))]
)
def test___same_value___equality___equal(left: ComplexInt, right: ComplexInt) -> None:
    assert left == right
    assert not (left != right)


@pytest.mark.parametrize(
    "left, right", [(ComplexInt(1, 2), ComplexInt(1, 3)), (ComplexInt(-3, 4), ComplexInt(3, 4))]
)
def test___different_value___equality___equal(left: ComplexInt, right: ComplexInt) -> None:
    assert left != right
    assert not (left == right)


def test___str() -> None:
    value = str(ComplexInt(3, 4))

    assert_type(value, str)
    assert isinstance(value, str)
    assert value == "3+4j"


def test___repr() -> None:
    value = repr(ComplexInt(3, 4))

    assert_type(value, str)
    assert isinstance(value, str)
    assert value == "nitypes.complex.ComplexInt(3, 4)"


def test___hash() -> None:
    value = hash(ComplexInt(3, 4))

    assert_type(value, int)
    assert isinstance(value, int)
    assert value == hash((3, 4))


def test___destructure() -> None:
    real, imag = ComplexInt(3, 4)

    assert real == 3
    assert imag == 4


def test___iter() -> None:
    values = list(ComplexInt(3, 4))

    assert values == [3, 4]


def test___list() -> None:
    values = [ComplexInt(3, 4)]

    assert_type(values, list[ComplexInt])
    assert isinstance(values[0], ComplexInt)
    assert values[0] == ComplexInt(3, 4)
