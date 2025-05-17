from __future__ import annotations

from decimal import Decimal

import pytest
from typing_extensions import assert_type

from nitypes.bintime import TimeValue


#############
# Constructor
#############
def test___no_args___construct___returns_zero_time_value() -> None:
    value = TimeValue()

    assert_type(value, TimeValue)
    assert isinstance(value, TimeValue)
    assert value._ticks == 0


def test___int_seconds___construct___returns_time_value() -> None:
    value = TimeValue(0x12345678_90ABCDEF)

    assert_type(value, TimeValue)
    assert isinstance(value, TimeValue)
    assert value._ticks == 0x12345678_90ABCDEF_00000000_00000000


def test___float_seconds___construct___returns_time_value() -> None:
    value = TimeValue(123456.789)

    assert_type(value, TimeValue)
    assert isinstance(value, TimeValue)
    assert (value._ticks >> 64) == 123456
    assert (value._ticks & 0xFFFFFFFF_FFFFFFFF) == pytest.approx(0.789 * (1 << 64))


def test___decimal_seconds___construct___returns_time_value() -> None:
    value = TimeValue(Decimal("123456.789"))

    assert_type(value, TimeValue)
    assert isinstance(value, TimeValue)
    assert (value._ticks >> 64) == 123456
    assert (value._ticks & 0xFFFFFFFF_FFFFFFFF) == pytest.approx(Decimal("0.789") * (1 << 64))


@pytest.mark.parametrize(
    "seconds",
    [
        1 << 63,
        (-1 << 63) - 1,
        # Note that double-precision floating point cannot exactly represent 1<<63.
        1e19,
        -1e19,
        Decimal("9223372036854775808"),
        Decimal("-9223372036854775809"),
    ],
)
def test___out_of_range___construct___raises_overflow_error(seconds: int | float | Decimal) -> None:
    with pytest.raises(OverflowError) as exc:
        _ = TimeValue(seconds)

    assert exc.value.args[0].startswith("The seconds value is out of range.")


def test___invalid_seconds_type___construct___raises_type_error() -> None:
    with pytest.raises(TypeError) as exc:
        _ = TimeValue("0")  # type: ignore[arg-type]

    assert exc.value.args[0].startswith("The seconds must be a number or timedelta.")


############
# from_ticks
############
def test___int_ticks___from_ticks___returns_time_value() -> None:
    value = TimeValue.from_ticks(0x12345678_90ABCDEF_FEDCBA09_87654321)

    assert_type(value, TimeValue)
    assert isinstance(value, TimeValue)
    assert value._ticks == 0x12345678_90ABCDEF_FEDCBA09_87654321


@pytest.mark.parametrize("ticks", [1 << 127, -(1 << 127) - 1])
def test___out_of_range___from_ticks___raises_overflow_error(ticks: int) -> None:
    with pytest.raises(OverflowError) as exc:
        _ = TimeValue.from_ticks(ticks)

    assert exc.value.args[0].startswith("The ticks value is out of range.")


def test___unsupported_type___from_ticks___raises_type_error() -> None:
    with pytest.raises(TypeError) as exc:
        _ = TimeValue.from_ticks("0")  # type: ignore[arg-type]

    assert exc.value.args[0].startswith("The ticks must be an integer.")


###############
# total_seconds
###############
@pytest.mark.parametrize(
    "seconds",
    [0.0, 1.0, 3.14159, -3.14159, 1.23456789e18, -1.23456789e18, 1.23456789e-18, -1.23456789e-18],
)
def test___float_seconds___total_seconds___approximately_matches(seconds: float) -> None:
    value = TimeValue(seconds)

    total_seconds = value.total_seconds()

    assert total_seconds == pytest.approx(seconds)


#########################
# precision_total_seconds
#########################
@pytest.mark.parametrize(
    "seconds",
    [
        Decimal("0.0"),
        Decimal("1.0"),
        Decimal("1.23456789e18"),
        Decimal("-1.23456789e18"),
        Decimal("9223372036854775807"),
        Decimal("-9223372036854775808"),
    ],
)
def test___whole_decimal_seconds___precision_total_seconds___exactly_matches(
    seconds: Decimal,
) -> None:
    value = TimeValue(seconds)

    total_seconds = value.precision_total_seconds()

    assert total_seconds == seconds


# Decimal is precise, but fixed point "bruises" small values.
@pytest.mark.parametrize(
    "seconds",
    [
        Decimal("3.14159"),
        Decimal("-3.14159"),
        Decimal("1.23456789e-18"),
        Decimal("-1.23456789e-18"),
    ],
)
def test___fractional_decimal_seconds___precision_total_seconds___approximately_matches(
    seconds: Decimal,
) -> None:
    value = TimeValue(seconds)

    total_seconds = value.precision_total_seconds()

    assert total_seconds == pytest.approx(seconds)


##################
# Unary arithmetic
##################
@pytest.mark.parametrize(
    "value, expected",
    [
        (TimeValue(0), TimeValue(0)),
        (TimeValue(2), TimeValue(-2)),
        (TimeValue(-2), TimeValue(2)),
        (
            TimeValue.from_ticks((1 << 124) + (2 << 64) + 3),
            TimeValue.from_ticks((-1 << 124) + (-2 << 64) + -3),
        ),
    ],
)
def test___time_values___neg___returns_negation(value: TimeValue, expected: TimeValue) -> None:
    assert -value == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (TimeValue(0), TimeValue(0)),
        (TimeValue(2), TimeValue(2)),
        (TimeValue(-2), TimeValue(-2)),
        (
            TimeValue.from_ticks((1 << 124) + (2 << 64) + 3),
            TimeValue.from_ticks((1 << 124) + (2 << 64) + 3),
        ),
        (
            TimeValue.from_ticks((-1 << 124) + (-2 << 64) + -3),
            TimeValue.from_ticks((-1 << 124) + (-2 << 64) + -3),
        ),
    ],
)
def test___time_values___pos___returns_identity(value: TimeValue, expected: TimeValue) -> None:
    assert +value == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (TimeValue(0), TimeValue(0)),
        (TimeValue(2), TimeValue(2)),
        (TimeValue(-2), TimeValue(2)),
        (
            TimeValue.from_ticks((1 << 124) + (2 << 64) + 3),
            TimeValue.from_ticks((1 << 124) + (2 << 64) + 3),
        ),
        (
            TimeValue.from_ticks((-1 << 124) + (-2 << 64) + -3),
            TimeValue.from_ticks((1 << 124) + (2 << 64) + 3),
        ),
    ],
)
def test___time_values___abs___returns_absolute_value(
    value: TimeValue, expected: TimeValue
) -> None:
    assert abs(value) == expected


###################
# Binary arithmetic
###################
@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeValue(0), TimeValue(0), TimeValue(0)),
        (TimeValue(2), TimeValue(2), TimeValue(4)),
        (TimeValue(2), TimeValue(-2), TimeValue(0)),
        (
            TimeValue.from_ticks((1 << 124) + (2 << 64) + 3),
            TimeValue.from_ticks((3 << 124) + (4 << 64) + 5),
            TimeValue.from_ticks((4 << 124) + (6 << 64) + 8),
        ),
        (
            TimeValue.from_ticks((1 << 124) + (2 << 64) + 3),
            TimeValue.from_ticks((-3 << 124) + (-4 << 64) + -5),
            TimeValue.from_ticks((-2 << 124) + (-2 << 64) + -2),
        ),
    ],
)
def test___time_values___add___returns_sum(
    left: TimeValue, right: TimeValue, expected: TimeValue
) -> None:
    assert left + right == expected
    assert right + left == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeValue(0), TimeValue(0), TimeValue(0)),
        (TimeValue(2), TimeValue(2), TimeValue(0)),
        (TimeValue(2), TimeValue(-2), TimeValue(4)),
        (
            TimeValue.from_ticks((1 << 124) + (2 << 64) + 3),
            TimeValue.from_ticks((3 << 124) + (4 << 64) + 5),
            TimeValue.from_ticks((-2 << 124) + (-2 << 64) + -2),
        ),
        (
            TimeValue.from_ticks((1 << 124) + (2 << 64) + 3),
            TimeValue.from_ticks((-3 << 124) + (-4 << 64) + -5),
            TimeValue.from_ticks((4 << 124) + (6 << 64) + 8),
        ),
    ],
)
def test___time_values___sub___returns_difference(
    left: TimeValue, right: TimeValue, expected: TimeValue
) -> None:
    assert left - right == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeValue(0), 0, TimeValue(0)),
        (TimeValue(1), 0, TimeValue(0)),
        (TimeValue(1), 1, TimeValue(1)),
        (TimeValue(1), -1, TimeValue(-1)),
        (TimeValue(100), 200, TimeValue(20000)),
        (
            TimeValue.from_ticks((1 << 124) + (2 << 64) + 3),
            2,
            TimeValue.from_ticks((2 << 124) + (4 << 64) + 6),
        ),
        (
            TimeValue.from_ticks((1 << 124) + (2 << 64) + 3),
            -3,
            TimeValue.from_ticks((-3 << 124) + (-6 << 64) + -9),
        ),
    ],
)
def test___int___mul___returns_exact_product(
    left: TimeValue, right: int, expected: TimeValue
) -> None:
    assert left * right == expected
    assert right * left == expected


# Verify that multiplying by a float does not reduce the TimeValue's precision.
@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeValue(0), 0.0, TimeValue(0)),
        (TimeValue(1), 0.0, TimeValue(0)),
        (TimeValue(1), 1.0, TimeValue(1)),
        (TimeValue(1), -1.0, TimeValue(-1)),
        (TimeValue(100), 200.0, TimeValue(20000)),
        (
            TimeValue.from_ticks((1 << 124) + (2 << 64) + 3),
            1.0,
            TimeValue.from_ticks((1 << 124) + (2 << 64) + 3),
        ),
        (
            TimeValue.from_ticks((1 << 124) + (2 << 64) + 3),
            -1.0,
            TimeValue.from_ticks((-1 << 124) + (-2 << 64) + -3),
        ),
    ],
)
def test___exact_float___mul___returns_exact_product(
    left: TimeValue, right: float, expected: TimeValue
) -> None:
    assert left * right == expected
    assert right * left == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeValue(100), 1.23456789, TimeValue(123.456789)),
        (TimeValue(1e18), 1.23456789, TimeValue(1.23456789e18)),
        (TimeValue(-1e18), 1.23456789, TimeValue(-1.23456789e18)),
    ],
)
def test___inexact_float___mul___returns_approximate_product(
    left: TimeValue, right: float, expected: TimeValue
) -> None:
    assert (left * right).total_seconds() == pytest.approx(expected.total_seconds())
    assert (right * left).total_seconds() == pytest.approx(expected.total_seconds())


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeValue(0), Decimal("0.0"), TimeValue(0)),
        (TimeValue(1), Decimal("0.0"), TimeValue(0)),
        (TimeValue(1), Decimal("1.0"), TimeValue(1)),
        (TimeValue(1), Decimal("-1.0"), TimeValue(-1)),
        (TimeValue(100), Decimal("200.0"), TimeValue(20000)),
        (
            TimeValue.from_ticks((1 << 124) + (2 << 64) + 3),
            Decimal("2.0"),
            TimeValue.from_ticks((2 << 124) + (4 << 64) + 6),
        ),
        (
            TimeValue.from_ticks((1 << 124) + (2 << 64) + 3),
            Decimal("-3.0"),
            TimeValue.from_ticks((-3 << 124) + (-6 << 64) + -9),
        ),
        (TimeValue(100), Decimal("1.23456789"), TimeValue(Decimal("123.456789"))),
        (TimeValue(1e18), Decimal("1.23456789"), TimeValue(Decimal("1.23456789e18"))),
        (TimeValue(-1e18), Decimal("1.23456789"), TimeValue(Decimal("-1.23456789e18"))),
    ],
)
def test___decimal___mul___returns_exact_product(
    left: TimeValue, right: float, expected: TimeValue
) -> None:
    assert left * right == expected
    assert right * left == expected
