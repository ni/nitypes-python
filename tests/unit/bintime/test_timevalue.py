from __future__ import annotations

from decimal import Decimal

import pytest
from typing_extensions import assert_type

from nitypes.bintime import TimeValue


##############
# Constructors
##############
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


def test___int_seconds_and_ticks___construct___returns_time_value() -> None:
    value = TimeValue(0x1234567890ABCEF, 0xFEDCBA09_87654321)

    assert_type(value, TimeValue)
    assert isinstance(value, TimeValue)
    assert value._ticks == 0x1234567890ABCEF_FEDCBA09_87654321


@pytest.mark.parametrize("ticks", [1 << 127, -(1 << 127) - 1])
def test___int_ticks_out_of_range___construct___raises_overflow_error(ticks: int) -> None:
    with pytest.raises(OverflowError) as exc:
        _ = TimeValue(ticks=ticks)

    assert exc.value.args[0].startswith("The time value is out of range")


def test___float_seconds___construct___returns_time_value() -> None:
    value = TimeValue(123456.789)

    assert_type(value, TimeValue)
    assert isinstance(value, TimeValue)
    assert (value._ticks >> 64) == 123456
    assert (value._ticks & 0xFFFFFFFF_FFFFFFFF) == pytest.approx(0.789 * (1 << 64))


def test___float_seconds_and_ticks___construct___raises_value_error() -> None:
    with pytest.raises(ValueError) as exc:
        _ = TimeValue(123456.0, 789.0)  # type: ignore[overload]

    assert exc.value.args[0].startswith("The ticks argument is not supported.")


# Note that double-precision floating point cannot exactly represent 1<<63.
@pytest.mark.parametrize("seconds", [1e19, -1e19])
def test___float_seconds_out_of_range___construct___raises_overflow_error(seconds: float) -> None:
    with pytest.raises(OverflowError) as exc:
        _ = TimeValue(seconds)

    assert exc.value.args[0].startswith("The time value is out of range")


def test___decimal_seconds___construct___returns_time_value() -> None:
    value = TimeValue(Decimal("123456.789"))

    assert_type(value, TimeValue)
    assert isinstance(value, TimeValue)
    assert (value._ticks >> 64) == 123456
    assert (value._ticks & 0xFFFFFFFF_FFFFFFFF) == pytest.approx(Decimal("0.789") * (1 << 64))


def test___decimal_seconds_and_ticks___construct___raises_value_error() -> None:
    with pytest.raises(ValueError) as exc:
        _ = TimeValue(Decimal("123456"), Decimal("789"))  # type: ignore[overload]

    assert exc.value.args[0].startswith("The ticks argument is not supported.")


@pytest.mark.parametrize(
    "seconds", [Decimal("9223372036854775808"), Decimal("-9223372036854775809")]
)
def test___decimal_seconds_out_of_range___construct___raises_overflow_error(
    seconds: Decimal,
) -> None:
    with pytest.raises(OverflowError) as exc:
        _ = TimeValue(seconds)

    assert exc.value.args[0].startswith("The time value is out of range")


def test___invalid_seconds_type___construct___raises_type_error() -> None:
    with pytest.raises(TypeError) as exc:
        _ = TimeValue("0")  # type: ignore[overload]

    assert exc.value.args[0].startswith("The seconds must be an integer or float.")


def test___invalid_ticks_type___construct___raises_type_error() -> None:
    with pytest.raises(TypeError) as exc:
        _ = TimeValue(ticks="0")  # type: ignore[overload]

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


##############
# Add/subtract
##############
@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeValue(0), TimeValue(0), TimeValue(0)),
        (TimeValue(2), TimeValue(2), TimeValue(4)),
        (TimeValue(2), TimeValue(-2), TimeValue(0)),
        (TimeValue(1, 2), TimeValue(3, 4), TimeValue(4, 6)),
        (TimeValue(1, 2), TimeValue(-3, -4), TimeValue(-2, -2)),
        (TimeValue(1 << 60, 2 << 60), TimeValue(3 << 60, 4 << 60), TimeValue(4 << 60, 6 << 60)),
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
        (TimeValue(1, 2), TimeValue(3, 4), TimeValue(-2, -2)),
        (TimeValue(1, 2), TimeValue(-3, -4), TimeValue(4, 6)),
        (TimeValue(1 << 60, 2 << 60), TimeValue(3 << 60, 4 << 60), TimeValue(-2 << 60, -2 << 60)),
    ],
)
def test___time_values___sub___returns_difference(
    left: TimeValue, right: TimeValue, expected: TimeValue
) -> None:
    assert left - right == expected


#############
# Neg/pos/abs
#############
@pytest.mark.parametrize(
    "value, expected",
    [
        (TimeValue(0), TimeValue(0)),
        (TimeValue(2), TimeValue(-2)),
        (TimeValue(-2), TimeValue(2)),
        (TimeValue(1 << 60, 2 << 60), TimeValue(-1 << 60, -2 << 60)),
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
        (TimeValue(1 << 60, 2 << 60), TimeValue(1 << 60, 2 << 60)),
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
        (TimeValue(1 << 60, 2 << 60), TimeValue(1 << 60, 2 << 60)),
        (TimeValue(-1 << 60, -2 << 60), TimeValue(1 << 60, 2 << 60)),
    ],
)
def test___time_values___abs___returns_absolute_value(
    value: TimeValue, expected: TimeValue
) -> None:
    assert abs(value) == expected


#################
# Multiply/divide
#################
@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeValue(0), 0, TimeValue(0)),
        (TimeValue(1), 0, TimeValue(0)),
        (TimeValue(1), 1, TimeValue(1)),
        (TimeValue(1), -1, TimeValue(-1)),
        (TimeValue(100), 200, TimeValue(20000)),
        (TimeValue(100, 200), -300, TimeValue(-30000, -60000)),
        (TimeValue(1 << 60, 2 << 60), 1, TimeValue(1 << 60, 2 << 60)),
        (TimeValue(1 << 60, 2 << 60), -3, TimeValue(-3 << 60, -6 << 60)),
    ],
)
def test___time_value_and_int___mul___returns_product(
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
        (TimeValue(100, 200), -300.0, TimeValue(-30000, -60000)),
        (TimeValue(1 << 60), 1.0, TimeValue(1 << 60)),
        (TimeValue(1 << 60), -3.0, TimeValue(-3 << 60)),
        (TimeValue(1 << 60), 0.5, TimeValue(1 << 59)),
        (TimeValue(1 << 60, 2 << 60), 1.0, TimeValue(1 << 60, 2 << 60)),
        (TimeValue(1 << 60, 2 << 60), -3.0, TimeValue(-3 << 60, -6 << 60)),
        (TimeValue(1 << 60, 2 << 60), 0.5, TimeValue(1 << 59, 2 << 59)),
    ],
)
def test___time_value_and_exact_float___mul___returns_exact_product(
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
def test___time_value_and_inexact_float___mul___returns_approximate_product(
    left: TimeValue, right: float, expected: TimeValue
) -> None:
    assert (left * right).total_seconds() == pytest.approx(expected.total_seconds())
    assert (right * left).total_seconds() == pytest.approx(expected.total_seconds())
