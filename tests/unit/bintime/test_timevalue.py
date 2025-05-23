from __future__ import annotations

import copy
import datetime as dt
import pickle
import random
from decimal import Decimal
from typing import Any, Generator

import hightime as ht
import pytest
from typing_extensions import assert_type

from nitypes.bintime import TimeValue
from nitypes.bintime._timevalue import _INT128_MAX, _INT128_MIN


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


def test___dt_timedelta___construct___returns_nearest_time_value() -> None:
    # The microseconds are not exactly representable as a binary fraction, so they are rounded.
    value = TimeValue(dt.timedelta(days=12345, seconds=23456, microseconds=345_678))

    assert_type(value, TimeValue)
    assert isinstance(value, TimeValue)
    assert (
        value.days,
        value.seconds,
        value.microseconds,
        value.femtoseconds,
        value.yoctoseconds,
    ) == (12345, 23456, 345_677, 999_999_999, 999_972_046)


def test___ht_timedelta___construct___returns_nearest_time_value() -> None:
    # The yoctoseconds exceed the precision of NI-BTF, so they are rounded.
    value = TimeValue(
        ht.timedelta(
            days=12345,
            seconds=23456,
            microseconds=345_678,
            femtoseconds=456_789_012,
            yoctoseconds=567_890_123,
        )
    )

    assert_type(value, TimeValue)
    assert isinstance(value, TimeValue)
    assert (
        value.days,
        value.seconds,
        value.microseconds,
        value.femtoseconds,
        value.yoctoseconds,
    ) == (12345, 23456, 345_678, 456_789_012, 567_896_592)


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


#########################################################
# days, seconds, microseconds, femtoseconds, yoctoseconds
#########################################################
@pytest.mark.parametrize(
    "value, expected",
    [
        (TimeValue(0), (0, 0, 0, 0, 0)),
        (TimeValue(1), (0, 1, 0, 0, 0)),
        (TimeValue(60), (0, 60, 0, 0, 0)),
        (TimeValue(3600), (0, 3600, 0, 0, 0)),
        (TimeValue(86400), (1, 0, 0, 0, 0)),
        (TimeValue(86400 * 365), (365, 0, 0, 0, 0)),
        (TimeValue(86400 * 365 * 100), (36500, 0, 0, 0, 0)),
        (TimeValue(-1), (-1, 86399, 0, 0, 0)),
        (TimeValue(-60), (-1, 86400 - 60, 0, 0, 0)),
        (TimeValue(-3600), (-1, 86400 - 3600, 0, 0, 0)),
        (TimeValue(-86400), (-1, 0, 0, 0, 0)),
        (TimeValue(-86400 * 365), (-365, 0, 0, 0, 0)),
        (TimeValue(-86400 * 365 * 100), (-36500, 0, 0, 0, 0)),
        (TimeValue(Decimal("0.5")), (0, 0, 500_000, 0, 0)),
        (TimeValue(Decimal("0.005")), (0, 0, 4_999, 999_999_999, 999_995_663)),
        (TimeValue(Decimal("0.000_005")), (0, 0, 5, 0, 13_114)),
        (TimeValue(Decimal("0.000_000_000_000_005")), (0, 0, 0, 5, 15_158)),
        (TimeValue(Decimal("0.000_000_000_000_000_005")), (0, 0, 0, 0, 4_987_329)),
        (TimeValue(Decimal("-0.5")), (-1, 86399, 500_000, 0, 0)),
        (TimeValue(Decimal("-0.005")), (-1, 86399, 995_000, 0, 4_336)),
        (TimeValue(Decimal("-0.000_005")), (-1, 86399, 999_994, 999_999_999, 999_986_885)),
        (
            TimeValue(Decimal("-0.000_000_000_000_005")),
            (-1, 86399, 999_999, 999_999_994, 999_984_841),
        ),
        (
            TimeValue(Decimal("-0.000_000_000_000_000_005")),
            (-1, 86399, 999_999, 999_999_999, 995_012_670),
        ),
    ],
)
def test___various_values___unit_properties___return_unit_values(
    value: TimeValue, expected: tuple[int, ...]
) -> None:
    assert (
        value.days,
        value.seconds,
        value.microseconds,
        value.femtoseconds,
        value.yoctoseconds,
    ) == expected


###############
# total_seconds
###############
@pytest.mark.parametrize(
    "seconds",
    [0.0, 1.0, 3.14159, -3.14159, 1.23456789e18, -1.23456789e18, 1.23456789e-18, -1.23456789e-18],
)
def test___float_seconds___total_seconds___approximate_match(seconds: float) -> None:
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
def test___whole_decimal_seconds___precision_total_seconds___exact_match(
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
def test___fractional_decimal_seconds___precision_total_seconds___approximate_match(
    seconds: Decimal,
) -> None:
    value = TimeValue(seconds)

    total_seconds = value.precision_total_seconds()

    assert total_seconds == pytest.approx(seconds)


@pytest.mark.parametrize(
    "ticks",
    [
        0,
        1,
        -1,
        1 << 64,
        -1 << 64,
        (1 << 64) + 2,
        (-1 << 64) + -2,
        (1 << 124) + (2 << 64) + 3,
        (-1 << 124) + (-2 << 64) + -3,
        0x12345678_12345678_12345678_12345678,
        -0x12345678_12345678_12345678_12345678,
        (1 << 126) + 1,
        (-1 << 126) - 1,
    ],
)
def test___round_trip___precision_total_seconds___exact_match(ticks: int) -> None:
    value = TimeValue.from_ticks(ticks)

    total_seconds = value.precision_total_seconds()
    round_trip_value = TimeValue(total_seconds)

    assert round_trip_value._ticks == ticks


def test___random_round_trip___precision_total_seconds___exact_match(
    _random_state: tuple[Any, ...],
) -> None:
    random.seed(1)
    for iteration in range(0, 1000):
        ticks = random.randint(_INT128_MIN, _INT128_MAX)
        value = TimeValue.from_ticks(ticks)

        total_seconds = value.precision_total_seconds()
        round_trip_value = TimeValue(total_seconds)

        assert round_trip_value._ticks == ticks


@pytest.fixture
def _random_state() -> Generator[tuple[Any, ...]]:
    state = random.getstate()
    try:
        yield state
    finally:
        random.setstate(state)


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


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeValue(1), TimeValue(1), 1),
        (TimeValue(20000), TimeValue(200), 100),
        (
            TimeValue.from_ticks((2 << 124) + (4 << 64) + 6),
            TimeValue.from_ticks((1 << 124) + (2 << 64) + 3),
            2,
        ),
        (
            TimeValue.from_ticks((-3 << 124) + (-6 << 64) + -9),
            TimeValue.from_ticks((1 << 124) + (2 << 64) + 3),
            -3,
        ),
    ],
)
def test___time_value___floordiv___returns_int(
    left: TimeValue, right: TimeValue, expected: int
) -> None:
    assert_type(left // right, int)
    assert left // right == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeValue(1), 1, TimeValue(1)),
        (TimeValue(20000), 100, TimeValue(200)),
        (
            TimeValue.from_ticks((2 << 124) + (4 << 64) + 6),
            2,
            TimeValue.from_ticks((1 << 124) + (2 << 64) + 3),
        ),
        (
            TimeValue.from_ticks((-3 << 124) + (-6 << 64) + -9),
            -3,
            TimeValue.from_ticks((1 << 124) + (2 << 64) + 3),
        ),
    ],
)
def test___int___floordiv___returns_time_value(
    left: TimeValue, right: int, expected: TimeValue
) -> None:
    assert_type(left // right, TimeValue)
    assert left // right == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeValue(1), TimeValue(1), 1.0),
        (TimeValue(20000), TimeValue(200), 100.0),
        (TimeValue(200), TimeValue(20000), 0.01),
        (
            TimeValue.from_ticks((2 << 124) + (4 << 64) + 6),
            TimeValue.from_ticks((1 << 124) + (2 << 64) + 3),
            2.0,
        ),
        (
            TimeValue.from_ticks((-3 << 124) + (-6 << 64) + -9),
            TimeValue.from_ticks((1 << 124) + (2 << 64) + 3),
            -3.0,
        ),
    ],
)
def test___time_value___truediv___returns_float(
    left: TimeValue, right: TimeValue, expected: int
) -> None:
    assert_type(left / right, float)
    assert left / right == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeValue(1), 1.0, TimeValue(1)),
        (TimeValue(20000), 100.0, TimeValue(200)),
        (TimeValue(200), 0.01, TimeValue(20000)),
        (
            TimeValue.from_ticks((2 << 124) + (4 << 64) + 6),
            2.0,
            TimeValue.from_ticks((1 << 124) + (2 << 64) + 3),
        ),
        (
            TimeValue.from_ticks((-3 << 124) + (-6 << 64) + -9),
            -3.0,
            TimeValue.from_ticks((1 << 124) + (2 << 64) + 3),
        ),
    ],
)
def test___float___truediv___returns_approximate_time_value(
    left: TimeValue, right: float, expected: TimeValue
) -> None:
    assert_type(left / right, TimeValue)
    assert (left / right).total_seconds() == pytest.approx(expected.total_seconds())


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeValue(1), TimeValue(1), TimeValue(0)),
        (TimeValue(20042), TimeValue(200), TimeValue(42)),
        (
            TimeValue.from_ticks((2 << 124) + (5 << 64) + 6),
            TimeValue.from_ticks((1 << 124) + (2 << 64) + 3),
            TimeValue.from_ticks(1 << 64),
        ),
        (
            TimeValue.from_ticks((-3 << 124) + (-6 << 64) + -9),
            TimeValue.from_ticks((1 << 124) + (2 << 64) + 3),
            TimeValue.from_ticks(0),
        ),
    ],
)
def test___time_value___mod___returns_time_value(
    left: TimeValue, right: TimeValue, expected: TimeValue
) -> None:
    assert_type(left % right, TimeValue)
    assert left % right == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeValue(1), TimeValue(1), (1, TimeValue(0))),
        (TimeValue(20042), TimeValue(200), (100, TimeValue(42))),
        (
            TimeValue.from_ticks((2 << 124) + (5 << 64) + 6),
            TimeValue.from_ticks((1 << 124) + (2 << 64) + 3),
            (2, TimeValue.from_ticks(1 << 64)),
        ),
        (
            TimeValue.from_ticks((-3 << 124) + (-6 << 64) + -9),
            TimeValue.from_ticks((1 << 124) + (2 << 64) + 3),
            (-3, TimeValue.from_ticks(0)),
        ),
    ],
)
def test___time_value___divmod___returns_int_and_time_value(
    left: TimeValue, right: TimeValue, expected: tuple[int, TimeValue]
) -> None:
    assert_type(divmod(left, right), tuple[int, TimeValue])
    assert divmod(left, right) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (TimeValue(0), False),
        (TimeValue(1), True),
        (TimeValue(20042), True),
        (
            TimeValue.from_ticks((2 << 124) + (5 << 64) + 6),
            True,
        ),
        (TimeValue.from_ticks((-3 << 124) + (-6 << 64) + -9), True),
    ],
)
def test___time_value___bool___returns_not_zero(value: TimeValue, expected: bool) -> None:
    assert bool(value) == expected
    assert (not value) == (not expected)


_VARIOUS_VALUES = [
    TimeValue(0),
    TimeValue(2),
    TimeValue(-2),
    TimeValue.from_ticks((1 << 124) + (2 << 64) + 3),
    TimeValue.from_ticks((-1 << 124) + (-2 << 64) + -3),
    TimeValue.min,
    TimeValue.max,
]


def test___various_values___hash___returns_probably_unique_int() -> None:
    hashes = set([hash(x) for x in _VARIOUS_VALUES])
    assert len(hashes) == len(_VARIOUS_VALUES)


@pytest.mark.parametrize(
    "value",
    [
        TimeValue(0),
        TimeValue(2),
        TimeValue(-2),
        TimeValue.from_ticks((1 << 124) + (2 << 64) + 3),
        TimeValue.from_ticks((-1 << 124) + (-2 << 64) + -3),
    ],
)
def test___various_values___copy___makes_copy(value: TimeValue) -> None:
    new_value = copy.copy(value)
    assert new_value is not value
    assert new_value == value


@pytest.mark.parametrize(
    "value",
    [
        TimeValue(0),
        TimeValue(2),
        TimeValue(-2),
        TimeValue.from_ticks((1 << 124) + (2 << 64) + 3),
        TimeValue.from_ticks((-1 << 124) + (-2 << 64) + -3),
    ],
)
def test___various_values___pickle_unpickle___makes_copy(value: TimeValue) -> None:
    new_value = pickle.loads(pickle.dumps(value))
    assert new_value is not value
    assert new_value == value


def test___time_value___pickle___references_public_modules() -> None:
    value = TimeValue(123)
    value_bytes = pickle.dumps(value)

    assert b"nitypes.bintime" in value_bytes
    assert b"nitypes.bintime._timevalue" not in value_bytes


@pytest.mark.parametrize(
    "value, expected",
    [
        (TimeValue(0), "0:00:00.000000000000000000"),
        (TimeValue(1), "0:00:01.000000000000000000"),
        (TimeValue(60), "0:01:00.000000000000000000"),
        (TimeValue(3600), "1:00:00.000000000000000000"),
        (TimeValue(86400), "1 day, 0:00:00.000000000000000000"),
        (TimeValue(86400 * 3), "3 days, 0:00:00.000000000000000000"),
        (TimeValue(-1), "-1 day, 23:59:59.000000000000000000"),
        (TimeValue(-60), "-1 day, 23:59:00.000000000000000000"),
        (TimeValue(-3600), "-1 day, 23:00:00.000000000000000000"),
        (TimeValue(-86400), "-1 day, 0:00:00.000000000000000000"),
        (TimeValue(-86400 * 3), "-3 days, 0:00:00.000000000000000000"),
        (TimeValue(Decimal("0.5")), "0:00:00.500000000000000000"),
        (TimeValue(Decimal("0.005")), "0:00:00.005000000000000000"),
        (TimeValue(Decimal("0.000_005")), "0:00:00.000005000000000000"),
        (TimeValue(Decimal("0.000_000_000_000_005")), "0:00:00.000000000000005000"),
        (TimeValue(Decimal("0.000_000_000_000_000_005")), "0:00:00.000000000000000005"),
        (TimeValue(Decimal("-0.5")), "-1 day, 23:59:59.500000000000000000"),
        (TimeValue(Decimal("-0.005")), "-1 day, 23:59:59.995000000000000000"),
        (TimeValue(Decimal("-0.000_005")), "-1 day, 23:59:59.999995000000000000"),
        (TimeValue(Decimal("-0.000_000_000_000_005")), "-1 day, 23:59:59.999999999999995000"),
        (TimeValue(Decimal("-0.000_000_000_000_000_005")), "-1 day, 23:59:59.999999999999999995"),
    ],
)
def test___various_values___str___looks_ok(value: TimeValue, expected: str) -> None:
    assert str(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (TimeValue(0), "nitypes.bintime.TimeValue(Decimal('0'))"),
        (TimeValue(1), "nitypes.bintime.TimeValue(Decimal('1'))"),
        (TimeValue(60), "nitypes.bintime.TimeValue(Decimal('60'))"),
        (TimeValue(3600), "nitypes.bintime.TimeValue(Decimal('3600'))"),
        (TimeValue(86400), "nitypes.bintime.TimeValue(Decimal('86400'))"),
        (TimeValue(86400 * 3), "nitypes.bintime.TimeValue(Decimal('259200'))"),
        (TimeValue(-1), "nitypes.bintime.TimeValue(Decimal('-1'))"),
        (TimeValue(-60), "nitypes.bintime.TimeValue(Decimal('-60'))"),
        (TimeValue(-3600), "nitypes.bintime.TimeValue(Decimal('-3600'))"),
        (TimeValue(-86400), "nitypes.bintime.TimeValue(Decimal('-86400'))"),
        (TimeValue(-86400 * 3), "nitypes.bintime.TimeValue(Decimal('-259200'))"),
        (TimeValue(Decimal("0.5")), "nitypes.bintime.TimeValue(Decimal('0.5'))"),
        (
            TimeValue(Decimal("0.25")),
            "nitypes.bintime.TimeValue(Decimal('0.25'))",
        ),
        (
            TimeValue(Decimal("0.125")),
            "nitypes.bintime.TimeValue(Decimal('0.125'))",
        ),
        (TimeValue(Decimal("-0.5")), "nitypes.bintime.TimeValue(Decimal('-0.5'))"),
        (
            TimeValue(Decimal("-0.25")),
            "nitypes.bintime.TimeValue(Decimal('-0.25'))",
        ),
        (
            TimeValue(Decimal("-0.125")),
            "nitypes.bintime.TimeValue(Decimal('-0.125'))",
        ),
        # The fractional part gets bruised because 0.005 isn't expressible as 2^-N
        (
            TimeValue(Decimal("0.005")),
            "nitypes.bintime.TimeValue(Decimal('0.004999999999999999995663191310057982263970188796520233154296875'))",
        ),
        (
            TimeValue(Decimal("-0.005")),
            "nitypes.bintime.TimeValue(Decimal('-0.004999999999999999995663191310057982263970188796520233154296875'))",
        ),
    ],
)
def test___various_values___repr___looks_ok(value: TimeValue, expected: str) -> None:
    assert repr(value) == expected
