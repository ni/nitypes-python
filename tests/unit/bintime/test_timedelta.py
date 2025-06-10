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

from nitypes.bintime import TimeDelta
from nitypes.bintime._timedelta import _INT128_MAX, _INT128_MIN

_BT_EPSILON = ht.timedelta(yoctoseconds=54210)
_DT_EPSILON = ht.timedelta(microseconds=1)


#############
# Constructor
#############
def test___no_args___construct___returns_zero_timedelta() -> None:
    value = TimeDelta()

    assert_type(value, TimeDelta)
    assert isinstance(value, TimeDelta)
    assert value._ticks == 0


def test___int_seconds___construct___returns_timedelta() -> None:
    value = TimeDelta(0x12345678_90ABCDEF)

    assert_type(value, TimeDelta)
    assert isinstance(value, TimeDelta)
    assert value._ticks == 0x12345678_90ABCDEF_00000000_00000000


def test___float_seconds___construct___returns_timedelta() -> None:
    value = TimeDelta(123456.789)

    assert_type(value, TimeDelta)
    assert isinstance(value, TimeDelta)
    assert (value._ticks >> 64) == 123456
    assert (value._ticks & 0xFFFFFFFF_FFFFFFFF) == pytest.approx(0.789 * (1 << 64))


def test___decimal_seconds___construct___returns_timedelta() -> None:
    value = TimeDelta(Decimal("123456.789"))

    assert_type(value, TimeDelta)
    assert isinstance(value, TimeDelta)
    assert (value._ticks >> 64) == 123456
    assert (value._ticks & 0xFFFFFFFF_FFFFFFFF) == pytest.approx(Decimal("0.789") * (1 << 64))


def test___dt_timedelta___construct___returns_nearest_timedelta() -> None:
    # The microseconds are not exactly representable as a binary fraction, so they are rounded.
    value = TimeDelta(dt.timedelta(days=12345, seconds=23456, microseconds=345_678))

    assert_type(value, TimeDelta)
    assert isinstance(value, TimeDelta)
    assert (
        value.days,
        value.seconds,
        value.microseconds,
        value.femtoseconds,
        value.yoctoseconds,
    ) == (12345, 23456, 345_677, 999_999_999, 999_972_046)


def test___ht_timedelta___construct___returns_nearest_timedelta() -> None:
    # The yoctoseconds exceed the precision of NI-BTF, so they are rounded.
    value = TimeDelta(
        ht.timedelta(
            days=12345,
            seconds=23456,
            microseconds=345_678,
            femtoseconds=456_789_012,
            yoctoseconds=567_890_123,
        )
    )

    assert_type(value, TimeDelta)
    assert isinstance(value, TimeDelta)
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
        _ = TimeDelta(seconds)

    assert exc.value.args[0].startswith("The seconds value is out of range.")


def test___invalid_seconds_type___construct___raises_type_error() -> None:
    with pytest.raises(TypeError) as exc:
        _ = TimeDelta("0")  # type: ignore[call-overload]

    assert exc.value.args[0].startswith("The seconds must be a number or timedelta.")


############
# from_ticks
############
def test___int_ticks___from_ticks___returns_timedelta() -> None:
    value = TimeDelta.from_ticks(0x12345678_90ABCDEF_FEDCBA09_87654321)

    assert_type(value, TimeDelta)
    assert isinstance(value, TimeDelta)
    assert value._ticks == 0x12345678_90ABCDEF_FEDCBA09_87654321


@pytest.mark.parametrize("ticks", [1 << 127, -(1 << 127) - 1])
def test___out_of_range___from_ticks___raises_overflow_error(ticks: int) -> None:
    with pytest.raises(OverflowError) as exc:
        _ = TimeDelta.from_ticks(ticks)

    assert exc.value.args[0].startswith("The ticks value is out of range.")


def test___unsupported_type___from_ticks___raises_type_error() -> None:
    with pytest.raises(TypeError) as exc:
        _ = TimeDelta.from_ticks("0")  # type: ignore[arg-type]

    assert exc.value.args[0].startswith("The ticks must be an integer.")


#########################################################
# days, seconds, microseconds, femtoseconds, yoctoseconds
#########################################################
@pytest.mark.parametrize(
    "value, expected",
    [
        (TimeDelta(0), (0, 0, 0, 0, 0)),
        (TimeDelta(1), (0, 1, 0, 0, 0)),
        (TimeDelta(60), (0, 60, 0, 0, 0)),
        (TimeDelta(3600), (0, 3600, 0, 0, 0)),
        (TimeDelta(86400), (1, 0, 0, 0, 0)),
        (TimeDelta(86400 * 365), (365, 0, 0, 0, 0)),
        (TimeDelta(86400 * 365 * 100), (36500, 0, 0, 0, 0)),
        (TimeDelta(-1), (-1, 86399, 0, 0, 0)),
        (TimeDelta(-60), (-1, 86400 - 60, 0, 0, 0)),
        (TimeDelta(-3600), (-1, 86400 - 3600, 0, 0, 0)),
        (TimeDelta(-86400), (-1, 0, 0, 0, 0)),
        (TimeDelta(-86400 * 365), (-365, 0, 0, 0, 0)),
        (TimeDelta(-86400 * 365 * 100), (-36500, 0, 0, 0, 0)),
        (TimeDelta(Decimal("0.5")), (0, 0, 500_000, 0, 0)),
        (TimeDelta(Decimal("0.005")), (0, 0, 4_999, 999_999_999, 999_995_663)),
        (TimeDelta(Decimal("0.000_005")), (0, 0, 5, 0, 13_114)),
        (TimeDelta(Decimal("0.000_000_000_000_005")), (0, 0, 0, 5, 15_158)),
        (TimeDelta(Decimal("0.000_000_000_000_000_005")), (0, 0, 0, 0, 4_987_329)),
        (TimeDelta(Decimal("-0.5")), (-1, 86399, 500_000, 0, 0)),
        (TimeDelta(Decimal("-0.005")), (-1, 86399, 995_000, 0, 4_336)),
        (TimeDelta(Decimal("-0.000_005")), (-1, 86399, 999_994, 999_999_999, 999_986_885)),
        (
            TimeDelta(Decimal("-0.000_000_000_000_005")),
            (-1, 86399, 999_999, 999_999_994, 999_984_841),
        ),
        (
            TimeDelta(Decimal("-0.000_000_000_000_000_005")),
            (-1, 86399, 999_999, 999_999_999, 995_012_670),
        ),
    ],
)
def test___various_values___unit_properties___return_unit_values(
    value: TimeDelta, expected: tuple[int, ...]
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
    value = TimeDelta(seconds)

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
    value = TimeDelta(seconds)

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
    value = TimeDelta(seconds)

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
    value = TimeDelta.from_ticks(ticks)

    total_seconds = value.precision_total_seconds()
    round_trip_value = TimeDelta(total_seconds)

    assert round_trip_value._ticks == ticks


def test___random_round_trip___precision_total_seconds___exact_match(
    _random_state: tuple[Any, ...],
) -> None:
    random.seed(1)
    for iteration in range(0, 1000):
        ticks = random.randint(_INT128_MIN, _INT128_MAX)
        value = TimeDelta.from_ticks(ticks)

        total_seconds = value.precision_total_seconds()
        round_trip_value = TimeDelta(total_seconds)

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
        (TimeDelta(0), TimeDelta(0)),
        (TimeDelta(2), TimeDelta(-2)),
        (TimeDelta(-2), TimeDelta(2)),
        (
            TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
            TimeDelta.from_ticks((-1 << 124) + (-2 << 64) + -3),
        ),
    ],
)
def test___timedeltas___neg___returns_negation(value: TimeDelta, expected: TimeDelta) -> None:
    assert -value == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (TimeDelta(0), TimeDelta(0)),
        (TimeDelta(2), TimeDelta(2)),
        (TimeDelta(-2), TimeDelta(-2)),
        (
            TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
            TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
        ),
        (
            TimeDelta.from_ticks((-1 << 124) + (-2 << 64) + -3),
            TimeDelta.from_ticks((-1 << 124) + (-2 << 64) + -3),
        ),
    ],
)
def test___timedeltas___pos___returns_identity(value: TimeDelta, expected: TimeDelta) -> None:
    assert +value == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (TimeDelta(0), TimeDelta(0)),
        (TimeDelta(2), TimeDelta(2)),
        (TimeDelta(-2), TimeDelta(2)),
        (
            TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
            TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
        ),
        (
            TimeDelta.from_ticks((-1 << 124) + (-2 << 64) + -3),
            TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
        ),
    ],
)
def test___timedeltas___abs___returns_absolute_value(value: TimeDelta, expected: TimeDelta) -> None:
    assert abs(value) == expected


###################
# Binary arithmetic
###################
@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeDelta(0), TimeDelta(0), TimeDelta(0)),
        (TimeDelta(2), TimeDelta(2), TimeDelta(4)),
        (TimeDelta(2), TimeDelta(-2), TimeDelta(0)),
        (
            TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
            TimeDelta.from_ticks((3 << 124) + (4 << 64) + 5),
            TimeDelta.from_ticks((4 << 124) + (6 << 64) + 8),
        ),
        (
            TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
            TimeDelta.from_ticks((-3 << 124) + (-4 << 64) + -5),
            TimeDelta.from_ticks((-2 << 124) + (-2 << 64) + -2),
        ),
    ],
)
def test___timedeltas___add___returns_sum(
    left: TimeDelta, right: TimeDelta, expected: TimeDelta
) -> None:
    assert left + right == expected
    assert right + left == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeDelta(0), dt.timedelta(seconds=0), TimeDelta(0)),
        (TimeDelta(2), dt.timedelta(seconds=2), TimeDelta(4)),
        (TimeDelta(2), dt.timedelta(seconds=-2), TimeDelta(0)),
        (
            TimeDelta(Decimal("1e15")),
            dt.timedelta(seconds=123),
            TimeDelta(Decimal("1_000_000_000_000_123")),
        ),
        (
            TimeDelta(Decimal("1e15")),
            -dt.timedelta(seconds=1),
            TimeDelta(Decimal("999_999_999_999_999")),
        ),
        (
            TimeDelta(Decimal("1e15")),
            dt.timedelta(microseconds=15625),  # exact binary fraction
            TimeDelta(Decimal("1_000_000_000_000_000.015_625")),
        ),
    ],
)
def test___dt_timedelta___add___returns_sum(
    left: TimeDelta, right: dt.timedelta, expected: TimeDelta
) -> None:
    assert left + right == expected
    assert right + left == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (
            TimeDelta(Decimal("1e15")),
            dt.timedelta(microseconds=314159),
            TimeDelta(Decimal("1_000_000_000_000_000.314_159")),
        ),
    ],
)
def test___dt_timedelta_inexact_result___add___returns_approximate_sum(
    left: TimeDelta, right: dt.timedelta, expected: TimeDelta
) -> None:
    assert (left + right).precision_total_seconds() == pytest.approx(
        expected.precision_total_seconds()
    )
    assert (right + left).precision_total_seconds() == pytest.approx(
        expected.precision_total_seconds()
    )


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeDelta(0), ht.timedelta(seconds=0), TimeDelta(0)),
        (TimeDelta(2), ht.timedelta(seconds=2), TimeDelta(4)),
        (TimeDelta(2), ht.timedelta(seconds=-2), TimeDelta(0)),
        (
            TimeDelta(Decimal("1e15")),
            ht.timedelta(seconds=123),
            TimeDelta(Decimal("1_000_000_000_000_123")),
        ),
        (
            TimeDelta(Decimal("1e15")),
            -ht.timedelta(seconds=1),
            TimeDelta(Decimal("999_999_999_999_999")),
        ),
        (
            TimeDelta(Decimal("1e15")),
            ht.timedelta(microseconds=15625),  # exact binary fraction
            TimeDelta(Decimal("1_000_000_000_000_000.015_625")),
        ),
    ],
)
def test___ht_timedelta___add___returns_sum(
    left: TimeDelta, right: ht.timedelta, expected: TimeDelta
) -> None:
    assert left + right == expected
    assert right + left == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (
            TimeDelta(Decimal("1e15")),
            ht.timedelta(femtoseconds=314159),
            TimeDelta(Decimal("1_000_000_000_000_000.000_000_000_314_159")),
        ),
    ],
)
def test___ht_timedelta_inexact_result___add___returns_approximate_sum(
    left: TimeDelta, right: ht.timedelta, expected: TimeDelta
) -> None:
    assert (left + right).precision_total_seconds() == pytest.approx(
        expected.precision_total_seconds()
    )
    assert (right + left).precision_total_seconds() == pytest.approx(
        expected.precision_total_seconds()
    )


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (dt.datetime(2025, 1, 1), TimeDelta(0), dt.datetime(2025, 1, 1)),
        (dt.datetime(2025, 1, 1), TimeDelta(2), dt.datetime(2025, 1, 1, 0, 0, 2)),
        (dt.datetime(2025, 1, 1), TimeDelta(-2), dt.datetime(2024, 12, 31, 23, 59, 58)),
    ],
)
def test___dt_datetime___add___returns_sum(
    left: dt.datetime, right: TimeDelta, expected: dt.datetime
) -> None:
    assert left + right == expected
    assert right + left == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (dt.datetime(2025, 1, 1), TimeDelta(1e-6), dt.datetime(2025, 1, 1, 0, 0, 0, 1)),
        (dt.datetime(2025, 1, 1), TimeDelta(-1e-6), dt.datetime(2024, 12, 31, 23, 59, 59, 999999)),
    ],
)
def test___dt_datetime_inexact_result___add___returns_approximate_sum(
    left: dt.datetime, right: TimeDelta, expected: dt.datetime
) -> None:
    assert abs((left + right) - expected) <= _DT_EPSILON
    assert abs((right + left) - expected) <= _DT_EPSILON


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (ht.datetime(2025, 1, 1), TimeDelta(0), ht.datetime(2025, 1, 1)),
        (ht.datetime(2025, 1, 1), TimeDelta(2), ht.datetime(2025, 1, 1, 0, 0, 2)),
        (ht.datetime(2025, 1, 1), TimeDelta(-2), ht.datetime(2024, 12, 31, 23, 59, 58)),
    ],
)
def test___ht_datetime___add___returns_sum(
    left: ht.datetime, right: TimeDelta, expected: ht.datetime
) -> None:
    assert left + right == expected
    assert right + left == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (ht.datetime(2025, 1, 1), TimeDelta(1e-6), ht.datetime(2025, 1, 1, 0, 0, 0, 1)),
        (ht.datetime(2025, 1, 1), TimeDelta(-1e-6), ht.datetime(2024, 12, 31, 23, 59, 59, 999999)),
        (
            ht.datetime(2025, 1, 1),
            TimeDelta(Decimal("1e-15")),
            ht.datetime(2025, 1, 1, 0, 0, 0, 0, 1),
        ),
    ],
)
def test___ht_datetime_inexact_result___add___returns_approximate_sum(
    left: ht.datetime, right: TimeDelta, expected: ht.datetime
) -> None:
    assert abs((left + right) - expected) <= _BT_EPSILON
    assert abs((right + left) - expected) <= _BT_EPSILON


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeDelta(0), TimeDelta(0), TimeDelta(0)),
        (TimeDelta(2), TimeDelta(2), TimeDelta(0)),
        (TimeDelta(2), TimeDelta(-2), TimeDelta(4)),
        (
            TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
            TimeDelta.from_ticks((3 << 124) + (4 << 64) + 5),
            TimeDelta.from_ticks((-2 << 124) + (-2 << 64) + -2),
        ),
        (
            TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
            TimeDelta.from_ticks((-3 << 124) + (-4 << 64) + -5),
            TimeDelta.from_ticks((4 << 124) + (6 << 64) + 8),
        ),
    ],
)
def test___timedeltas___sub___returns_difference(
    left: TimeDelta, right: TimeDelta, expected: TimeDelta
) -> None:
    assert left - right == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeDelta(0), dt.timedelta(seconds=0), TimeDelta(0)),
        (TimeDelta(2), dt.timedelta(seconds=2), TimeDelta(0)),
        (TimeDelta(2), dt.timedelta(seconds=-2), TimeDelta(4)),
        (
            TimeDelta(Decimal("1e15")),
            dt.timedelta(seconds=123),
            TimeDelta(Decimal("999_999_999_999_877")),
        ),
        (
            TimeDelta(Decimal("1e15")),
            -dt.timedelta(seconds=1),
            TimeDelta(Decimal("1_000_000_000_000_001")),
        ),
        (
            TimeDelta(Decimal("1e15")),
            dt.timedelta(microseconds=15625),  # exact binary fraction
            TimeDelta(Decimal("999_999_999_999_999.984_375")),
        ),
    ],
)
def test___dt_timedelta___sub___returns_difference(
    left: TimeDelta, right: dt.timedelta, expected: TimeDelta
) -> None:
    assert left - right == expected
    assert right - left == -expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (
            TimeDelta(Decimal("1e15")),
            dt.timedelta(microseconds=314159),
            TimeDelta(Decimal("999_999_999_999_999.685_841")),
        ),
    ],
)
def test___dt_timedelta_inexact_result___sub___returns_approximate_difference(
    left: TimeDelta, right: dt.timedelta, expected: TimeDelta
) -> None:
    assert (left - right).precision_total_seconds() == pytest.approx(
        expected.precision_total_seconds()
    )
    assert (right - left).precision_total_seconds() == pytest.approx(
        -expected.precision_total_seconds()
    )


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeDelta(0), ht.timedelta(seconds=0), TimeDelta(0)),
        (TimeDelta(2), ht.timedelta(seconds=2), TimeDelta(0)),
        (TimeDelta(2), ht.timedelta(seconds=-2), TimeDelta(4)),
        (
            TimeDelta(Decimal("1e15")),
            ht.timedelta(seconds=123),
            TimeDelta(Decimal("999_999_999_999_877")),
        ),
        (
            TimeDelta(Decimal("1e15")),
            -ht.timedelta(seconds=1),
            TimeDelta(Decimal("1_000_000_000_000_001")),
        ),
        (
            TimeDelta(Decimal("1e15")),
            ht.timedelta(microseconds=15625),  # exact binary fraction
            TimeDelta(Decimal("999_999_999_999_999.984_375")),
        ),
    ],
)
def test___ht_timedelta___sub___returns_difference(
    left: TimeDelta, right: ht.timedelta, expected: TimeDelta
) -> None:
    assert left - right == expected
    assert right - left == -expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (
            TimeDelta(Decimal("1e15")),
            ht.timedelta(femtoseconds=314159),
            TimeDelta(Decimal("999_999_999_999_999.999_999_999_685_800")),
        ),
    ],
)
def test___ht_timedelta_inexact_result___sub___returns_approximate_difference(
    left: TimeDelta, right: ht.timedelta, expected: TimeDelta
) -> None:
    assert (left - right).precision_total_seconds() == pytest.approx(
        expected.precision_total_seconds()
    )
    assert (right - left).precision_total_seconds() == pytest.approx(
        -expected.precision_total_seconds()
    )


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (dt.datetime(2025, 1, 1), TimeDelta(0), dt.datetime(2025, 1, 1)),
        (dt.datetime(2025, 1, 1), TimeDelta(2), dt.datetime(2024, 12, 31, 23, 59, 58)),
        (dt.datetime(2025, 1, 1), TimeDelta(-2), dt.datetime(2025, 1, 1, 0, 0, 2)),
    ],
)
def test___dt_datetime___sub___returns_sum(
    left: dt.datetime, right: TimeDelta, expected: dt.datetime
) -> None:
    assert left - right == expected
    # __sub__(timedelta, datetime) is not supported.


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (dt.datetime(2025, 1, 1), TimeDelta(1e-6), dt.datetime(2024, 12, 31, 23, 59, 59, 999999)),
        (dt.datetime(2025, 1, 1), TimeDelta(-1e-6), dt.datetime(2025, 1, 1, 0, 0, 0, 1)),
    ],
)
def test___dt_datetime_inexact_result___sub___returns_approximate_sum(
    left: dt.datetime, right: TimeDelta, expected: dt.datetime
) -> None:
    assert abs((left - right) - expected) <= _DT_EPSILON
    # __sub__(timedelta, datetime) is not supported.


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (ht.datetime(2025, 1, 1), TimeDelta(0), ht.datetime(2025, 1, 1)),
        (ht.datetime(2025, 1, 1), TimeDelta(2), ht.datetime(2024, 12, 31, 23, 59, 58)),
        (ht.datetime(2025, 1, 1), TimeDelta(-2), ht.datetime(2025, 1, 1, 0, 0, 2)),
    ],
)
def test___ht_datetime___sub___returns_sum(
    left: ht.datetime, right: TimeDelta, expected: ht.datetime
) -> None:
    assert left - right == expected
    # __sub__(timedelta, datetime) is not supported.


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (ht.datetime(2025, 1, 1), TimeDelta(1e-6), ht.datetime(2024, 12, 31, 23, 59, 59, 999999)),
        (ht.datetime(2025, 1, 1), TimeDelta(-1e-6), ht.datetime(2025, 1, 1, 0, 0, 0, 1)),
        (
            ht.datetime(2025, 1, 1),
            TimeDelta(Decimal("1e-15")),
            ht.datetime(2025, 1, 1) - ht.timedelta(femtoseconds=1),
        ),
    ],
)
def test___ht_datetime_inexact_result___sub___returns_approximate_sum(
    left: ht.datetime, right: TimeDelta, expected: ht.datetime
) -> None:
    assert abs((left - right) - expected) <= _BT_EPSILON
    # __sub__(timedelta, datetime) is not supported.


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeDelta(0), 0, TimeDelta(0)),
        (TimeDelta(1), 0, TimeDelta(0)),
        (TimeDelta(1), 1, TimeDelta(1)),
        (TimeDelta(1), -1, TimeDelta(-1)),
        (TimeDelta(100), 200, TimeDelta(20000)),
        (
            TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
            2,
            TimeDelta.from_ticks((2 << 124) + (4 << 64) + 6),
        ),
        (
            TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
            -3,
            TimeDelta.from_ticks((-3 << 124) + (-6 << 64) + -9),
        ),
    ],
)
def test___int___mul___returns_exact_product(
    left: TimeDelta, right: int, expected: TimeDelta
) -> None:
    assert left * right == expected
    assert right * left == expected


# Verify that multiplying by a float does not reduce the TimeValue's precision.
@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeDelta(0), 0.0, TimeDelta(0)),
        (TimeDelta(1), 0.0, TimeDelta(0)),
        (TimeDelta(1), 1.0, TimeDelta(1)),
        (TimeDelta(1), -1.0, TimeDelta(-1)),
        (TimeDelta(100), 200.0, TimeDelta(20000)),
        (
            TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
            1.0,
            TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
        ),
        (
            TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
            -1.0,
            TimeDelta.from_ticks((-1 << 124) + (-2 << 64) + -3),
        ),
    ],
)
def test___exact_float___mul___returns_exact_product(
    left: TimeDelta, right: float, expected: TimeDelta
) -> None:
    assert left * right == expected
    assert right * left == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeDelta(100), 1.23456789, TimeDelta(123.456789)),
        (TimeDelta(1e18), 1.23456789, TimeDelta(1.23456789e18)),
        (TimeDelta(-1e18), 1.23456789, TimeDelta(-1.23456789e18)),
    ],
)
def test___inexact_float___mul___returns_approximate_product(
    left: TimeDelta, right: float, expected: TimeDelta
) -> None:
    assert (left * right).total_seconds() == pytest.approx(expected.total_seconds())
    assert (right * left).total_seconds() == pytest.approx(expected.total_seconds())


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeDelta(0), Decimal("0.0"), TimeDelta(0)),
        (TimeDelta(1), Decimal("0.0"), TimeDelta(0)),
        (TimeDelta(1), Decimal("1.0"), TimeDelta(1)),
        (TimeDelta(1), Decimal("-1.0"), TimeDelta(-1)),
        (TimeDelta(100), Decimal("200.0"), TimeDelta(20000)),
        (
            TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
            Decimal("2.0"),
            TimeDelta.from_ticks((2 << 124) + (4 << 64) + 6),
        ),
        (
            TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
            Decimal("-3.0"),
            TimeDelta.from_ticks((-3 << 124) + (-6 << 64) + -9),
        ),
        (TimeDelta(100), Decimal("1.23456789"), TimeDelta(Decimal("123.456789"))),
        (TimeDelta(1e18), Decimal("1.23456789"), TimeDelta(Decimal("1.23456789e18"))),
        (TimeDelta(-1e18), Decimal("1.23456789"), TimeDelta(Decimal("-1.23456789e18"))),
    ],
)
def test___decimal___mul___returns_exact_product(
    left: TimeDelta, right: float, expected: TimeDelta
) -> None:
    assert left * right == expected
    assert right * left == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeDelta(1), TimeDelta(1), 1),
        (TimeDelta(20000), TimeDelta(200), 100),
        (
            TimeDelta.from_ticks((2 << 124) + (4 << 64) + 6),
            TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
            2,
        ),
        (
            TimeDelta.from_ticks((-3 << 124) + (-6 << 64) + -9),
            TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
            -3,
        ),
    ],
)
def test___timedelta___floordiv___returns_int(
    left: TimeDelta, right: TimeDelta, expected: int
) -> None:
    assert_type(left // right, int)
    assert left // right == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeDelta(1), 1, TimeDelta(1)),
        (TimeDelta(20000), 100, TimeDelta(200)),
        (
            TimeDelta.from_ticks((2 << 124) + (4 << 64) + 6),
            2,
            TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
        ),
        (
            TimeDelta.from_ticks((-3 << 124) + (-6 << 64) + -9),
            -3,
            TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
        ),
    ],
)
def test___int___floordiv___returns_timedelta(
    left: TimeDelta, right: int, expected: TimeDelta
) -> None:
    assert_type(left // right, TimeDelta)
    assert left // right == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeDelta(1), TimeDelta(1), 1.0),
        (TimeDelta(20000), TimeDelta(200), 100.0),
        (TimeDelta(200), TimeDelta(20000), 0.01),
        (
            TimeDelta.from_ticks((2 << 124) + (4 << 64) + 6),
            TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
            2.0,
        ),
        (
            TimeDelta.from_ticks((-3 << 124) + (-6 << 64) + -9),
            TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
            -3.0,
        ),
    ],
)
def test___timedelta___truediv___returns_float(
    left: TimeDelta, right: TimeDelta, expected: int
) -> None:
    assert_type(left / right, float)
    assert left / right == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeDelta(1), 1.0, TimeDelta(1)),
        (TimeDelta(20000), 100.0, TimeDelta(200)),
        (TimeDelta(200), 0.01, TimeDelta(20000)),
        (
            TimeDelta.from_ticks((2 << 124) + (4 << 64) + 6),
            2.0,
            TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
        ),
        (
            TimeDelta.from_ticks((-3 << 124) + (-6 << 64) + -9),
            -3.0,
            TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
        ),
    ],
)
def test___float___truediv___returns_approximate_timedelta(
    left: TimeDelta, right: float, expected: TimeDelta
) -> None:
    assert_type(left / right, TimeDelta)
    assert (left / right).total_seconds() == pytest.approx(expected.total_seconds())


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeDelta(1), TimeDelta(1), TimeDelta(0)),
        (TimeDelta(20042), TimeDelta(200), TimeDelta(42)),
        (
            TimeDelta.from_ticks((2 << 124) + (5 << 64) + 6),
            TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
            TimeDelta.from_ticks(1 << 64),
        ),
        (
            TimeDelta.from_ticks((-3 << 124) + (-6 << 64) + -9),
            TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
            TimeDelta.from_ticks(0),
        ),
    ],
)
def test___timedelta___mod___returns_timedelta(
    left: TimeDelta, right: TimeDelta, expected: TimeDelta
) -> None:
    assert_type(left % right, TimeDelta)
    assert left % right == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeDelta(1), dt.timedelta(seconds=1), TimeDelta(0)),
        (TimeDelta(20042), dt.timedelta(seconds=200), TimeDelta(42)),
    ],
)
def test___dt_timedelta___mod___returns_timedelta(
    left: TimeDelta, right: dt.timedelta, expected: TimeDelta
) -> None:
    assert_type(left % right, TimeDelta)
    assert left % right == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeDelta(1), ht.timedelta(seconds=1), TimeDelta(0)),
        (TimeDelta(20042), ht.timedelta(seconds=200), TimeDelta(42)),
    ],
)
def test___ht_timedelta___mod___returns_timedelta(
    left: TimeDelta, right: ht.timedelta, expected: TimeDelta
) -> None:
    assert_type(left % right, TimeDelta)
    assert left % right == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeDelta(1), TimeDelta(1), (1, TimeDelta(0))),
        (TimeDelta(20042), TimeDelta(200), (100, TimeDelta(42))),
        (
            TimeDelta.from_ticks((2 << 124) + (5 << 64) + 6),
            TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
            (2, TimeDelta.from_ticks(1 << 64)),
        ),
        (
            TimeDelta.from_ticks((-3 << 124) + (-6 << 64) + -9),
            TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
            (-3, TimeDelta.from_ticks(0)),
        ),
    ],
)
def test___timedelta___divmod___returns_int_and_timedelta(
    left: TimeDelta, right: TimeDelta, expected: tuple[int, TimeDelta]
) -> None:
    assert_type(divmod(left, right), tuple[int, TimeDelta])
    assert divmod(left, right) == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeDelta(1), dt.timedelta(seconds=1), (1, TimeDelta(0))),
        (TimeDelta(20042), dt.timedelta(seconds=200), (100, TimeDelta(42))),
    ],
)
def test___dt_timedelta___divmod___returns_int_and_timedelta(
    left: TimeDelta, right: dt.timedelta, expected: tuple[int, TimeDelta]
) -> None:
    assert_type(divmod(left, right), tuple[int, TimeDelta])
    assert divmod(left, right) == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (TimeDelta(1), dt.timedelta(seconds=1), (1, TimeDelta(0))),
        (TimeDelta(20042), dt.timedelta(seconds=200), (100, TimeDelta(42))),
    ],
)
def test___ht_timedelta___divmod___returns_int_and_timedelta(
    left: TimeDelta, right: ht.timedelta, expected: tuple[int, TimeDelta]
) -> None:
    assert_type(divmod(left, right), tuple[int, TimeDelta])
    assert divmod(left, right) == expected


############
# Comparison
############
@pytest.mark.parametrize(
    "left, right",
    [
        (TimeDelta(0), TimeDelta(0)),
        (TimeDelta(1), TimeDelta(1)),
        (TimeDelta(-1), TimeDelta(-1)),
        (
            TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
            TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
        ),
        (
            -TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
            -TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
        ),
        (TimeDelta(1), dt.timedelta(seconds=1)),
        (TimeDelta(1), ht.timedelta(seconds=1)),
        (dt.timedelta(seconds=1), TimeDelta(1)),
        pytest.param(
            ht.timedelta(seconds=1),
            TimeDelta(1),
            marks=pytest.mark.xfail(reason="https://github.com/ni/hightime/issues/60"),
        ),
    ],
)
def test___same_value___comparison___equal(
    left: TimeDelta | dt.timedelta | ht.timedelta, right: TimeDelta | dt.timedelta | ht.timedelta
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
        (TimeDelta(0), TimeDelta(1)),
        (TimeDelta(1), TimeDelta(2)),
        (TimeDelta(-1), TimeDelta(0)),
        (
            TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
            TimeDelta.from_ticks((1 << 124) + (2 << 64) + 4),
        ),
        (
            -TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
            -TimeDelta.from_ticks((1 << 124) + (2 << 64) + 2),
        ),
        (TimeDelta(1), dt.timedelta(seconds=2)),
        (TimeDelta(1), ht.timedelta(seconds=2)),
        (TimeDelta(1), ht.timedelta(seconds=1, femtoseconds=1)),
        (TimeDelta(1), ht.timedelta(seconds=1, yoctoseconds=1)),
        (dt.timedelta(seconds=1), TimeDelta(2)),
        pytest.param(
            ht.timedelta(seconds=1),
            TimeDelta(2),
            marks=pytest.mark.xfail(reason="https://github.com/ni/hightime/issues/60"),
        ),
        pytest.param(
            ht.timedelta(seconds=1) - ht.timedelta(femtoseconds=1),
            TimeDelta(1),
            marks=pytest.mark.xfail(reason="https://github.com/ni/hightime/issues/60"),
        ),
        pytest.param(
            ht.timedelta(seconds=1) - ht.timedelta(yoctoseconds=1),
            TimeDelta(1),
            marks=pytest.mark.xfail(reason="https://github.com/ni/hightime/issues/60"),
        ),
    ],
)
def test___lesser_value___comparison___lesser(
    left: TimeDelta | dt.timedelta | ht.timedelta, right: TimeDelta | dt.timedelta | ht.timedelta
) -> None:
    assert left < right
    assert left <= right
    assert not (left == right)
    assert left != right
    assert not (left > right)
    assert not (left >= right)


###############
# Miscellaneous
###############
@pytest.mark.parametrize(
    "value, expected",
    [
        (TimeDelta(0), False),
        (TimeDelta(1), True),
        (TimeDelta(20042), True),
        (
            TimeDelta.from_ticks((2 << 124) + (5 << 64) + 6),
            True,
        ),
        (TimeDelta.from_ticks((-3 << 124) + (-6 << 64) + -9), True),
    ],
)
def test___timedelta___bool___returns_not_zero(value: TimeDelta, expected: bool) -> None:
    assert bool(value) == expected
    assert (not value) == (not expected)


_VARIOUS_VALUES = [
    TimeDelta(0),
    TimeDelta(2),
    TimeDelta(-2),
    TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
    TimeDelta.from_ticks((-1 << 124) + (-2 << 64) + -3),
    TimeDelta.min,
    TimeDelta.max,
]


def test___various_values___hash___returns_probably_unique_int() -> None:
    hashes = set([hash(x) for x in _VARIOUS_VALUES])
    assert len(hashes) == len(_VARIOUS_VALUES)


@pytest.mark.parametrize(
    "value",
    [
        TimeDelta(0),
        TimeDelta(2),
        TimeDelta(-2),
        TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
        TimeDelta.from_ticks((-1 << 124) + (-2 << 64) + -3),
    ],
)
def test___various_values___copy___makes_copy(value: TimeDelta) -> None:
    new_value = copy.copy(value)
    assert new_value is not value
    assert new_value == value


@pytest.mark.parametrize(
    "value",
    [
        TimeDelta(0),
        TimeDelta(2),
        TimeDelta(-2),
        TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3),
        TimeDelta.from_ticks((-1 << 124) + (-2 << 64) + -3),
    ],
)
def test___various_values___pickle_unpickle___makes_copy(value: TimeDelta) -> None:
    new_value = pickle.loads(pickle.dumps(value))
    assert new_value is not value
    assert new_value == value


def test___timedelta___pickle___references_public_modules() -> None:
    value = TimeDelta(123)
    value_bytes = pickle.dumps(value)

    assert b"nitypes.bintime" in value_bytes
    assert b"nitypes.bintime._timedelta" not in value_bytes


@pytest.mark.parametrize(
    "value, expected",
    [
        (TimeDelta(0), "0:00:00.000000000000000000"),
        (TimeDelta(1), "0:00:01.000000000000000000"),
        (TimeDelta(60), "0:01:00.000000000000000000"),
        (TimeDelta(3600), "1:00:00.000000000000000000"),
        (TimeDelta(86400), "1 day, 0:00:00.000000000000000000"),
        (TimeDelta(86400 * 3), "3 days, 0:00:00.000000000000000000"),
        (TimeDelta(-1), "-1 day, 23:59:59.000000000000000000"),
        (TimeDelta(-60), "-1 day, 23:59:00.000000000000000000"),
        (TimeDelta(-3600), "-1 day, 23:00:00.000000000000000000"),
        (TimeDelta(-86400), "-1 day, 0:00:00.000000000000000000"),
        (TimeDelta(-86400 * 3), "-3 days, 0:00:00.000000000000000000"),
        (TimeDelta(Decimal("0.5")), "0:00:00.500000000000000000"),
        (TimeDelta(Decimal("0.005")), "0:00:00.005000000000000000"),
        (TimeDelta(Decimal("0.000_005")), "0:00:00.000005000000000000"),
        (TimeDelta(Decimal("0.000_000_000_000_005")), "0:00:00.000000000000005000"),
        (TimeDelta(Decimal("0.000_000_000_000_000_005")), "0:00:00.000000000000000005"),
        (TimeDelta(Decimal("-0.5")), "-1 day, 23:59:59.500000000000000000"),
        (TimeDelta(Decimal("-0.005")), "-1 day, 23:59:59.995000000000000000"),
        (TimeDelta(Decimal("-0.000_005")), "-1 day, 23:59:59.999995000000000000"),
        (TimeDelta(Decimal("-0.000_000_000_000_005")), "-1 day, 23:59:59.999999999999995000"),
        (TimeDelta(Decimal("-0.000_000_000_000_000_005")), "-1 day, 23:59:59.999999999999999995"),
    ],
)
def test___various_values___str___looks_ok(value: TimeDelta, expected: str) -> None:
    assert str(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (TimeDelta(0), "nitypes.bintime.TimeDelta(Decimal('0'))"),
        (TimeDelta(1), "nitypes.bintime.TimeDelta(Decimal('1'))"),
        (TimeDelta(60), "nitypes.bintime.TimeDelta(Decimal('60'))"),
        (TimeDelta(3600), "nitypes.bintime.TimeDelta(Decimal('3600'))"),
        (TimeDelta(86400), "nitypes.bintime.TimeDelta(Decimal('86400'))"),
        (TimeDelta(86400 * 3), "nitypes.bintime.TimeDelta(Decimal('259200'))"),
        (TimeDelta(-1), "nitypes.bintime.TimeDelta(Decimal('-1'))"),
        (TimeDelta(-60), "nitypes.bintime.TimeDelta(Decimal('-60'))"),
        (TimeDelta(-3600), "nitypes.bintime.TimeDelta(Decimal('-3600'))"),
        (TimeDelta(-86400), "nitypes.bintime.TimeDelta(Decimal('-86400'))"),
        (TimeDelta(-86400 * 3), "nitypes.bintime.TimeDelta(Decimal('-259200'))"),
        (TimeDelta(Decimal("0.5")), "nitypes.bintime.TimeDelta(Decimal('0.5'))"),
        (
            TimeDelta(Decimal("0.25")),
            "nitypes.bintime.TimeDelta(Decimal('0.25'))",
        ),
        (
            TimeDelta(Decimal("0.125")),
            "nitypes.bintime.TimeDelta(Decimal('0.125'))",
        ),
        (TimeDelta(Decimal("-0.5")), "nitypes.bintime.TimeDelta(Decimal('-0.5'))"),
        (
            TimeDelta(Decimal("-0.25")),
            "nitypes.bintime.TimeDelta(Decimal('-0.25'))",
        ),
        (
            TimeDelta(Decimal("-0.125")),
            "nitypes.bintime.TimeDelta(Decimal('-0.125'))",
        ),
        # The fractional part gets bruised because 0.005 isn't expressible as 2^-N
        (
            TimeDelta(Decimal("0.005")),
            "nitypes.bintime.TimeDelta(Decimal('0.004999999999999999995663191310057982263970188796520233154296875'))",
        ),
        (
            TimeDelta(Decimal("-0.005")),
            "nitypes.bintime.TimeDelta(Decimal('-0.004999999999999999995663191310057982263970188796520233154296875'))",
        ),
    ],
)
def test___various_values___repr___looks_ok(value: TimeDelta, expected: str) -> None:
    assert repr(value) == expected
