from __future__ import annotations

import copy
import datetime as dt
import pickle

import hightime as ht
import pytest
from typing_extensions import assert_type
from tzlocal import get_localzone

from nitypes.bintime import DateTime, TimeDelta
from nitypes.bintime._timedelta import (
    _BITS_PER_SECOND,
    _FRACTIONAL_SECONDS_MASK,
    WholeAndFractionalSeconds,
)


#############
# Constructor
#############
def test___no_args___construct___returns_epoch() -> None:
    value = DateTime()

    assert_type(value, DateTime)
    assert isinstance(value, DateTime)
    assert value._offset._ticks == 0


def test___dt_datetime___construct___returns_datetime() -> None:
    # 0.015625 is an exact binary fraction.
    value = DateTime(dt.datetime(2025, 2, 14, 8, 15, 59, 15625, dt.timezone.utc))

    assert_type(value, DateTime)
    assert isinstance(value, DateTime)
    assert (
        value.year,
        value.month,
        value.day,
        value.hour,
        value.minute,
        value.second,
        value.microsecond,
    ) == (2025, 2, 14, 8, 15, 59, 15625)


def test___ht_datetime___construct___returns_nearest_datetime() -> None:
    # The microseconds are not exactly representable as a binary fraction, so they are rounded.
    value = DateTime(
        ht.datetime(2025, 2, 14, 8, 15, 59, 15625, 123_456_789, 234_567_890, dt.timezone.utc)
    )

    assert_type(value, DateTime)
    assert isinstance(value, DateTime)
    assert (
        value.year,
        value.month,
        value.day,
        value.hour,
        value.minute,
        value.second,
        value.microsecond,
        value.femtosecond,
        value.yoctosecond,
    ) == (2025, 2, 14, 8, 15, 59, 15625, 123_456_789, 234_569_278)


def test___unit_args___construct___returns_nearest_datetime() -> None:
    # The microseconds are not exactly representable as a binary fraction, so they are rounded.
    value = DateTime(2025, 2, 14, 8, 15, 59, 15625, 123_456_789, 234_567_890, dt.timezone.utc)

    assert_type(value, DateTime)
    assert isinstance(value, DateTime)
    assert (
        value.year,
        value.month,
        value.day,
        value.hour,
        value.minute,
        value.second,
        value.microsecond,
        value.femtosecond,
        value.yoctosecond,
    ) == (2025, 2, 14, 8, 15, 59, 15625, 123_456_789, 234_569_278)


def test___naive_dt_datetime___construct___raises_value_error() -> None:
    with pytest.raises(ValueError) as exc:
        _ = DateTime(dt.datetime(2025, 2, 14, 8, 15, 59, 15625))

    assert exc.value.args[0].startswith("The tzinfo must be datetime.timezone.utc.")


def test___naive_ht_datetime___construct___raises_value_error() -> None:
    with pytest.raises(ValueError) as exc:
        _ = DateTime(ht.datetime(2025, 2, 14, 8, 15, 59, 15625))

    assert exc.value.args[0].startswith("The tzinfo must be datetime.timezone.utc.")


def test___naive_unit_args___construct___raises_value_error() -> None:
    with pytest.raises(ValueError) as exc:
        _ = DateTime(2025, 2, 14, 8, 15, 59, 15625, 123_456_789, 234_567_890)

    assert exc.value.args[0].startswith("The tzinfo must be datetime.timezone.utc.")


def test___local_dt_datetime___construct___raises_value_error() -> None:
    with pytest.raises(ValueError) as exc:
        _ = DateTime(dt.datetime(2025, 2, 14, 8, 15, 59, 15625, get_localzone()))

    assert exc.value.args[0].startswith("The tzinfo must be datetime.timezone.utc.")


def test___local_ht_datetime___construct___raises_value_error() -> None:
    with pytest.raises(ValueError) as exc:
        _ = DateTime(ht.datetime(2025, 2, 14, 8, 15, 59, 15625, tzinfo=get_localzone()))

    assert exc.value.args[0].startswith("The tzinfo must be datetime.timezone.utc.")


def test___local_unit_args___construct___raises_value_error() -> None:
    with pytest.raises(ValueError) as exc:
        _ = DateTime(2025, 2, 14, 8, 15, 59, 15625, 123_456_789, 234_567_890, get_localzone())

    assert exc.value.args[0].startswith("The tzinfo must be datetime.timezone.utc.")


############
# from_ticks
############
def test___int_ticks___from_ticks___returns_time_value() -> None:
    value = DateTime.from_ticks(0x12345678_90ABCDEF_FEDCBA09_87654321)

    assert_type(value, DateTime)
    assert isinstance(value, DateTime)
    assert value._offset._ticks == 0x12345678_90ABCDEF_FEDCBA09_87654321


#############
# from_offset
#############
def test___time_value___from_offset___returns_time_value() -> None:
    value = DateTime.from_offset(TimeDelta.from_ticks(0x12345678_90ABCDEF_FEDCBA09_87654321))

    assert_type(value, DateTime)
    assert isinstance(value, DateTime)
    assert value._offset._ticks == 0x12345678_90ABCDEF_FEDCBA09_87654321


##############################################
# year, month, day, hour, minute, second, etc.
##############################################
@pytest.mark.parametrize(
    "other, expected",
    [
        (
            ht.datetime(dt.MINYEAR, 1, 1, 0, 0, 0, 0, 0, 0, dt.timezone.utc),
            (dt.MINYEAR, 1, 1, 0, 0, 0, 0, 0, 0),
        ),
        (
            ht.datetime(
                1850, 12, 25, 8, 15, 30, 123_456, 234_567_789, 345_567_890, dt.timezone.utc
            ),
            (1850, 12, 25, 8, 15, 30, 123_456, 234_567_789, 345_578_196),
        ),
        (
            ht.datetime(
                1903, 12, 31, 23, 59, 59, 123_456, 234_567_789, 345_567_890, dt.timezone.utc
            ),
            (1903, 12, 31, 23, 59, 59, 123_456, 234_567_789, 345_578_196),
        ),
        (
            ht.datetime(1904, 1, 1, 0, 30, 0, 0, 0, 1_000_000, dt.timezone.utc),
            (1904, 1, 1, 0, 30, 0, 0, 0, 975_781),
        ),
        (
            ht.datetime(2000, 1, 1, 0, 0, 0, 0, 0, 0, dt.timezone.utc),
            (2000, 1, 1, 0, 0, 0, 0, 0, 0),
        ),
        (
            ht.datetime(
                dt.MAXYEAR,
                12,
                31,
                23,
                59,
                59,
                999_999,
                999_999_999,
                999_000_000,  # with 999_999_999, binary fraction rounding pushes us to MAXYEAR + 1
                dt.timezone.utc,
            ),
            (dt.MAXYEAR, 12, 31, 23, 59, 59, 999_999, 999_999_999, 999_024_218),
        ),
    ],
)
def test___various_values___unit_properties___return_unit_values(
    other: ht.datetime, expected: tuple[int, ...]
) -> None:
    value = DateTime(other)
    assert (
        value.year,
        value.month,
        value.day,
        value.hour,
        value.minute,
        value.second,
        value.microsecond,
        value.femtosecond,
        value.yoctosecond,
    ) == expected


###################
# Binary arithmetic
###################
@pytest.mark.parametrize(
    "left, right, expected",
    [
        (
            DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
            TimeDelta(8 * 3600 + 15 * 60 + 30),
            DateTime(2025, 1, 1, 8, 15, 30, tzinfo=dt.timezone.utc),
        ),
    ],
)
def test___time_value___add___returns_datetime(
    left: DateTime, right: TimeDelta, expected: DateTime
) -> None:
    assert_type(left + right, DateTime)
    assert_type(right + left, DateTime)
    assert left + right == expected
    assert right + left == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (
            DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
            dt.timedelta(hours=8, minutes=15, seconds=30),
            DateTime(2025, 1, 1, 8, 15, 30, tzinfo=dt.timezone.utc),
        ),
    ],
)
def test___dt_timedelta___add___returns_datetime(
    left: DateTime, right: dt.timedelta, expected: DateTime
) -> None:
    assert_type(left + right, DateTime)
    assert_type(right + left, DateTime)
    assert left + right == expected
    assert right + left == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (
            DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
            ht.timedelta(hours=8, minutes=15, seconds=30),
            DateTime(2025, 1, 1, 8, 15, 30, tzinfo=dt.timezone.utc),
        ),
    ],
)
def test___ht_timedelta___add___returns_datetime(
    left: DateTime, right: ht.timedelta, expected: DateTime
) -> None:
    assert_type(left + right, DateTime)
    assert_type(right + left, DateTime)
    assert left + right == expected
    assert right + left == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (
            DateTime(2025, 1, 1, 8, 15, 30, tzinfo=dt.timezone.utc),
            TimeDelta(8 * 3600 + 15 * 60 + 30),
            DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
        ),
    ],
)
def test___time_value___sub___returns_datetime(
    left: DateTime, right: TimeDelta, expected: DateTime
) -> None:
    assert_type(left - right, DateTime)
    assert left - right == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (
            DateTime(2025, 1, 1, 8, 15, 30, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
            TimeDelta(8 * 3600 + 15 * 60 + 30),
        ),
    ],
)
def test___datetime___sub___returns_time_value(
    left: DateTime, right: DateTime, expected: TimeDelta
) -> None:
    assert_type(left - right, TimeDelta)
    assert left - right == expected


############
# Comparison
############
@pytest.mark.parametrize(
    "left, right",
    [
        (
            DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
        ),
        (
            DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
            dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc),
        ),
        (
            dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
        ),
        (
            DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
            ht.datetime(2025, 1, 1, tzinfo=dt.timezone.utc),
        ),
        pytest.param(
            ht.datetime(2025, 1, 1, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
            marks=pytest.mark.xfail(reason="https://github.com/ni/hightime/issues/60"),
        ),
    ],
)
def test___same_value___comparison___equal(
    left: DateTime | dt.datetime | ht.datetime, right: DateTime | dt.datetime | ht.datetime
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
        (
            DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
        ),
        (
            DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
            dt.datetime(2025, 1, 2, tzinfo=dt.timezone.utc),
        ),
        (
            dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
        ),
        (
            DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
            ht.datetime(2025, 1, 2, tzinfo=dt.timezone.utc),
        ),
        (
            DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
            ht.datetime(2025, 1, 1, femtosecond=1, tzinfo=dt.timezone.utc),
        ),
        (
            DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
            ht.datetime(2025, 1, 1, yoctosecond=1, tzinfo=dt.timezone.utc),
        ),
        pytest.param(
            ht.datetime(2025, 1, 1, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
            marks=pytest.mark.xfail(reason="https://github.com/ni/hightime/issues/60"),
        ),
        pytest.param(
            ht.datetime(2025, 1, 1, tzinfo=dt.timezone.utc) - ht.timedelta(femtoseconds=1),
            DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
            marks=pytest.mark.xfail(reason="https://github.com/ni/hightime/issues/60"),
        ),
        pytest.param(
            ht.datetime(2025, 1, 1, tzinfo=dt.timezone.utc) - ht.timedelta(yoctoseconds=1),
            DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
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
_VARIOUS_VALUES = [
    DateTime(dt.MINYEAR, 1, 1, 0, 0, 0, 0, 0, 0, dt.timezone.utc),
    DateTime(1850, 12, 25, 8, 15, 30, 123_456, 234_567_789, 345_567_890, dt.timezone.utc),
    DateTime(1903, 12, 31, 23, 59, 59, 123_456, 234_567_789, 345_567_890, dt.timezone.utc),
    DateTime(1904, 1, 1, 0, 30, 0, 0, 0, 1_000_000, dt.timezone.utc),
    DateTime(2000, 1, 1, 0, 0, 0, 0, 0, 0, dt.timezone.utc),
    DateTime(
        dt.MAXYEAR,
        12,
        31,
        23,
        59,
        59,
        999_999,
        999_999_999,
        999_000_000,  # with 999_999_999, binary fraction rounding pushes us to MAXYEAR + 1
        dt.timezone.utc,
    ),
]


def test___various_values___hash___returns_probably_unique_int() -> None:
    hashes = set([hash(x) for x in _VARIOUS_VALUES])
    assert len(hashes) == len(_VARIOUS_VALUES)


@pytest.mark.parametrize("value", _VARIOUS_VALUES)
def test___various_values___copy___makes_copy(value: DateTime) -> None:
    new_value = copy.copy(value)
    assert new_value is not value
    assert new_value == value


@pytest.mark.parametrize("value", _VARIOUS_VALUES)
def test___various_values___pickle_unpickle___makes_copy(value: DateTime) -> None:
    new_value = pickle.loads(pickle.dumps(value))
    assert new_value is not value
    assert new_value == value


def test___time_value___pickle___references_public_modules() -> None:
    value = DateTime()
    value_bytes = pickle.dumps(value)

    assert b"nitypes.bintime" in value_bytes
    assert b"nitypes.bintime._datetime" not in value_bytes
    assert b"nitypes.bintime._time_value" not in value_bytes


@pytest.mark.parametrize(
    "value, expected",
    [
        (
            DateTime.min,
            "0001-01-01 00:00:00+00:00",
        ),
        (
            DateTime(1850, 12, 25, 8, 15, 30, 123_456, 234_567_789, 345_567_890, dt.timezone.utc),
            "1850-12-25 08:15:30.123456234567789345578196+00:00",
        ),
        (
            DateTime(1903, 12, 31, 23, 59, 59, 123_456, 234_567_789, 345_567_890, dt.timezone.utc),
            "1903-12-31 23:59:59.123456234567789345578196+00:00",
        ),
        (
            DateTime(1904, 1, 1, 0, 30, 0, 0, 0, 1_000_000, dt.timezone.utc),
            "1904-01-01 00:30:00.000000000000000000975781+00:00",
        ),
        (
            DateTime(2000, 1, 1, 0, 0, 0, 0, 0, 0, dt.timezone.utc),
            "2000-01-01 00:00:00+00:00",
        ),
        (
            DateTime.max,
            "9999-12-31 23:59:59.999999999999999999945789+00:00",
        ),
    ],
)
def test___various_values___str___looks_ok(value: TimeDelta, expected: str) -> None:
    assert str(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (
            DateTime.min,
            "nitypes.bintime.DateTime(1, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)",
        ),
        (
            DateTime(1850, 12, 25, 8, 15, 30, 123_456, 234_567_789, 345_567_890, dt.timezone.utc),
            "nitypes.bintime.DateTime(1850, 12, 25, 8, 15, 30, 123456, 234567789, 345578196, tzinfo=datetime.timezone.utc)",
        ),
        (
            DateTime(1903, 12, 31, 23, 59, 59, 123_456, 234_567_789, 345_567_890, dt.timezone.utc),
            "nitypes.bintime.DateTime(1903, 12, 31, 23, 59, 59, 123456, 234567789, 345578196, tzinfo=datetime.timezone.utc)",
        ),
        (
            DateTime(1904, 1, 1, 0, 30, 0, 0, 0, 1_000_000, dt.timezone.utc),
            "nitypes.bintime.DateTime(1904, 1, 1, 0, 30, 0, 0, 0, 975781, tzinfo=datetime.timezone.utc)",
        ),
        (
            DateTime(2000, 1, 1, 0, 0, 0, 0, 0, 0, dt.timezone.utc),
            "nitypes.bintime.DateTime(2000, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)",
        ),
        (
            DateTime.max,
            "nitypes.bintime.DateTime(9999, 12, 31, 23, 59, 59, 999999, 999999999, 999945789, tzinfo=datetime.timezone.utc)",
        ),
    ],
)
def test___various_values___repr___looks_ok(value: TimeDelta, expected: str) -> None:
    assert repr(value) == expected


@pytest.mark.parametrize(
    "seconds",
    [
        0,
        2,
        -2,
        1.5,
        1234.5678,
    ],
)
def test___various_values___get_ticks___returns_correct_value(seconds: float) -> None:
    ticks = TimeDelta._to_ticks(seconds)
    value = DateTime.from_ticks(ticks)

    assert value.ticks == ticks


@pytest.mark.parametrize(
    "seconds",
    [
        0,
        2,
        -2,
        1.5,
        1234.5678,
    ],
)
def test___various_values___get_whole_fract_sec___returns_correct_values(seconds: float) -> None:
    ticks = TimeDelta._to_ticks(seconds)
    value = DateTime.from_ticks(ticks)

    whole_seconds, fractional_seconds = value.to_tuple()
    assert whole_seconds == ticks >> _BITS_PER_SECOND
    assert fractional_seconds == ticks & _FRACTIONAL_SECONDS_MASK


@pytest.mark.parametrize(
    "seconds",
    [
        0,
        2,
        -2,
        1.5,
        1234.5678,
    ],
)
def test___various_values___from_tuple___datetime_correct(seconds: float) -> None:
    ticks = TimeDelta._to_ticks(seconds)
    whole_seconds = ticks >> _BITS_PER_SECOND
    fractional_seconds = ticks & _FRACTIONAL_SECONDS_MASK

    value = DateTime.from_tuple(WholeAndFractionalSeconds(whole_seconds, fractional_seconds))

    assert value.ticks == ticks
    assert value.to_tuple() == (whole_seconds, fractional_seconds)
