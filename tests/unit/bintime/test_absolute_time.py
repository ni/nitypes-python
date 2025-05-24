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
from tzlocal import get_localzone

from nitypes.bintime import AbsoluteTime, TimeValue


#############
# Constructor
#############
def test___no_args___construct___returns_epoch() -> None:
    value = AbsoluteTime()

    assert_type(value, AbsoluteTime)
    assert isinstance(value, AbsoluteTime)
    assert value._offset._ticks == 0


def test___dt_datetime___construct___returns_absolute_time() -> None:
    # 0.015625 is an exact binary fraction.
    value = AbsoluteTime(dt.datetime(2025, 2, 14, 8, 15, 59, 15625, dt.timezone.utc))

    assert_type(value, AbsoluteTime)
    assert isinstance(value, AbsoluteTime)
    assert (
        value.year,
        value.month,
        value.day,
        value.hour,
        value.minute,
        value.second,
        value.microsecond,
    ) == (2025, 2, 14, 8, 15, 59, 15625)


def test___ht_datetime___construct___returns_nearest_absolute_time() -> None:
    # The microseconds are not exactly representable as a binary fraction, so they are rounded.
    value = AbsoluteTime(
        ht.datetime(2025, 2, 14, 8, 15, 59, 15625, 123_456_789, 234_567_890, dt.timezone.utc)
    )

    assert_type(value, AbsoluteTime)
    assert isinstance(value, AbsoluteTime)
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
        _ = AbsoluteTime(dt.datetime(2025, 2, 14, 8, 15, 59, 15625))

    assert exc.value.args[0].startswith("The value.tzinfo must be a datetime.timezone.utc.")


def test___naive_ht_datetime___construct___raises_value_error() -> None:
    with pytest.raises(ValueError) as exc:
        _ = AbsoluteTime(ht.datetime(2025, 2, 14, 8, 15, 59, 15625))

    assert exc.value.args[0].startswith("The value.tzinfo must be a datetime.timezone.utc.")


def test___local_dt_datetime___construct___raises_value_error() -> None:
    with pytest.raises(ValueError) as exc:
        _ = AbsoluteTime(dt.datetime(2025, 2, 14, 8, 15, 59, 15625, get_localzone()))

    assert exc.value.args[0].startswith("The value.tzinfo must be a datetime.timezone.utc.")


def test___local_ht_datetime___construct___raises_value_error() -> None:
    with pytest.raises(ValueError) as exc:
        _ = AbsoluteTime(ht.datetime(2025, 2, 14, 8, 15, 59, 15625, tzinfo=get_localzone()))

    assert exc.value.args[0].startswith("The value.tzinfo must be a datetime.timezone.utc.")


############
# from_ticks
############
def test___int_ticks___from_ticks___returns_time_value() -> None:
    value = AbsoluteTime.from_ticks(0x12345678_90ABCDEF_FEDCBA09_87654321)

    assert_type(value, AbsoluteTime)
    assert isinstance(value, AbsoluteTime)
    assert value._offset._ticks == 0x12345678_90ABCDEF_FEDCBA09_87654321


#############
# from_offset
#############
def test___time_value___from_offset___returns_time_value() -> None:
    value = AbsoluteTime.from_offset(TimeValue.from_ticks(0x12345678_90ABCDEF_FEDCBA09_87654321))

    assert_type(value, AbsoluteTime)
    assert isinstance(value, AbsoluteTime)
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
    value = AbsoluteTime(other)
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
            AbsoluteTime(dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)),
            TimeValue(8 * 3600 + 15 * 60 + 30),
            AbsoluteTime(dt.datetime(2025, 1, 1, 8, 15, 30, tzinfo=dt.timezone.utc)),
        ),
    ],
)
def test___time_value___add___returns_absolute_time(
    left: AbsoluteTime, right: TimeValue, expected: AbsoluteTime
) -> None:
    assert left + right == expected
    assert right + left == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (
            AbsoluteTime(dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)),
            dt.timedelta(hours=8, minutes=15, seconds=30),
            AbsoluteTime(dt.datetime(2025, 1, 1, 8, 15, 30, tzinfo=dt.timezone.utc)),
        ),
    ],
)
def test___dt_timedelta___add___returns_absolute_time(
    left: AbsoluteTime, right: dt.timedelta, expected: AbsoluteTime
) -> None:
    assert left + right == expected
    assert right + left == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (
            AbsoluteTime(dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)),
            ht.timedelta(hours=8, minutes=15, seconds=30),
            AbsoluteTime(dt.datetime(2025, 1, 1, 8, 15, 30, tzinfo=dt.timezone.utc)),
        ),
    ],
)
def test___ht_timedelta___add___returns_absolute_time(
    left: AbsoluteTime, right: ht.timedelta, expected: AbsoluteTime
) -> None:
    assert left + right == expected
    assert right + left == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (
            AbsoluteTime(dt.datetime(2025, 1, 1, 8, 15, 30, tzinfo=dt.timezone.utc)),
            TimeValue(8 * 3600 + 15 * 60 + 30),
            AbsoluteTime(dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)),
        ),
    ],
)
def test___time_value___sub___returns_absolute_time(
    left: AbsoluteTime, right: TimeValue, expected: AbsoluteTime
) -> None:
    assert left - right == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (
            AbsoluteTime(dt.datetime(2025, 1, 1, 8, 15, 30, tzinfo=dt.timezone.utc)),
            AbsoluteTime(dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)),
            TimeValue(8 * 3600 + 15 * 60 + 30),
        ),
    ],
)
def test___absolute_time___sub___returns_time_value(
    left: AbsoluteTime, right: AbsoluteTime, expected: TimeValue
) -> None:
    assert left - right == expected


############
# Comparison
############
@pytest.mark.parametrize(
    "left, right",
    [
        (
            AbsoluteTime(dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)),
            AbsoluteTime(dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)),
        ),
        (
            AbsoluteTime(dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)),
            dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc),
        ),
        (
            dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc),
            AbsoluteTime(dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)),
        ),
        (
            AbsoluteTime(dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)),
            ht.datetime(2025, 1, 1, tzinfo=dt.timezone.utc),
        ),
        pytest.param(
            ht.datetime(2025, 1, 1, tzinfo=dt.timezone.utc),
            AbsoluteTime(dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)),
            marks=pytest.mark.xfail(reason="https://github.com/ni/hightime/issues/60"),
        ),
    ],
)
def test___same_value___comparison___equal(
    left: AbsoluteTime | dt.datetime | ht.datetime, right: AbsoluteTime | dt.datetime | ht.datetime
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
            AbsoluteTime(dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)),
            AbsoluteTime(dt.datetime(2025, 1, 2, tzinfo=dt.timezone.utc)),
        ),
        (
            AbsoluteTime(dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)),
            dt.datetime(2025, 1, 2, tzinfo=dt.timezone.utc),
        ),
        (
            dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc),
            AbsoluteTime(dt.datetime(2025, 1, 2, tzinfo=dt.timezone.utc)),
        ),
        (
            AbsoluteTime(dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)),
            ht.datetime(2025, 1, 2, tzinfo=dt.timezone.utc),
        ),
        pytest.param(
            ht.datetime(2025, 1, 1, tzinfo=dt.timezone.utc),
            AbsoluteTime(dt.datetime(2025, 1, 2, tzinfo=dt.timezone.utc)),
            marks=pytest.mark.xfail(reason="https://github.com/ni/hightime/issues/60"),
        ),
    ],
)
def test___lesser_value___comparison___lesser(
    left: TimeValue | dt.timedelta | ht.timedelta, right: TimeValue | dt.timedelta | ht.timedelta
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
    AbsoluteTime(ht.datetime(dt.MINYEAR, 1, 1, 0, 0, 0, 0, 0, 0, dt.timezone.utc)),
    AbsoluteTime(
        ht.datetime(1850, 12, 25, 8, 15, 30, 123_456, 234_567_789, 345_567_890, dt.timezone.utc)
    ),
    AbsoluteTime(
        ht.datetime(1903, 12, 31, 23, 59, 59, 123_456, 234_567_789, 345_567_890, dt.timezone.utc)
    ),
    AbsoluteTime(ht.datetime(1904, 1, 1, 0, 30, 0, 0, 0, 1_000_000, dt.timezone.utc)),
    AbsoluteTime(ht.datetime(2000, 1, 1, 0, 0, 0, 0, 0, 0, dt.timezone.utc)),
    AbsoluteTime(
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
        )
    ),
]


def test___various_values___hash___returns_probably_unique_int() -> None:
    hashes = set([hash(x) for x in _VARIOUS_VALUES])
    assert len(hashes) == len(_VARIOUS_VALUES)


@pytest.mark.parametrize("value", _VARIOUS_VALUES)
def test___various_values___copy___makes_copy(value: AbsoluteTime) -> None:
    new_value = copy.copy(value)
    assert new_value is not value
    assert new_value == value


@pytest.mark.parametrize("value", _VARIOUS_VALUES)
def test___various_values___pickle_unpickle___makes_copy(value: AbsoluteTime) -> None:
    new_value = pickle.loads(pickle.dumps(value))
    assert new_value is not value
    assert new_value == value


def test___time_value___pickle___references_public_modules() -> None:
    value = AbsoluteTime()
    value_bytes = pickle.dumps(value)

    assert b"nitypes.bintime" in value_bytes
    assert b"nitypes.bintime._absolute_time" not in value_bytes
    assert b"nitypes.bintime._time_value" not in value_bytes


@pytest.mark.parametrize(
    "value, expected",
    [
        (
            AbsoluteTime.min,
            "0001-01-01 00:00:00+00:00",
        ),
        (
            AbsoluteTime(
                ht.datetime(
                    1850, 12, 25, 8, 15, 30, 123_456, 234_567_789, 345_567_890, dt.timezone.utc
                )
            ),
            "1850-12-25 08:15:30.123456234567789345578196+00:00",
        ),
        (
            AbsoluteTime(
                ht.datetime(
                    1903, 12, 31, 23, 59, 59, 123_456, 234_567_789, 345_567_890, dt.timezone.utc
                )
            ),
            "1903-12-31 23:59:59.123456234567789345578196+00:00",
        ),
        (
            AbsoluteTime(ht.datetime(1904, 1, 1, 0, 30, 0, 0, 0, 1_000_000, dt.timezone.utc)),
            "1904-01-01 00:30:00.000000000000000000975781+00:00",
        ),
        (
            AbsoluteTime(ht.datetime(2000, 1, 1, 0, 0, 0, 0, 0, 0, dt.timezone.utc)),
            "2000-01-01 00:00:00+00:00",
        ),
        (
            AbsoluteTime.max,
            "9999-12-31 23:59:59.999999999999999999945789+00:00",
        ),
    ],
)
def test___various_values___str___looks_ok(value: TimeValue, expected: str) -> None:
    assert str(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (
            AbsoluteTime.min,
            "nitypes.bintime.AbsoluteTime(hightime.datetime(1, 1, 1, 0, 0, tzinfo=datetime.timezone.utc))",
        ),
        (
            AbsoluteTime(
                ht.datetime(
                    1850, 12, 25, 8, 15, 30, 123_456, 234_567_789, 345_567_890, dt.timezone.utc
                )
            ),
            "nitypes.bintime.AbsoluteTime(hightime.datetime(1850, 12, 25, 8, 15, 30, 123456, 234567789, 345578196, tzinfo=datetime.timezone.utc))",
        ),
        (
            AbsoluteTime(
                ht.datetime(
                    1903, 12, 31, 23, 59, 59, 123_456, 234_567_789, 345_567_890, dt.timezone.utc
                )
            ),
            "nitypes.bintime.AbsoluteTime(hightime.datetime(1903, 12, 31, 23, 59, 59, 123456, 234567789, 345578196, tzinfo=datetime.timezone.utc))",
        ),
        (
            AbsoluteTime(ht.datetime(1904, 1, 1, 0, 30, 0, 0, 0, 1_000_000, dt.timezone.utc)),
            "nitypes.bintime.AbsoluteTime(hightime.datetime(1904, 1, 1, 0, 30, 0, 0, 0, 975781, tzinfo=datetime.timezone.utc))",
        ),
        (
            AbsoluteTime(ht.datetime(2000, 1, 1, 0, 0, 0, 0, 0, 0, dt.timezone.utc)),
            "nitypes.bintime.AbsoluteTime(hightime.datetime(2000, 1, 1, 0, 0, tzinfo=datetime.timezone.utc))",
        ),
        (
            AbsoluteTime.max,
            "nitypes.bintime.AbsoluteTime(hightime.datetime(9999, 12, 31, 23, 59, 59, 999999, 999999999, 999945789, tzinfo=datetime.timezone.utc))",
        ),
    ],
)
def test___various_values___repr___looks_ok(value: TimeValue, expected: str) -> None:
    assert repr(value) == expected
