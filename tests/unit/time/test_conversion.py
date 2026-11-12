from __future__ import annotations

import datetime as dt
from typing import Any, assert_type
from zoneinfo import ZoneInfo

import hightime as ht
import pytest

import nitypes.bintime as bt
from nitypes.time import convert_datetime, convert_timedelta

_BT_EPSILON = ht.timedelta(yoctoseconds=54210)
_DT_EPSILON = ht.timedelta(microseconds=1)
_NON_UTC_TZ = dt.timezone(dt.timedelta(hours=5, minutes=30))
_NON_UTC_DATETIMES = [
    dt.datetime(1850, 12, 25, 8, 15, 30, 123_456, tzinfo=_NON_UTC_TZ),
    dt.datetime(1904, 1, 1, 0, 30, tzinfo=_NON_UTC_TZ),
    dt.datetime(1950, 6, 15, 12, 0, 1, 654_321, tzinfo=_NON_UTC_TZ),
    dt.datetime(2025, 2, 14, 8, 15, 59, 15625, tzinfo=_NON_UTC_TZ),
]
_NON_UTC_HIGHTIME_DATETIMES = [
    ht.datetime(
        value.year,
        value.month,
        value.day,
        value.hour,
        value.minute,
        value.second,
        value.microsecond,
        tzinfo=value.tzinfo,
    )
    for value in _NON_UTC_DATETIMES
]


###############################################################################
# convert_datetime
###############################################################################
def test___bt_to_bt___convert_datetime___returns_original_object() -> None:
    value_in = bt.DateTime.now(dt.timezone.utc)

    value_out = convert_datetime(bt.DateTime, value_in)

    assert_type(value_out, bt.DateTime)
    assert value_out is value_in


def test___dt_to_dt___convert_datetime___returns_original_object() -> None:
    value_in = dt.datetime.now(dt.timezone.utc)

    value_out = convert_datetime(dt.datetime, value_in)

    assert_type(value_out, dt.datetime)
    assert value_out is value_in


def test___ht_to_ht___convert_datetime___returns_original_object() -> None:
    value_in = ht.datetime.now(dt.timezone.utc)

    value_out = convert_datetime(ht.datetime, value_in)

    assert_type(value_out, ht.datetime)
    assert value_out is value_in


def test___bt_to_dt___convert_datetime___returns_equivalent_dt_datetime() -> None:
    value_in = bt.DateTime.now(dt.timezone.utc)

    value_out = convert_datetime(dt.datetime, value_in)

    assert_type(value_out, dt.datetime)
    assert isinstance(value_out, dt.datetime)
    assert abs(value_out - value_in) <= _DT_EPSILON + _BT_EPSILON
    assert value_out.tzinfo is value_in.tzinfo
    assert value_out.fold == 0


def test___bt_to_ht___convert_datetime___returns_equivalent_ht_datetime() -> None:
    value_in = bt.DateTime.now(dt.timezone.utc)

    value_out = convert_datetime(ht.datetime, value_in)

    assert_type(value_out, ht.datetime)
    assert isinstance(value_out, ht.datetime)
    assert value_out == value_in
    assert value_out.tzinfo is value_in.tzinfo
    assert value_out.fold == 0


def test___dt_to_bt___convert_datetime___returns_equivalent_bt_datetime() -> None:
    value_in = dt.datetime.now(dt.timezone.utc)

    value_out = convert_datetime(bt.DateTime, value_in)

    assert_type(value_out, bt.DateTime)
    assert isinstance(value_out, bt.DateTime)
    assert value_out == value_in
    assert value_out.tzinfo is value_in.tzinfo


def test___dt_to_ht___convert_datetime___returns_equivalent_ht_datetime() -> None:
    value_in = dt.datetime.now(dt.timezone.utc)

    value_out = convert_datetime(ht.datetime, value_in)

    assert_type(value_out, ht.datetime)
    assert isinstance(value_out, ht.datetime)
    assert value_out == value_in
    assert value_out.tzinfo is value_in.tzinfo
    assert value_out.fold == value_in.fold


@pytest.mark.parametrize(
    "month, day, hour, minute, fold, utc_hour",
    [
        (1, 15, 12, 0, 0, 17),
        (7, 15, 12, 0, 0, 16),
        (11, 2, 1, 30, 0, 5),
        (11, 2, 1, 30, 1, 6),
    ],
)
def test___zoneinfo_dt_to_ht___convert_datetime___preserves_instant_and_fold(
    month: int, day: int, hour: int, minute: int, fold: int, utc_hour: int
) -> None:
    zone = ZoneInfo("America/New_York")
    value_in = dt.datetime(2025, month, day, hour, minute, 0, 15625, zone, fold=fold)
    expected_utc = dt.datetime(2025, month, day, utc_hour, minute, 0, 15625, dt.timezone.utc)

    value_out = convert_datetime(ht.datetime, value_in)
    round_trip = convert_datetime(dt.datetime, value_out)

    assert_type(value_out, ht.datetime)
    assert isinstance(value_out, ht.datetime)
    assert value_out.tzinfo is zone
    assert value_out.fold == fold
    assert round_trip == value_in
    assert round_trip.astimezone(dt.timezone.utc) == expected_utc
    assert value_out.femtosecond == 0
    assert value_out.yoctosecond == 0


def test___ht_to_bt___convert_datetime___returns_equivalent_bt_datetime() -> None:
    value_in = ht.datetime.now(dt.timezone.utc)

    value_out = convert_datetime(bt.DateTime, value_in)

    assert_type(value_out, bt.DateTime)
    assert isinstance(value_out, bt.DateTime)
    assert abs(value_out - value_in) <= _BT_EPSILON
    assert value_out.tzinfo is value_in.tzinfo


def test___ht_to_dt___convert_datetime___returns_equivalent_dt_datetime() -> None:
    value_in = ht.datetime.now(dt.timezone.utc)

    value_out = convert_datetime(dt.datetime, value_in)

    assert_type(value_out, dt.datetime)
    assert isinstance(value_out, dt.datetime)
    assert value_out == value_in
    assert value_out.tzinfo is value_in.tzinfo
    assert value_out.fold == value_in.fold


def test___precise_ht_to_bt___convert_datetime___loses_precision() -> None:
    # ht.datetime.now always sets femtosecond and yoctosecond to 0, so add an offset.
    value_in = ht.datetime.now(dt.timezone.utc) + ht.timedelta(femtoseconds=1, yoctoseconds=2)

    value_out = convert_datetime(bt.DateTime, value_in)

    assert value_out != value_in
    assert abs(value_out - value_in) <= _BT_EPSILON


def test___precise_ht_to_dt___convert_datetime___loses_precision() -> None:
    # ht.datetime.now always sets femtosecond and yoctosecond to 0, so add an offset.
    value_in = ht.datetime.now(dt.timezone.utc) + ht.timedelta(femtoseconds=1, yoctoseconds=2)

    value_out = convert_datetime(dt.datetime, value_in)

    assert value_out != value_in
    assert abs(value_out - value_in) <= _DT_EPSILON


# TODO (DST): Extend to a named ZoneInfo zone in winter/summer, transition boundaries, and both
# folds of a repeated hour for all requested types. For datetime <-> hightime, assert fold and
# tzinfo preservation as well as independently expected UTC instants; conversion to bintime
# should distinguish the folds by one hour and conversion back should return UTC with fold=0.
# TODO: The hightime.astimezone() oracle below currently raises TypeError for ZoneInfo inputs.
# Use explicit UTC expectations for those cases (retaining femtosecond/yoctosecond checks).
# Also cover naive input rejection when requesting bintime; the other types may remain naive.
@pytest.mark.parametrize("value_in", [*_NON_UTC_DATETIMES, *_NON_UTC_HIGHTIME_DATETIMES])
@pytest.mark.parametrize("requested_type", [bt.DateTime, dt.datetime, ht.datetime])
def test___non_utc_datetime___convert_datetime___preserves_instant(
    requested_type: type[Any],
    value_in: dt.datetime | ht.datetime,
) -> None:
    value_out = convert_datetime(requested_type, value_in)

    expected_utc = value_in.astimezone(dt.timezone.utc)
    if requested_type is bt.DateTime:
        assert abs(value_out - expected_utc) <= _BT_EPSILON
        assert value_out.tzinfo is dt.timezone.utc
    else:
        assert value_out == value_in
        assert value_out.tzinfo is value_in.tzinfo


@pytest.mark.parametrize("requested_type", [bt.DateTime, dt.datetime, ht.datetime])
def test___variable_requested_type___convert_datetime___static_return_type_unknown(
    requested_type: type[Any],
) -> None:
    value_in = dt.datetime.now(dt.timezone.utc)

    value_out = convert_datetime(requested_type, value_in)

    # Mypy infers Any, which seems right.
    # Pyright infers dt.datetime, which seems wrong.
    assert_type(value_out, Any)  # pyright: ignore[reportAssertTypeFailure]
    assert isinstance(value_out, requested_type)


@pytest.mark.parametrize(
    "value_in",
    [
        bt.DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
        dt.datetime(2025, 1, 1),
        ht.datetime(2025, 1, 1),
    ],
)
def test___invalid_requested_type___convert_datetime___raises_type_error(
    value_in: bt.DateTime | dt.datetime | ht.datetime,
) -> None:
    with pytest.raises(TypeError) as exc:
        _ = convert_datetime(str, value_in)  # type: ignore[type-var]

    assert exc.value.args[0].startswith("The requested type must be a datetime type.")


@pytest.mark.parametrize("requested_type", [bt.DateTime, dt.datetime, ht.datetime])
def test___invalid_value___convert_datetime___raises_type_error(requested_type: type[Any]) -> None:
    value_in = "10:30 a.m."

    with pytest.raises(TypeError) as exc:
        _ = convert_datetime(requested_type, value_in)  # type: ignore[arg-type]

    assert exc.value.args[0].startswith("The value must be a datetime.")


###############################################################################
# convert_timedelta
###############################################################################
def test___bt_to_bt___convert_timedelta___returns_original_object() -> None:
    value_in = bt.TimeDelta.from_ticks((1 << 124) + (2 << 64) + 3)

    value_out = convert_timedelta(bt.TimeDelta, value_in)

    assert_type(value_out, bt.TimeDelta)
    assert value_out is value_in


def test___dt_to_dt___convert_timedelta___returns_original_object() -> None:
    value_in = dt.timedelta(days=1, seconds=2, microseconds=3)

    value_out = convert_timedelta(dt.timedelta, value_in)

    assert_type(value_out, dt.timedelta)
    assert value_out is value_in


def test___ht_to_ht___convert_timedelta___returns_original_object() -> None:
    value_in = ht.timedelta(days=1, seconds=2, microseconds=3, femtoseconds=4, yoctoseconds=5)

    value_out = convert_timedelta(ht.timedelta, value_in)

    assert_type(value_out, ht.timedelta)
    assert value_out is value_in


def test___bt_to_dt___convert_timedelta___returns_equivalent_dt_timedelta() -> None:
    # 1 << 124 ticks is 1 << 60 seconds, which is too big for dt.timedelta.
    value_in = bt.TimeDelta.from_ticks((1 << 92) + (2 << 64) + 3)

    value_out = convert_timedelta(dt.timedelta, value_in)

    assert_type(value_out, dt.timedelta)
    assert isinstance(value_out, dt.timedelta)
    assert abs(value_out - value_in) <= _DT_EPSILON


def test___bt_to_ht___convert_timedelta___returns_equivalent_ht_timedelta() -> None:
    # 1 << 124 ticks is 1 << 60 seconds, which is too big for ht.timedelta too, apparently.
    value_in = bt.TimeDelta.from_ticks((1 << 92) + (2 << 64) + 3)

    value_out = convert_timedelta(ht.timedelta, value_in)

    assert_type(value_out, ht.timedelta)
    assert isinstance(value_out, ht.timedelta)
    assert abs(value_out - value_in) <= _BT_EPSILON


def test___dt_to_bt___convert_timedelta___returns_equivalent_bt_timedelta() -> None:
    value_in = dt.timedelta(days=1, seconds=2, microseconds=3)

    value_out = convert_timedelta(bt.TimeDelta, value_in)

    assert_type(value_out, bt.TimeDelta)
    assert isinstance(value_out, bt.TimeDelta)
    assert abs(value_out - value_in) <= _BT_EPSILON


def test___dt_to_ht___convert_timedelta___returns_equivalent_ht_timedelta() -> None:
    value_in = dt.timedelta(days=1, seconds=2, microseconds=3)

    value_out = convert_timedelta(ht.timedelta, value_in)

    assert_type(value_out, ht.timedelta)
    assert isinstance(value_out, ht.timedelta)
    assert value_out == value_in


def test___ht_to_bt___convert_timedelta___returns_equivalent_bt_timedelta() -> None:
    value_in = ht.timedelta(days=1, seconds=2, microseconds=3)

    value_out = convert_timedelta(bt.TimeDelta, value_in)

    assert_type(value_out, bt.TimeDelta)
    assert isinstance(value_out, bt.TimeDelta)
    assert abs(value_out - value_in) <= _BT_EPSILON


def test___ht_to_dt___convert_timedelta___returns_equivalent_dt_timedelta() -> None:
    value_in = ht.timedelta(days=1, seconds=2, microseconds=3)

    value_out = convert_timedelta(dt.timedelta, value_in)

    assert_type(value_out, dt.timedelta)
    assert isinstance(value_out, dt.timedelta)
    assert value_out == value_in


def test___precise_ht_to_bt___convert_timedelta___loses_precision() -> None:
    value_in = ht.timedelta(days=1, seconds=2, microseconds=3, femtoseconds=4, yoctoseconds=5)

    value_out = convert_timedelta(bt.TimeDelta, value_in)

    assert value_out != value_in
    assert abs(value_out - value_in) <= _BT_EPSILON


def test___precise_ht_to_dt___convert_timedelta___loses_precision() -> None:
    value_in = ht.timedelta(days=1, seconds=2, microseconds=3, femtoseconds=4, yoctoseconds=5)

    value_out = convert_timedelta(dt.timedelta, value_in)

    assert value_out != value_in
    assert abs(value_out - value_in) <= _DT_EPSILON


@pytest.mark.parametrize("requested_type", [dt.timedelta, ht.timedelta])
def test___variable_requested_type___convert_timedelta___static_return_type_unknown(
    requested_type: type[Any],
) -> None:
    value_in = dt.timedelta(days=1, seconds=2, microseconds=3)

    value_out = convert_timedelta(requested_type, value_in)

    # Mypy infers Any, which seems right.
    # Pyright infers dt.datetime, which seems wrong.
    assert_type(value_out, Any)  # pyright: ignore[reportAssertTypeFailure]
    assert isinstance(value_out, requested_type)


@pytest.mark.parametrize("value_in", [dt.timedelta(), ht.timedelta()])
def test___invalid_requested_type___convert_timedelta___raises_type_error(
    value_in: dt.timedelta | ht.timedelta,
) -> None:
    with pytest.raises(TypeError) as exc:
        _ = convert_timedelta(str, value_in)  # type: ignore[type-var]

    assert exc.value.args[0].startswith("The requested type must be a timedelta type.")


@pytest.mark.parametrize("requested_type", [dt.timedelta, ht.timedelta])
def test___invalid_value___convert_timedelta___raises_type_error(requested_type: type[Any]) -> None:
    value_in = "10:30 a.m."

    with pytest.raises(TypeError) as exc:
        _ = convert_timedelta(requested_type, value_in)  # type: ignore[arg-type]

    assert exc.value.args[0].startswith("The value must be a timedelta.")
