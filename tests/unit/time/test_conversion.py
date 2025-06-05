from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any

import hightime as ht
import pytest
from typing_extensions import assert_type

import nitypes.bintime as bt
from nitypes.time import convert_datetime, convert_timedelta

_BT_EPSILON = ht.timedelta(yoctoseconds=54210)
_DT_EPSILON = ht.timedelta(microseconds=1)

# Work around https://github.com/ni/hightime/issues/60
_DT_EPSILON_AS_BT = bt.TimeDelta(1e-6)


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


def test___bt_to_dt___convert_datetime___returns_equivalant_dt_datetime() -> None:
    value_in = bt.DateTime.now(dt.timezone.utc)

    value_out = convert_datetime(dt.datetime, value_in)

    assert_type(value_out, dt.datetime)
    assert isinstance(value_out, dt.datetime)
    assert abs(value_out - value_in) <= _DT_EPSILON_AS_BT
    assert value_out.tzinfo is value_in.tzinfo
    assert value_out.fold == 0


def test___bt_to_ht___convert_datetime___returns_equivalant_ht_datetime() -> None:
    value_in = bt.DateTime.now(dt.timezone.utc)

    value_out = convert_datetime(ht.datetime, value_in)

    assert_type(value_out, ht.datetime)
    assert isinstance(value_out, ht.datetime)
    assert value_out == value_in
    assert value_out.tzinfo is value_in.tzinfo
    assert value_out.fold == 0


def test___dt_to_bt___convert_datetime___returns_equivalant_bt_datetime() -> None:
    value_in = dt.datetime.now(dt.timezone.utc)

    value_out = convert_datetime(bt.DateTime, value_in)

    assert_type(value_out, bt.DateTime)
    assert isinstance(value_out, bt.DateTime)
    assert value_out == value_in
    assert value_out.tzinfo is value_in.tzinfo


def test___dt_to_ht___convert_datetime___returns_equivalant_ht_datetime() -> None:
    value_in = dt.datetime.now(dt.timezone.utc)

    value_out = convert_datetime(ht.datetime, value_in)

    assert_type(value_out, ht.datetime)
    assert isinstance(value_out, ht.datetime)
    assert value_out == value_in
    assert value_out.tzinfo is value_in.tzinfo
    assert value_out.fold == value_in.fold


def test___ht_to_bt___convert_datetime___returns_equivalant_bt_datetime() -> None:
    value_in = ht.datetime.now(dt.timezone.utc)

    value_out = convert_datetime(bt.DateTime, value_in)

    assert_type(value_out, bt.DateTime)
    assert isinstance(value_out, bt.DateTime)
    assert abs(value_out - value_in) <= _BT_EPSILON
    assert value_out.tzinfo is value_in.tzinfo


def test___ht_to_dt___convert_datetime___returns_equivalant_dt_datetime() -> None:
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


def test___bt_to_dt___convert_timedelta___returns_equivalant_dt_timedelta() -> None:
    # 1 << 124 ticks is 1 << 60 seconds, which is too big for dt.timedelta.
    value_in = bt.TimeDelta.from_ticks((1 << 92) + (2 << 64) + 3)

    value_out = convert_timedelta(dt.timedelta, value_in)

    assert_type(value_out, dt.timedelta)
    assert isinstance(value_out, dt.timedelta)
    assert abs(value_out - value_in) <= _DT_EPSILON


def test___bt_to_ht___convert_timedelta___returns_equivalant_ht_timedelta() -> None:
    # 1 << 124 ticks is 1 << 60 seconds, which is too big for ht.timedelta too, apparently.
    value_in = bt.TimeDelta.from_ticks((1 << 92) + (2 << 64) + 3)

    value_out = convert_timedelta(ht.timedelta, value_in)

    assert_type(value_out, ht.timedelta)
    assert isinstance(value_out, ht.timedelta)
    assert abs(value_out - value_in) <= _BT_EPSILON


def test___dt_to_bt___convert_timedelta___returns_equivalant_bt_timedelta() -> None:
    value_in = dt.timedelta(days=1, seconds=2, microseconds=3)

    value_out = convert_timedelta(bt.TimeDelta, value_in)

    assert_type(value_out, bt.TimeDelta)
    assert isinstance(value_out, bt.TimeDelta)
    assert abs(value_out - value_in) <= _DT_EPSILON


def test___dt_to_ht___convert_timedelta___returns_equivalant_ht_timedelta() -> None:
    value_in = dt.timedelta(days=1, seconds=2, microseconds=3)

    value_out = convert_timedelta(ht.timedelta, value_in)

    assert_type(value_out, ht.timedelta)
    assert isinstance(value_out, ht.timedelta)
    assert value_out == value_in


def test___ht_to_bt___convert_timedelta___returns_equivalant_bt_timedelta() -> None:
    value_in = ht.timedelta(days=1, seconds=2, microseconds=3)

    value_out = convert_timedelta(bt.TimeDelta, value_in)

    assert_type(value_out, bt.TimeDelta)
    assert isinstance(value_out, bt.TimeDelta)
    assert abs(value_out - value_in) <= _BT_EPSILON


def test___ht_to_dt___convert_timedelta___returns_equivalant_dt_timedelta() -> None:
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
