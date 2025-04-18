from __future__ import annotations

import datetime as dt
import sys
from typing import Any

import hightime as ht
import pytest

from nitypes.time import convert_datetime, convert_timedelta

if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type


###############################################################################
# convert_datetime
###############################################################################
def test___dt_to_dt___convert_datetime___returns_original_object() -> None:
    value_in = dt.datetime.now(dt.timezone.utc)

    value_out = convert_datetime(dt.datetime, value_in)

    assert_type(value_out, dt.datetime)
    assert isinstance(value_out, dt.datetime)
    assert value_out is value_in


def test___ht_to_ht___convert_datetime___returns_original_object() -> None:
    value_in = ht.datetime.now(dt.timezone.utc)

    value_out = convert_datetime(ht.datetime, value_in)

    assert_type(value_out, ht.datetime)
    assert value_out is value_in


def test___dt_to_ht___convert_datetime___returns_equivalant_ht_datetime() -> None:
    value_in = dt.datetime.now(dt.timezone.utc)

    value_out = convert_datetime(ht.datetime, value_in)

    assert_type(value_out, ht.datetime)
    assert isinstance(value_out, ht.datetime)
    assert value_out == value_in
    assert value_out.tzinfo is value_in.tzinfo
    assert value_out.fold == value_in.fold


def test___ht_to_dt___convert_datetime___returns_equivalant_dt_datetime() -> None:
    value_in = ht.datetime.now(dt.timezone.utc)

    value_out = convert_datetime(dt.datetime, value_in)

    assert_type(value_out, dt.datetime)
    assert isinstance(value_out, dt.datetime)
    assert value_out == value_in
    assert value_out.tzinfo is value_in.tzinfo
    assert value_out.fold == value_in.fold


def test___precise_ht_to_dt___convert_datetime___loses_precision() -> None:
    # ht.datetime.now always sets femtosecond and yoctosecond to 0, so add an offset.
    value_in = ht.datetime.now(dt.timezone.utc) + ht.timedelta(femtoseconds=1, yoctoseconds=2)

    value_out = convert_datetime(dt.datetime, value_in)

    assert_type(value_out, dt.datetime)
    assert isinstance(value_out, dt.datetime)
    assert value_out != value_in
    assert value_out == value_in.replace(femtosecond=0, yoctosecond=0)


@pytest.mark.parametrize("requested_type", [dt.datetime, ht.datetime])
def test___variable_requested_type___convert_datetime___static_return_type_unknown(
    requested_type: type[Any],
) -> None:
    value_in = dt.datetime.now(dt.timezone.utc)

    value_out = convert_datetime(requested_type, value_in)

    assert_type(value_out, Any)
    assert isinstance(value_out, requested_type)


@pytest.mark.parametrize("value_in", [dt.datetime(2025, 1, 1), ht.datetime(2025, 1, 1)])
def test___invalid_requested_type___convert_datetime___raises_type_error(
    value_in: dt.datetime | ht.datetime,
) -> None:
    with pytest.raises(TypeError) as exc:
        _ = convert_datetime(str, value_in)  # type: ignore[type-var]

    assert exc.value.args[0].startswith("The requested type must be a datetime type.")


@pytest.mark.parametrize("requested_type", [dt.datetime, ht.datetime])
def test___invalid_value___convert_datetime___raises_type_error(requested_type: type[Any]) -> None:
    value_in = "10:30 a.m."

    with pytest.raises(TypeError) as exc:
        _ = convert_datetime(requested_type, value_in)  # type: ignore[arg-type]

    assert exc.value.args[0].startswith("The value must be a datetime.")


###############################################################################
# convert_timedelta
###############################################################################
def test___dt_to_dt___convert_timedelta___returns_original_object() -> None:
    value_in = dt.timedelta(days=1, seconds=2, microseconds=3)

    value_out = convert_timedelta(dt.timedelta, value_in)

    assert_type(value_out, dt.timedelta)
    assert isinstance(value_out, dt.timedelta)
    assert value_out is value_in


def test___ht_to_ht___convert_timedelta___returns_original_object() -> None:
    value_in = ht.timedelta(days=1, seconds=2, microseconds=3, femtoseconds=4, yoctoseconds=5)

    value_out = convert_timedelta(ht.timedelta, value_in)

    assert_type(value_out, ht.timedelta)
    assert value_out is value_in


def test___dt_to_ht___convert_timedelta___returns_equivalant_ht_timedelta() -> None:
    value_in = dt.timedelta(days=1, seconds=2, microseconds=3)

    value_out = convert_timedelta(ht.timedelta, value_in)

    assert_type(value_out, ht.timedelta)
    assert isinstance(value_out, ht.timedelta)
    assert value_out == value_in


def test___ht_to_dt___convert_timedelta___returns_equivalant_dt_timedelta() -> None:
    value_in = ht.timedelta(days=1, seconds=2, microseconds=3)

    value_out = convert_timedelta(dt.timedelta, value_in)

    assert_type(value_out, dt.timedelta)
    assert isinstance(value_out, dt.timedelta)
    assert value_out == value_in


def test___precise_ht_to_dt___convert_timedelta___loses_precision() -> None:
    value_in = ht.timedelta(days=1, seconds=2, microseconds=3, femtoseconds=4, yoctoseconds=5)

    value_out = convert_timedelta(dt.timedelta, value_in)

    assert_type(value_out, dt.timedelta)
    assert isinstance(value_out, dt.timedelta)
    assert value_out != value_in
    assert value_out == dt.timedelta(days=1, seconds=2, microseconds=3)


@pytest.mark.parametrize("requested_type", [dt.timedelta, ht.timedelta])
def test___variable_requested_type___convert_timedelta___static_return_type_unknown(
    requested_type: type[Any],
) -> None:
    value_in = dt.timedelta(days=1, seconds=2, microseconds=3)

    value_out = convert_timedelta(requested_type, value_in)

    assert_type(value_out, Any)
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
