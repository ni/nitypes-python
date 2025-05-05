from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from functools import singledispatch
from typing import Any, TypeVar, Union, cast

import hightime as ht
from typing_extensions import TypeAlias

from nitypes._exceptions import invalid_arg_type, invalid_requested_type

_AnyDateTime: TypeAlias = Union[dt.datetime, ht.datetime]
_TDateTime = TypeVar("_TDateTime", dt.datetime, ht.datetime)

_AnyTimeDelta: TypeAlias = Union[dt.timedelta, ht.timedelta]
_TTimeDelta = TypeVar("_TTimeDelta", dt.timedelta, ht.timedelta)


def convert_datetime(requested_type: type[_TDateTime], value: _AnyDateTime, /) -> _TDateTime:
    """Convert a datetime object to the specified type."""
    convert_func = _CONVERT_DATETIME_FOR_TYPE.get(requested_type)
    if convert_func is None:
        raise invalid_requested_type("datetime", requested_type)
    return cast(_TDateTime, convert_func(value))


@singledispatch
def _convert_to_dt_datetime(value: object, /) -> dt.datetime:
    raise invalid_arg_type("value", "datetime", value)


@_convert_to_dt_datetime.register
def _(value: dt.datetime, /) -> dt.datetime:
    return value


@_convert_to_dt_datetime.register
def _(value: ht.datetime, /) -> dt.datetime:
    return dt.datetime(
        value.year,
        value.month,
        value.day,
        value.hour,
        value.minute,
        value.second,
        value.microsecond,
        value.tzinfo,
        fold=value.fold,
    )


@singledispatch
def _convert_to_ht_datetime(value: object, /) -> ht.datetime:
    raise invalid_arg_type("value", "datetime", value)


@_convert_to_ht_datetime.register
def _(value: dt.datetime, /) -> ht.datetime:
    return ht.datetime(
        value.year,
        value.month,
        value.day,
        value.hour,
        value.minute,
        value.second,
        value.microsecond,
        value.tzinfo,
        fold=value.fold,
    )


@_convert_to_ht_datetime.register
def _(value: ht.datetime, /) -> ht.datetime:
    return value


_CONVERT_DATETIME_FOR_TYPE: dict[type[Any], Callable[[object], object]] = {
    dt.datetime: _convert_to_dt_datetime,
    ht.datetime: _convert_to_ht_datetime,
}


def convert_timedelta(requested_type: type[_TTimeDelta], value: _AnyTimeDelta, /) -> _TTimeDelta:
    """Convert a timedelta object to the specified type."""
    convert_func = _CONVERT_TIMEDELTA_FOR_TYPE.get(requested_type)
    if convert_func is None:
        raise invalid_requested_type("timedelta", requested_type)
    return cast(_TTimeDelta, convert_func(value))


@singledispatch
def _convert_to_dt_timedelta(value: object, /) -> dt.timedelta:
    raise invalid_arg_type("value", "timedelta", value)


@_convert_to_dt_timedelta.register
def _(value: dt.timedelta, /) -> dt.timedelta:
    return value


@_convert_to_dt_timedelta.register
def _(value: ht.timedelta, /) -> dt.timedelta:
    return dt.timedelta(value.days, value.seconds, value.microseconds)


@singledispatch
def _convert_to_ht_timedelta(value: object, /) -> ht.timedelta:
    raise invalid_arg_type("value", "timedelta", value)


@_convert_to_ht_timedelta.register
def _(value: dt.timedelta, /) -> ht.timedelta:
    return ht.timedelta(
        value.days,
        value.seconds,
        value.microseconds,
    )


@_convert_to_ht_timedelta.register
def _(value: ht.timedelta, /) -> ht.timedelta:
    return value


_CONVERT_TIMEDELTA_FOR_TYPE: dict[type[Any], Callable[[object], object]] = {
    dt.timedelta: _convert_to_dt_timedelta,
    ht.timedelta: _convert_to_ht_timedelta,
}
