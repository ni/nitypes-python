from __future__ import annotations

import datetime as dt
import sys
from functools import singledispatch
from typing import TypeVar, Union

import hightime as ht

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

_AnyDateTime: TypeAlias = Union[dt.datetime, ht.datetime]
_TDateTime = TypeVar("_TDateTime", dt.datetime, ht.datetime)

_AnyTimeDelta: TypeAlias = Union[dt.timedelta, ht.timedelta]
_TTimeDelta = TypeVar("_TTimeDelta", dt.timedelta, ht.timedelta)


def convert_datetime(requested_type: type[_TDateTime], value: _AnyDateTime, /) -> _TDateTime:
    """Convert a datetime object to the specified type."""
    if requested_type is dt.datetime:
        # `if requested_type is T` does not seem to narrow the type of _TDateTime.
        return _convert_to_dt_datetime(value)  # type: ignore[return-value]
    elif requested_type is ht.datetime:
        return _convert_to_ht_datetime(value)
    else:
        raise TypeError(
            "The requested type must be a datetime type.\n" f"Requested type: {requested_type}"
        )


@singledispatch
def _convert_to_dt_datetime(value: object, /) -> dt.datetime:
    raise TypeError("The value must be a datetime.\n" f"Provided value: {value}")


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
    raise TypeError("The value must be a datetime.\n" f"Provided value: {value}")


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


def convert_timedelta(requested_type: type[_TTimeDelta], value: _AnyTimeDelta, /) -> _TTimeDelta:
    """Convert a timedelta object to the specified type."""
    if requested_type is dt.timedelta:
        # `if requested_type is T` does not seem to narrow the type of _TTimeDelta.
        return _convert_to_dt_timedelta(value)  # type: ignore[return-value]
    elif requested_type is ht.timedelta:
        return _convert_to_ht_timedelta(value)
    else:
        raise TypeError(
            "The requested type must be a timedelta type.\n" f"Requested type: {requested_type}"
        )


# @convert_timedelta.register
# def _(requested_type: type[dt.timedelta], value: _TTimeDeltaIn, /) -> dt.timedelta:
#     return _convert_to_dt_timedelta(value)


# @convert_timedelta.register
# def _(requested_type: type[ht.timedelta], value: _TTimeDeltaIn, /) -> ht.timedelta:
#     return _convert_to_ht_timedelta(value)


@singledispatch
def _convert_to_dt_timedelta(value: object, /) -> dt.timedelta:
    raise TypeError("The value must be a timedelta.\n" f"Provided value: {value}")


@_convert_to_dt_timedelta.register
def _(value: dt.timedelta, /) -> dt.timedelta:
    return value


@_convert_to_dt_timedelta.register
def _(value: ht.timedelta, /) -> dt.timedelta:
    return dt.timedelta(value.days, value.seconds, value.microseconds)


@singledispatch
def _convert_to_ht_timedelta(value: object, /) -> ht.timedelta:
    raise TypeError("The value must be a timedelta.\n" f"Provided value: {value}")


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
