from __future__ import annotations

import datetime as dt
from typing import Union

import hightime as ht
from typing_extensions import TypeAlias, TypeVar

import nitypes.bintime as bt

__all__ = [
    "_AnyDateTime",
    "_AnyTimeDelta",
    "_ANY_DATETIME_TUPLE",
    "_ANY_TIMEDELTA_TUPLE",
    "_TTimestamp",
    "_TTimestamp_co",
    "_TTimeOffset",
    "_TTimeOffset_co",
    "_TSampleInterval",
    "_TSampleInterval_co",
]

_AnyDateTime: TypeAlias = Union[bt.DateTime, dt.datetime, ht.datetime]
_AnyTimeDelta: TypeAlias = Union[bt.TimeDelta, dt.timedelta, ht.timedelta]

_ANY_DATETIME_TUPLE = (bt.DateTime, dt.datetime, ht.datetime)
_ANY_TIMEDELTA_TUPLE = (bt.TimeDelta, dt.timedelta, ht.timedelta)

_TTimestamp = TypeVar(
    "_TTimestamp", bt.DateTime, dt.datetime, ht.datetime, _AnyDateTime, default=dt.datetime
)
_TTimestamp_co = TypeVar(
    "_TTimestamp_co",
    bt.DateTime,
    dt.datetime,
    ht.datetime,
    _AnyDateTime,
    covariant=True,
    default=dt.datetime,
)

_TTimeOffset = TypeVar(
    "_TTimeOffset",
    bt.TimeDelta,
    dt.timedelta,
    ht.timedelta,
    _AnyTimeDelta,
    default=dt.timedelta,
)
_TTimeOffset_co = TypeVar(
    "_TTimeOffset_co",
    bt.TimeDelta,
    dt.timedelta,
    ht.timedelta,
    _AnyTimeDelta,
    covariant=True,
    default=dt.timedelta,
)

_TSampleInterval = TypeVar(
    "_TSampleInterval",
    bt.TimeDelta,
    dt.timedelta,
    ht.timedelta,
    _AnyTimeDelta,
    default=dt.timedelta,
)
_TSampleInterval_co = TypeVar(
    "_TSampleInterval_co",
    bt.TimeDelta,
    dt.timedelta,
    ht.timedelta,
    _AnyTimeDelta,
    covariant=True,
    default=dt.timedelta,
)
