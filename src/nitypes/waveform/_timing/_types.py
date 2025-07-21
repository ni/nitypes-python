from __future__ import annotations

import datetime as dt

from typing_extensions import TypeVar

from nitypes.time import AnyDateTime, AnyTimeDelta

TTimestamp = TypeVar("TTimestamp", bound=AnyDateTime, default=dt.datetime)
"""Type variable for a timestamp."""

TTimestamp_co = TypeVar(
    "TTimestamp_co",
    bound=AnyDateTime,
    covariant=True,
    default=dt.datetime,
)
"""Covariant type variable for a timestamp."""

TTimeOffset = TypeVar(
    "TTimeOffset",
    bound=AnyTimeDelta,
    default=dt.timedelta,
)
"""Type variable for a time offset."""

TTimeOffset_co = TypeVar(
    "TTimeOffset_co",
    bound=AnyTimeDelta,
    covariant=True,
    default=dt.timedelta,
)
"""Covariant type variable for a time offset."""

TSampleInterval = TypeVar(
    "TSampleInterval",
    bound=AnyTimeDelta,
    default=dt.timedelta,
)
"""Type variable for a sample interval."""

TSampleInterval_co = TypeVar(
    "TSampleInterval_co",
    bound=AnyTimeDelta,
    covariant=True,
    default=dt.timedelta,
)
"""Covariant type variable for a sample interval."""

TOtherTimestamp = TypeVar("TOtherTimestamp", bound=AnyDateTime)
"""Another type variable for a timestamp."""

TOtherTimeOffset = TypeVar("TOtherTimeOffset", bound=AnyTimeDelta)
"""Another type variable for a time offset."""

TOtherSampleInterval = TypeVar("TOtherSampleInterval", bound=AnyTimeDelta)
"""Another type variable for a sample interval."""
