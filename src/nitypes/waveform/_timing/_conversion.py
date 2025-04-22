from __future__ import annotations

import datetime as dt
import sys
from functools import singledispatch
from typing import TypeVar, Union

import hightime as ht

from nitypes.time._conversion import convert_datetime, convert_timedelta
from nitypes.waveform._timing._precision import PrecisionTiming
from nitypes.waveform._timing._standard import Timing

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

_AnyTiming: TypeAlias = Union[Timing, PrecisionTiming]
_TTiming = TypeVar("_TTiming", Timing, PrecisionTiming)


def convert_timing(requested_type: type[_TTiming], value: _AnyTiming, /) -> _TTiming:
    """Convert a waveform timing object to the specified type."""
    if requested_type is Timing:
        # `if requested_type is T` does not seem to narrow the type of _TTiming.
        return _convert_to_standard_timing(value)  # type: ignore[return-value]
    elif requested_type is PrecisionTiming:
        return _convert_to_precision_timing(value)  # type: ignore[return-value]
    else:
        raise TypeError(
            "The requested type must be a waveform timing type.\n"
            f"Requested type: {requested_type}"
        )


@singledispatch
def _convert_to_standard_timing(value: object, /) -> Timing:
    raise TypeError("The value must be a waveform timing object.\n" f"Provided value: {value}")


@_convert_to_standard_timing.register
def _(value: Timing, /) -> Timing:
    return value


@_convert_to_standard_timing.register
def _(value: PrecisionTiming, /) -> Timing:
    return Timing(
        value._sample_interval_mode,
        None if value._timestamp is None else convert_datetime(dt.datetime, value._timestamp),
        (
            None
            if value._time_offset == PrecisionTiming._DEFAULT_TIME_OFFSET
            else convert_timedelta(dt.timedelta, value._time_offset)
        ),
        (
            None
            if value._sample_interval is None
            else convert_timedelta(dt.timedelta, value._sample_interval)
        ),
        (
            None
            if value._timestamps is None
            else [convert_datetime(dt.datetime, ts) for ts in value._timestamps]
        ),
    )


@singledispatch
def _convert_to_precision_timing(value: object, /) -> PrecisionTiming:
    raise TypeError("The value must be a waveform timing object.\n" f"Provided value: {value}")


@_convert_to_precision_timing.register
def _(value: Timing, /) -> PrecisionTiming:
    return PrecisionTiming(
        value._sample_interval_mode,
        None if value._timestamp is None else convert_datetime(ht.datetime, value._timestamp),
        (
            None
            if value._time_offset == PrecisionTiming._DEFAULT_TIME_OFFSET
            else convert_timedelta(ht.timedelta, value._time_offset)
        ),
        (
            None
            if value._sample_interval is None
            else convert_timedelta(ht.timedelta, value._sample_interval)
        ),
        (
            None
            if value._timestamps is None
            else [convert_datetime(ht.datetime, ts) for ts in value._timestamps]
        ),
    )


@_convert_to_precision_timing.register
def _(value: PrecisionTiming, /) -> PrecisionTiming:
    return value
