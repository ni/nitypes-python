from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from functools import singledispatch
from typing import Any, TypeVar, Union, cast

import hightime as ht
from typing_extensions import TypeAlias

from nitypes._exceptions import invalid_arg_type, invalid_requested_type
from nitypes.time._conversion import convert_datetime, convert_timedelta
from nitypes.waveform._timing._base import BaseTiming
from nitypes.waveform._timing._precision import PrecisionTiming
from nitypes.waveform._timing._standard import Timing

_AnyTiming: TypeAlias = Union[BaseTiming[Any, Any], Timing, PrecisionTiming]
_TTiming = TypeVar("_TTiming", bound=BaseTiming[Any, Any])


def convert_timing(requested_type: type[_TTiming], value: _AnyTiming, /) -> _TTiming:
    """Convert a waveform timing object to the specified type."""
    convert_func = _CONVERT_TIMING_FOR_TYPE.get(requested_type)
    if convert_func is None:
        raise invalid_requested_type("waveform timing", requested_type)
    return cast(_TTiming, convert_func(value))


@singledispatch
def _convert_to_standard_timing(value: object, /) -> Timing:
    raise invalid_arg_type("value", "waveform timing object", value)


@_convert_to_standard_timing.register
def _(value: Timing, /) -> Timing:
    return value


@_convert_to_standard_timing.register
def _(value: PrecisionTiming, /) -> Timing:
    if value is PrecisionTiming.empty:
        return Timing.empty
    return Timing(
        value._sample_interval_mode,
        None if value._timestamp is None else convert_datetime(dt.datetime, value._timestamp),
        (
            None
            if value._time_offset is None
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
    raise invalid_arg_type("value", "waveform timing object", value)


@_convert_to_precision_timing.register
def _(value: Timing, /) -> PrecisionTiming:
    if value is Timing.empty:
        return PrecisionTiming.empty
    return PrecisionTiming(
        value._sample_interval_mode,
        None if value._timestamp is None else convert_datetime(ht.datetime, value._timestamp),
        (
            None
            if value._time_offset is None
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


_CONVERT_TIMING_FOR_TYPE: dict[type[Any], Callable[[object], object]] = {
    Timing: _convert_to_standard_timing,
    PrecisionTiming: _convert_to_precision_timing,
}
