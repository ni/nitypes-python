"""Waveform timing data types for NI Python APIs."""

from nitypes.waveform._timing._sample_interval import SampleIntervalMode
from nitypes.waveform._timing._timing import Timing
from nitypes.waveform._timing._types import (
    _AnyDateTime,
    _AnyTimeDelta,
    _TSampleInterval,
    _TSampleInterval_co,
    _TTimeOffset,
    _TTimeOffset_co,
    _TTimestamp,
    _TTimestamp_co,
)

__all__ = [
    "_AnyDateTime",
    "_AnyTimeDelta",
    "_TTimestamp",
    "_TTimestamp_co",
    "_TTimeOffset",
    "_TTimeOffset_co",
    "_TSampleInterval",
    "_TSampleInterval_co",
    "SampleIntervalMode",
    "Timing",
]
