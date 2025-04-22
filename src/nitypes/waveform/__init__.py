"""Waveform data types for NI Python APIs."""

from nitypes.waveform._analog_waveform import AnalogWaveform
from nitypes.waveform._extended_properties import (
    ExtendedPropertyDictionary,
    ExtendedPropertyValue,
)
from nitypes.waveform._scaling._base import ScaleMode
from nitypes.waveform._scaling._linear import LinearScaleMode
from nitypes.waveform._scaling._none import NoneScaleMode
from nitypes.waveform._timing._base import BaseTiming, SampleIntervalMode
from nitypes.waveform._timing._precision import PrecisionTiming
from nitypes.waveform._timing._standard import Timing

__all__ = [
    "AnalogWaveform",
    "BaseTiming",
    "ExtendedPropertyDictionary",
    "ExtendedPropertyValue",
    "LinearScaleMode",
    "NoneScaleMode",
    "PrecisionTiming",
    "SampleIntervalMode",
    "ScaleMode",
    "Timing",
]

# Hide that it was defined in a helper file
AnalogWaveform.__module__ = __name__
BaseTiming.__module__ = __name__
ExtendedPropertyDictionary.__module__ = __name__
# ExtendedPropertyValue is a TypeAlias
LinearScaleMode.__module__ = __name__
NoneScaleMode.__module__ = __name__
PrecisionTiming.__module__ = __name__
SampleIntervalMode.__module__ = __name__
ScaleMode.__module__ = __name__
Timing.__module__ = __name__

ScaleMode.none = NoneScaleMode()
"""A scale mode that does not scale data."""
