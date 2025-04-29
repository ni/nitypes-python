"""Waveform data types for NI Python APIs."""

from nitypes.waveform._analog_waveform import AnalogWaveform
from nitypes.waveform._exceptions import TimingMismatchError
from nitypes.waveform._extended_properties import (
    ExtendedPropertyDictionary,
    ExtendedPropertyValue,
)
from nitypes.waveform._scaling import (
    NO_SCALING,
    LinearScaleMode,
    NoneScaleMode,
    ScaleMode,
)
from nitypes.waveform._timing import (
    BaseTiming,
    PrecisionTiming,
    SampleIntervalMode,
    Timing,
)
from nitypes.waveform._warnings import ScalingMismatchWarning, TimingMismatchWarning

__all__ = [
    "AnalogWaveform",
    "BaseTiming",
    "ExtendedPropertyDictionary",
    "ExtendedPropertyValue",
    "LinearScaleMode",
    "NO_SCALING",
    "NoneScaleMode",
    "PrecisionTiming",
    "SampleIntervalMode",
    "ScaleMode",
    "ScalingMismatchWarning",
    "Timing",
    "TimingMismatchError",
    "TimingMismatchWarning",
]

# Hide that it was defined in a helper file
AnalogWaveform.__module__ = __name__
BaseTiming.__module__ = __name__
ExtendedPropertyDictionary.__module__ = __name__
# ExtendedPropertyValue is a TypeAlias
LinearScaleMode.__module__ = __name__
# NO_SCALING is a constant
NoneScaleMode.__module__ = __name__
PrecisionTiming.__module__ = __name__
SampleIntervalMode.__module__ = __name__
ScaleMode.__module__ = __name__
ScalingMismatchWarning.__module__ = __name__
Timing.__module__ = __name__
TimingMismatchError.__module__ = __name__
TimingMismatchWarning.__module__ = __name__
