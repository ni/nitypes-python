"""Waveform data types for NI Python APIs."""

from nitypes.waveform._analog_waveform import AnalogWaveform
from nitypes.waveform._base_timing import BaseWaveformTiming, SampleIntervalMode
from nitypes.waveform._extended_properties import (
    ExtendedPropertyDictionary,
    ExtendedPropertyValue,
)
from nitypes.waveform._precision_timing import PrecisionWaveformTiming
from nitypes.waveform._timing import WaveformTiming

__all__ = [
    "AnalogWaveform",
    "BaseWaveformTiming",
    "ExtendedPropertyDictionary",
    "ExtendedPropertyValue",
    "PrecisionWaveformTiming",
    "SampleIntervalMode",
    "WaveformTiming",
]

# Hide that it was defined in a helper file
AnalogWaveform.__module__ = __name__
BaseWaveformTiming.__module__ = __name__
ExtendedPropertyDictionary.__module__ = __name__
# ExtendedPropertyValue is a TypeAlias
PrecisionWaveformTiming.__module__ = __name__
SampleIntervalMode.__module__ = __name__
WaveformTiming.__module__ = __name__
