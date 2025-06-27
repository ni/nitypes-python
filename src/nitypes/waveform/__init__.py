"""Waveform data types for NI Python APIs."""

from nitypes.waveform._analog import AnalogWaveform
from nitypes.waveform._complex import ComplexWaveform
from nitypes.waveform._digital import (
    DigitalState,
    DigitalWaveform,
    DigitalWaveformFailure,
    DigitalWaveformSignal,
    DigitalWaveformSignalCollection,
    DigitalWaveformTestResult,
)
from nitypes.waveform._exceptions import TimingMismatchError
from nitypes.waveform._extended_properties import (
    ExtendedPropertyDictionary,
    ExtendedPropertyValue,
)
from nitypes.waveform._numeric import NumericWaveform
from nitypes.waveform._scaling import (
    NO_SCALING,
    LinearScaleMode,
    NoneScaleMode,
    ScaleMode,
)
from nitypes.waveform._spectrum import Spectrum
from nitypes.waveform._timing import SampleIntervalMode, Timing
from nitypes.waveform._warnings import ScalingMismatchWarning, TimingMismatchWarning

__all__ = [
    "AnalogWaveform",
    "ComplexWaveform",
    "DigitalState",
    "DigitalWaveform",
    "DigitalWaveformFailure",
    "DigitalWaveformSignal",
    "DigitalWaveformSignalCollection",
    "DigitalWaveformTestResult",
    "ExtendedPropertyDictionary",
    "ExtendedPropertyValue",
    "LinearScaleMode",
    "NO_SCALING",
    "NoneScaleMode",
    "NumericWaveform",
    "SampleIntervalMode",
    "ScaleMode",
    "ScalingMismatchWarning",
    "Spectrum",
    "Timing",
    "TimingMismatchError",
    "TimingMismatchWarning",
]
__doctest_requires__ = {".": ["numpy>=2.0"]}


# Hide that it was defined in a helper file
AnalogWaveform.__module__ = __name__
ComplexWaveform.__module__ = __name__
DigitalState.__module__ = __name__
DigitalWaveform.__module__ = __name__
DigitalWaveformFailure.__module__ = __name__
DigitalWaveformSignal.__module__ = __name__
DigitalWaveformSignalCollection.__module__ = __name__
DigitalWaveformTestResult.__module__ = __name__
ExtendedPropertyDictionary.__module__ = __name__
# ExtendedPropertyValue is a TypeAlias
LinearScaleMode.__module__ = __name__
# NO_SCALING is a constant
NoneScaleMode.__module__ = __name__
NumericWaveform.__module__ = __name__
SampleIntervalMode.__module__ = __name__
ScaleMode.__module__ = __name__
ScalingMismatchWarning.__module__ = __name__
Spectrum.__module__ = __name__
Timing.__module__ = __name__
TimingMismatchError.__module__ = __name__
TimingMismatchWarning.__module__ = __name__
