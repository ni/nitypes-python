"""Waveform data types for NI Python APIs.

Analog Waveforms
================

An analog waveform represents a single analog signal with timing information and extended
properties such as units.

Constructing analog waveforms
-----------------------------

To construct an analog waveform, use the :any:`AnalogWaveform` class:

>>> AnalogWaveform()
nitypes.waveform.AnalogWaveform(0)
>>> AnalogWaveform(5)
nitypes.waveform.AnalogWaveform(5, raw_data=array([0., 0., 0., 0., 0.]))

To construct an analog waveform from a NumPy array, use the :any:`AnalogWaveform.from_array_1d`
method.

>>> import numpy as np
>>> AnalogWaveform.from_array_1d(np.array([1.0, 2.0, 3.0]))
nitypes.waveform.AnalogWaveform(3, raw_data=array([1., 2., 3.]))

You can also use :any:`AnalogWaveform.from_array_1d` to construct an analog waveform from a
sequence, such as a list. In this case, you must specify the NumPy data type.

>>> AnalogWaveform.from_array_1d([1.0, 2.0, 3.0], np.float64)
nitypes.waveform.AnalogWaveform(3, raw_data=array([1., 2., 3.]))

The 2D version, :any:`AnalogWaveform.from_array_2d`, constructs a list of waveforms, one for each
row of data in the array or nested sequence.

>>> nested_list = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
>>> AnalogWaveform.from_array_2d(nested_list, np.float64)  # doctest: +NORMALIZE_WHITESPACE
[nitypes.waveform.AnalogWaveform(3, raw_data=array([1., 2., 3.])),
 nitypes.waveform.AnalogWaveform(3, raw_data=array([4., 5., 6.]))]

Scaling analog data
-------------------

By default, analog waveforms contain floating point data in :any:`numpy.float64` format, but they
can also be used to scale raw integer data to floating-point:

>>> scale_mode = LinearScaleMode(gain=2.0, offset=0.5)
>>> wfm = AnalogWaveform.from_array_1d([1, 2, 3], np.int32, scale_mode=scale_mode)
>>> wfm  # doctest: +NORMALIZE_WHITESPACE
nitypes.waveform.AnalogWaveform(3, int32, raw_data=array([1, 2, 3], dtype=int32),
    scale_mode=nitypes.waveform.LinearScaleMode(2.0, 0.5))
>>> wfm.raw_data
array([1, 2, 3], dtype=int32)
>>> wfm.scaled_data
array([2.5, 4.5, 6.5])

Complex Waveforms
=================

A complex waveform represents a single complex-number signal, such as I/Q data, with timing
information and extended properties such as units.

Constructing complex waveforms
------------------------------

To construct a complex waveform, use the :any:`ComplexWaveform` class:

>>> ComplexWaveform.from_array_1d([1 + 2j, 3 + 4j], np.complex128)
nitypes.waveform.ComplexWaveform(2, raw_data=array([1.+2.j, 3.+4.j]))

Scaling complex-number data
---------------------------

Complex waveforms support scaling raw integer data to floating-point. Python and NumPy do not
have native support for complex integers, so this uses the :any:`ComplexInt32DType` structured data
type.

>>> from nitypes.complex import ComplexInt32DType
>>> wfm = ComplexWaveform.from_array_1d([(1, 2), (3, 4)], ComplexInt32DType, scale_mode=scale_mode)
>>> wfm  # doctest: +NORMALIZE_WHITESPACE
nitypes.waveform.ComplexWaveform(2, void32, raw_data=array([(1, 2), (3, 4)],
    dtype=[('real', '<i2'), ('imag', '<i2')]),
    scale_mode=nitypes.waveform.LinearScaleMode(2.0, 0.5))
>>> wfm.raw_data
array([(1, 2), (3, 4)], dtype=[('real', '<i2'), ('imag', '<i2')])
>>> wfm.scaled_data
array([2.5+4.j, 6.5+8.j])

Frequency Spectrums
===================

A frequency spectrum represents an analog signal with frequency information and extended properties
such as units.

Constructing spectrums
----------------------

To construct a spectrum, use the :any:`Spectrum` class:

>>> Spectrum.from_array_1d([1, 2, 3], np.float64, start_frequency=100, frequency_increment=10)  # doctest: +NORMALIZE_WHITESPACE
nitypes.waveform.Spectrum(3, data=array([1., 2., 3.]), start_frequency=100.0,
    frequency_increment=10.0)
"""  # noqa: W505 - doc line too long

from nitypes.waveform._analog import AnalogWaveform
from nitypes.waveform._complex import ComplexWaveform
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
__doctest_requires__ = {".": "numpy>=2.0"}


# Hide that it was defined in a helper file
AnalogWaveform.__module__ = __name__
ComplexWaveform.__module__ = __name__
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
