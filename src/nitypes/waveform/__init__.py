"""Waveform data types for NI Python APIs.

Analog Waveforms
================

An analog waveform represents a single analog signal with timing information and extended properties
such as units.

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

The 2D version, :any:`AnalogWaveform.from_array_2d`, returns multiple waveforms, one for each row of
data in the array or nested sequence.

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

Timing Information
------------------

Analog waveforms include timing information, such as the start time and sample interval, to support
analyzing and visualizing the data.

You can specify timing information by constructing a :any:`Timing` object and passing it to the
waveform constructor or factory method:

>>> import datetime as dt
>>> wfm = AnalogWaveform(timing=Timing.create_with_regular_interval(
...     dt.timedelta(seconds=1e-3), dt.datetime(2024, 12, 31, 23, 59, 59, tzinfo=dt.timezone.utc)
... ))
>>> wfm.timing  # doctest: +NORMALIZE_WHITESPACE
nitypes.waveform.Timing(nitypes.waveform.SampleIntervalMode.REGULAR,
    timestamp=datetime.datetime(2024, 12, 31, 23, 59, 59, tzinfo=datetime.timezone.utc),
    sample_interval=datetime.timedelta(microseconds=1000))

You can query the waveform's timing information using the :any:`Timing` object's properties:

>>> wfm.timing.start_time
datetime.datetime(2024, 12, 31, 23, 59, 59, tzinfo=datetime.timezone.utc)
>>> wfm.timing.sample_interval
datetime.timedelta(microseconds=1000)

Timing objects are immutable, so you cannot directly set their properties:

>>> wfm.timing.sample_interval = dt.timedelta(seconds=10e-3)  # doctest: +ELLIPSIS
Traceback (most recent call last):
...
AttributeError: ...

Instead, if you want to modify the timing information for an existing waveform, you can create a new
timing object and set the :any:`NumericWaveform.timing` property:

>>> wfm.timing = Timing.create_with_regular_interval(
...     dt.timedelta(seconds=1e-3), dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
... )
>>> wfm.timing  # doctest: +NORMALIZE_WHITESPACE
nitypes.waveform.Timing(nitypes.waveform.SampleIntervalMode.REGULAR,
    timestamp=datetime.datetime(2025, 1, 1, 0, 0, tzinfo=datetime.timezone.utc),
    sample_interval=datetime.timedelta(microseconds=1000))

Timing objects support time types from the :any:`datetime`, :any:`hightime`, and
:any:`nitypes.bintime` modules. If you need the timing information in a specific representation, use
the conversion methods:

>>> wfm.timing.to_datetime()  # doctest: +NORMALIZE_WHITESPACE
nitypes.waveform.Timing(nitypes.waveform.SampleIntervalMode.REGULAR,
    timestamp=datetime.datetime(2025, 1, 1, 0, 0, tzinfo=datetime.timezone.utc),
    sample_interval=datetime.timedelta(microseconds=1000))
>>> wfm.timing.to_hightime()  # doctest: +NORMALIZE_WHITESPACE
nitypes.waveform.Timing(nitypes.waveform.SampleIntervalMode.REGULAR,
    timestamp=hightime.datetime(2025, 1, 1, 0, 0, tzinfo=datetime.timezone.utc),
    sample_interval=hightime.timedelta(microseconds=1000))
>>> wfm.timing.to_bintime()  # doctest: +NORMALIZE_WHITESPACE
nitypes.waveform.Timing(nitypes.waveform.SampleIntervalMode.REGULAR,
    timestamp=nitypes.bintime.DateTime(2025, 1, 1, 0, 0, tzinfo=datetime.timezone.utc),
    sample_interval=nitypes.bintime.TimeDelta(Decimal('0.000999999999999999966606573')))

If :any:`NumericWaveform.timing` is not specified for a given waveform, it defaults to the
:any:`Timing.empty` singleton object.

>>> AnalogWaveform().timing
nitypes.waveform.Timing(nitypes.waveform.SampleIntervalMode.NONE)
>>> AnalogWaveform().timing is Timing.empty
True

Accessing unspecified properties of the timing object raises an exception:

>>> Timing.empty.sample_interval
Traceback (most recent call last):
...
RuntimeError: The waveform timing does not have a sample interval.

You can use :any:`Timing.sample_interval_mode` and ``has_*`` properties such as
:any:`Timing.has_timestamp` to query which properties of the timing object were specified:

>>> wfm.timing.sample_interval_mode
<SampleIntervalMode.REGULAR: 1>
>>> (wfm.timing.has_timestamp, wfm.timing.has_sample_interval)
(True, True)
>>> Timing.empty.sample_interval_mode
<SampleIntervalMode.NONE: 0>
>>> (Timing.empty.has_timestamp, Timing.empty.has_sample_interval)
(False, False)

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

Complex waveforms support scaling raw integer data to floating-point. Python and NumPy do not have
native support for complex integers, so this uses the :any:`ComplexInt32DType` structured data type.

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

Timing information
------------------

Complex waveforms have the same timing information as analog waveforms.

Digital Waveforms
=================

A digital waveform represents one or more digital signals with timing information and extended
properties such as channel name and signal names.

Constructing digital waveforms
------------------------------

To construct a digital waveform, use the :any:`DigitalWaveform` class:

>>> DigitalWaveform()
nitypes.waveform.DigitalWaveform(0, 1)
>>> DigitalWaveform(sample_count=5, signal_count=3)  # doctest: +NORMALIZE_WHITESPACE
nitypes.waveform.DigitalWaveform(5, 3, data=array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
[0, 0, 0]], dtype=uint8))

When displaying a digital waveform as a string, the first number is the sample count and the second
number is the signal count.

To construct a digital waveform from a NumPy array of line data, use the
:any:`DigitalWaveform.from_lines` method. Each array element represents a digital state, such as 1
for "on" or 0 for "off". The line data should be in a 1D array indexed by sample or a 2D array
indexed by (sample, signal). The digital waveform displays the line data as a 2D array.

>>> import numpy as np
>>> DigitalWaveform.from_lines(np.array([0, 1, 0], np.uint8))
nitypes.waveform.DigitalWaveform(3, 1, data=array([[0], [1], [0]], dtype=uint8))
>>> DigitalWaveform.from_lines(np.array([[0, 0], [1, 0], [0, 1], [1, 1]], np.uint8))
nitypes.waveform.DigitalWaveform(4, 2, data=array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=uint8))

You can also use :any:`DigitalWaveform.from_lines` to construct a digital waveform from a sequence,
such as a list.

>>> DigitalWaveform.from_lines([[0, 0], [1, 0], [0, 1], [1, 1]])
nitypes.waveform.DigitalWaveform(4, 2, data=array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=uint8))

To construct a digital waveform from a NumPy array of port data, use the
:any:`DigitalWaveform.from_port` method. Each element of the port data array represents a digital
sample taken over a port of signals. Each bit in the sample is a signal value, either 1 for "on" or
0 for "off". The least significant bit of the sample is placed at signal index 0 of the
DigitalWaveform.

>>> DigitalWaveform.from_port(np.array([0, 1, 2, 3], np.uint8))  # doctest: +NORMALIZE_WHITESPACE
nitypes.waveform.DigitalWaveform(4, 8, data=array([[0, 0, 0, 0, 0, 0, 0, 0],
[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]], dtype=uint8))

You can use a mask to specify which lines in the port to include in the waveform.

>>> DigitalWaveform.from_port(np.array([0, 1, 2, 3], np.uint8), 0x3)
nitypes.waveform.DigitalWaveform(4, 2, data=array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=uint8))

You can also use a non-NumPy sequence such as a list, but you must specify a mask so the waveform
knows how many bits are in each list element.

>>> DigitalWaveform.from_port([0, 1, 2, 3], 0x3)
nitypes.waveform.DigitalWaveform(4, 2, data=array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=uint8))

The 2D version, :any:`DigitalWaveform.from_ports`, returns multiple waveforms, one for each row of
data in the array or nested sequence.

>>> nested_list = [[0, 1, 2, 3], [3, 0, 3, 0]]
>>> DigitalWaveform.from_ports(nested_list, [0x3, 0x3])  # doctest: +NORMALIZE_WHITESPACE
[nitypes.waveform.DigitalWaveform(4, 2, data=array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=uint8)),
 nitypes.waveform.DigitalWaveform(4, 2, data=array([[1, 1], [0, 0], [1, 1], [0, 0]], dtype=uint8))]

Digital signals
---------------

You can access individual signals using the :any:`DigitalWaveform.signals` property.

>>> wfm = DigitalWaveform.from_port([0, 1, 2, 3], 0x3)
>>> wfm.signals[0]
nitypes.waveform.DigitalWaveformSignal(data=array([0, 1, 0, 1], dtype=uint8))
>>> wfm.signals[1]
nitypes.waveform.DigitalWaveformSignal(data=array([0, 0, 1, 1], dtype=uint8))

The :any:`DigitalWaveformSignal.data` property returns a view of the data for that signal.

>>> wfm.signals[0].data
array([0, 1, 0, 1], dtype=uint8)

Digital signal names
--------------------

The :any:`DigitalWaveformSignal.name` property allows you to get and set the signal names.

>>> wfm.signals[0].name = "port0/line0"
>>> wfm.signals[1].name = "port0/line1"
>>> wfm.signals[0].name
'port0/line0'
>>> wfm.signals[0]
nitypes.waveform.DigitalWaveformSignal(name='port0/line0', data=array([0, 1, 0, 1], dtype=uint8))

The signal names are stored in the ``NI_LineNames`` extended property on the digital waveform.

>>> wfm.extended_properties["NI_LineNames"]
'port0/line0, port0/line1'

When creating a digital waveform, you can directly set the ``NI_LineNames`` extended property.

>>> wfm = DigitalWaveform.from_port([2, 4], 0x7,
... extended_properties={"NI_LineNames": "Dev1/port1/line4, Dev1/port1/line5, Dev1/port1/line6"})
>>> wfm.signals[0]
nitypes.waveform.DigitalWaveformSignal(name='Dev1/port1/line4', data=array([0, 0], dtype=uint8))
>>> wfm.signals[1]
nitypes.waveform.DigitalWaveformSignal(name='Dev1/port1/line5', data=array([1, 0], dtype=uint8))
>>> wfm.signals[2]
nitypes.waveform.DigitalWaveformSignal(name='Dev1/port1/line6', data=array([0, 1], dtype=uint8))

Digital state types
-------------------

By default, digital waveforms use a NumPy ``dtype`` of :any:`numpy.uint8`, which uses a byte of
memory for each digital state.

Using ``np.uint8`` allows the waveform to contain digital states other than "on" or off", such as
such as :any:`DigitalState.FORCE_OFF` (``X``) or :any:`DigitalState.COMPARE_HIGH` (``H``). This
capability is used for digital pattern applications.

You can also construct a digital waveform using a NumPy ``dtype`` of :any:`numpy.bool`. This also
uses a byte of memory for each digital state, but it restricts the states to "on" and "off".

Testing digital waveforms
-------------------------

You can use :any:`DigitalWaveform.test` to compare an acquired waveform against an expected
waveform. This returns a :any:`DigitalWaveformTestResult` object, which has a Boolean ``success``
property and a ``failures`` property containing a collection of :any:`DigitalWaveformFailure`
objects, which indicate the location of each test failure.

Here is an example. The expected waveform counts in binary using ``COMPARE_LOW`` (``L``) and
``COMPARE_HIGH`` (``H``), but signal 1 of the actual waveform is stuck high.

>>> actual = DigitalWaveform.from_lines([[0, 1], [1, 1], [0, 1], [1, 1]])
>>> expected = DigitalWaveform.from_lines([[DigitalState.COMPARE_LOW, DigitalState.COMPARE_LOW],
... [DigitalState.COMPARE_HIGH, DigitalState.COMPARE_LOW],
... [DigitalState.COMPARE_LOW, DigitalState.COMPARE_HIGH],
... [DigitalState.COMPARE_HIGH, DigitalState.COMPARE_HIGH]])
>>> result = actual.test(expected)
>>> result.success
False
>>> len(result.failures)
2

The failures indicate the sample indices into the actual and expected waveforms, the signal index,
and the digital state from the actual and expected waveforms:

>>> result.failures[0]  # doctest: +NORMALIZE_WHITESPACE
DigitalWaveformFailure(sample_index=0, expected_sample_index=0, signal_index=1,
actual_state=<DigitalState.FORCE_UP: 1>, expected_state=<DigitalState.COMPARE_LOW: 3>)
>>> result.failures[1]  # doctest: +NORMALIZE_WHITESPACE
DigitalWaveformFailure(sample_index=1, expected_sample_index=1, signal_index=1,
actual_state=<DigitalState.FORCE_UP: 1>, expected_state=<DigitalState.COMPARE_LOW: 3>)

Timing information
------------------

Digital waveforms have the same timing information as analog waveforms.

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
