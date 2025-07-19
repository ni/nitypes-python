Introduction
============

About
-----

The ``nitypes`` Python package defines data types for NI Python APIs:

- Analog, complex, and digital waveforms
- Frequency spectrums
- Complex integers
- Time conversion

NI created and supports this package.

Operating System Support
------------------------

``nitypes`` supports Windows and Linux operating systems.

Python Version Support
----------------------

``nitypes`` supports CPython 3.9+ and PyPy3.

Installation
------------

Installing NI driver Python APIs that support waveforms will automatically install ``nitypes``.

You can also directly install the ``nitypes`` package using ``pip`` or by listing it as a dependency in
your project's ``pyproject.toml`` file.

Waveforms
---------

Analog Waveforms
^^^^^^^^^^^^^^^^

The :class:`AnalogWaveform` class represents a single analog signal with timing
information and extended properties (such as channel name and units).
Multi-channel analog data is represented using a collection of waveforms, such
as ``list[AnalogWaveform]``.

Complex Waveforms
^^^^^^^^^^^^^^^^^

The :class:`ComplexWaveform` class represents a complex-number signal, such as I/Q
data, with timing information and extended properties (such as channel name and
units).

Digital Waveforms
^^^^^^^^^^^^^^^^^

The :class:`DigitalWaveform` class represents one or more digital signals with
timing information and extended properties (such as channel name and signal
names).

Frequency Spectrums
^^^^^^^^^^^^^^^^^^^

The :class:`Spectrum` class represents a frequency spectrum with frequency range
information and extended properties (such as channel name and units).

Complex Numbers
---------------

Complex Integers
^^^^^^^^^^^^^^^^

:any:`ComplexInt32DType` is a NumPy structured data type object representing a
complex integer with 16-bit ``real`` and ``imag`` fields. This structured data
type has the same memory layout as the NIComplexI16 C struct used by NI driver
APIs.

Complex Number Conversion
^^^^^^^^^^^^^^^^^^^^^^^^^

You can use the :any:`convert_complex` function to convert complex-number NumPy
arrays between :any:`ComplexInt32DType` and the standard :any:`numpy.complex64`
and :any:`numpy.complex128` data types.

Time
----

Time Conversion
^^^^^^^^^^^^^^^

You can use the :any:`convert_datetime` and :any:`convert_timedelta` functions
to convert time values between the standard :class:`DateTime` library, the
high-precision :any:`hightime` library, and :any:`bintime`.

Binary Time
^^^^^^^^^^^

The :any:`bintime` module implements the NI Binary Time Format (NI-BTF), a
high-resolution time format used by NI software. An NI-BTF time value is a
128-bit fixed point number consisting of a 64-bit whole seconds part and a
64-bit fractional seconds part.

Scalar Values
-------------

Scalar
^^^^^^

:class:`Scalar` is a data type that represents a single scalar value with units
information. Valid types for the scalar value are :any:`bool`, :any:`int`,
:any:`float`, and :any:`str`.
