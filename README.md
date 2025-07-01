| **Info**      | Data types for NI Python APIs |
| :------------ | :-----------------------------|
| **Author**    | National Instruments          |

# Table of Contents

- [Table of Contents](#table-of-contents)
- [About](#about)
  - [Documentation](#documentation)
  - [Operating System Support](#operating-system-support)
  - [Python Version Support](#python-version-support)
- [Installation](#installation)
- [Waveforms](#waveforms)
  - [Analog Waveforms](#analog-waveforms)
  - [Complex Waveforms](#complex-waveforms)
  - [Digital Waveforms](#digital-waveforms)
  - [Frequency Spectrums](#frequency-spectrums)
- [Complex Numbers](#complex-numbers)
  - [Complex Integers](#complex-integers)
  - [Complex Number Conversion](#complex-number-conversion)
- [Time](#time)
  - [Time Conversion](#time-conversion)
  - [Binary Time](#binary-time)
- [Scalar Values](#scalar-values)
  - [Scalar](#scalar)

# About

The `nitypes` Python package defines data types for NI Python APIs:

- Analog, complex, and digital waveforms
- Frequency spectrums
- Complex integers
- Time conversion

NI created and supports this package.

## Documentation

See the [API Reference](https://nitypes.readthedocs.io/).

## Operating System Support

`nitypes` supports Windows and Linux operating systems.

## Python Version Support

`nitypes` supports CPython 3.9+ and PyPy3.

# Installation

Installing NI driver Python APIs that support waveforms will automatically install `nitypes`.

You can also directly install the `nitypes` package using `pip` or by listing it as a dependency in
your project's `pyproject.toml` file.

# Waveforms

## Analog Waveforms

The `nitypes.waveform.AnalogWaveform` class represents a single analog signal with timing
information and extended properties (such as channel name and units). Multi-channel analog data is
represented using a collection of waveforms, such as `list[nitypes.waveform.AnalogWaveform]`. For
more details, see [Analog
Waveforms](https://nitypes.readthedocs.io/en/latest/autoapi/nitypes/waveform/index.html#analog-waveforms)
in the API Reference.

## Complex Waveforms

The `nitypes.waveform.ComplexWaveform` class represents a complex-number signal, such as I/Q data,
with timing information and extended properties (such as channel name and units). For more details,
see [Complex
Waveforms](https://nitypes.readthedocs.io/en/latest/autoapi/nitypes/waveform/index.html#complex-waveforms)
in the API Reference.

## Digital Waveforms

The `nitypes.waveform.DigitalWaveform` class represents one or more digital signals with timing
information and extended properties (such as channel name and signal names). For more details, see
[Digital
Waveforms](https://nitypes.readthedocs.io/en/latest/autoapi/nitypes/waveform/index.html#complex-waveforms)
in the API Reference.

## Frequency Spectrums

The `nitypes.waveform.Spectrum` class represents a frequency spectrum with frequency range
information and extended properties (such as channel name and units). For more details, see
[Frequency
Spectrums](https://nitypes.readthedocs.io/en/latest/autoapi/nitypes/waveform/index.html#frequency-spectrums)
in the API Reference.

# Complex Numbers

## Complex Integers

`nitypes.complex.ComplexInt32DType` is a NumPy structured data type object representing a complex
integer with 16-bit `real` and `imag` fields. This structured data type has the same memory layout
as the NIComplexI16 C struct used by NI driver APIs. For more details, see [Complex
Integers](https://nitypes.readthedocs.io/en/latest/autoapi/nitypes/complex/index.html#complex-integers)
in the API Reference.

## Complex Number Conversion

You can use the `nitypes.complex.convert_complex()` function to convert complex-number NumPy arrays
between `nitypes.complex.ComplexInt32DType` and the standard `np.complex64` and `np.complex128` data
types. For more details, see [Complex >>
Conversion](https://nitypes.readthedocs.io/en/latest/autoapi/nitypes/complex/index.html#conversion)
in the API Reference.

# Time

## Time Conversion

You can use the `nitypes.time.convert_datetime()` and `nitypes.time.convert_timedelta()` functions
to convert time values between the standard `datetime` library, the high-precision `hightime`
library, and `bintime`. For more details, see [Time >>
Conversion](https://nitypes.readthedocs.io/en/latest/autoapi/nitypes/time/index.html#conversion) in
the API Reference.

## Binary Time

The `nitypes.bintime` module implements the NI Binary Time Format (NI-BTF), a high-resolution time
format used by NI software. An NI-BTF time value is a 128-bit fixed point number consisting of a
64-bit whole seconds part and a 64-bit fractional seconds part. For more details, see [NI Binary
Time Format](https://nitypes.readthedocs.io/en/latest/autoapi/nitypes/bintime/index.html#ni-binary-time-format)
in the API Reference.

# Scalar Values

## Scalar

`nitypes.scalar.Scalar` is a data type that represents a single scalar value with units
information and extended properties. Valid types for the scalar value are `bool`, `int`, `float`,
and `str`. For more details, see
[Scalar](https://nitypes.readthedocs.io/en/latest/autoapi/nitypes/scalar/index.html#scalar) in the
API Reference.