r"""Binary time data types for NI Python APIs.

NI Binary Time Format
=====================

This module implements the NI Binary Time Format (`NI-BTF`_), a high-resolution time format used by
NI software.

An NI-BTF time value is a 128-bit fixed point number consisting of a 64-bit whole seconds part and a
64-bit fractional seconds part. There are two types of NI-BTF time values:

* An NI-BTF absolute time represents a point in time as the number of seconds after midnight,
  January 1, 1904, UTC.
* An NI-BTF time interval represents a difference between two points in time.

NI-BTF time types are also supported in LabVIEW, LabWindows/CVI, and .NET. You can use NI-BTF time
to efficiently share high-resolution date-time information with other NI application development
environments.

.. _ni-btf: https://www.ni.com/docs/en-US/bundle/labwindows-cvi/page/cvi/libref/ni-btf.htm

DateTime
========

The :any:`DateTime` class represents a NI-BTF absolute time as a Python object.

Constructing
------------

As with :any:`datetime.datetime`, you can construct a :any:`DateTime` by specifying the year,
month, day, etc.:

>>> import datetime
>>> DateTime(2025, 5, 25, 16, 45, tzinfo=datetime.timezone.utc)
nitypes.bintime.DateTime(2025, 5, 25, 16, 45, tzinfo=datetime.timezone.utc)

.. note::
    :any:`DateTime` only supports :any:`datetime.timezone.utc`. It does not support time-zone-naive
    objects or time zones other than UTC.

You can also construct a :any:`DateTime` from a :any:`datetime.datetime` or
:any:`hightime.datetime`:

>>> DateTime(datetime.datetime(2025, 5, 25, 16, 45, tzinfo=datetime.timezone.utc))
nitypes.bintime.DateTime(2025, 5, 25, 16, 45, tzinfo=datetime.timezone.utc)
>>> import hightime
>>> DateTime(hightime.datetime(2025, 5, 25, 16, 45, tzinfo=datetime.timezone.utc))
nitypes.bintime.DateTime(2025, 5, 25, 16, 45, tzinfo=datetime.timezone.utc)

You can get the current time of day by calling :any:`DateTime.now`:

>>> DateTime.now(datetime.timezone.utc) # doctest: +ELLIPSIS
nitypes.bintime.DateTime(...)

Properties
----------

Like other ``datetime`` objects, :any:`DateTime` has properties for the year, month, day, hour,
minute, second, and microsecond.

>>> x = DateTime(datetime.datetime(2025, 5, 25, 16, 45, tzinfo=datetime.timezone.utc))
>>> (x.year, x.month, x.day)
(2025, 5, 25)
>>> (x.hour, x.minute, x.second, x.microsecond)
(16, 45, 0, 0)

Like :any:`hightime.datetime`, it also supports the femtosecond and yoctosecond properties.

>>> (x.femtosecond, x.yoctosecond)
(0, 0)

Resolution
----------

NI-BTF is a high-resolution time format, so it has significantly higher resolution than
:any:`datetime.datetime`. However, :any:`hightime.datetime` has even higher resolution:

========================   ================================
Class                      Smallest Time Increment
========================   ================================
:any:`datetime.datetime`   1 microsecond (1e-6 sec)
:any:`DateTime`            54210 yoctoseconds (5.4e-20 sec)
:any:`hightime.datetime`   1 yoctosecond (1e-24 sec)
========================   ================================

As a result, :any:`hightime.datetime` can represent the time down to the exact yoctosecond, but
:any:`DateTime` rounds the yoctosecond field.

>>> x = hightime.datetime(2025, 1, 1, yoctosecond=123456789, tzinfo=datetime.timezone.utc)
>>> x
hightime.datetime(2025, 1, 1, 0, 0, 0, 0, 0, 123456789, tzinfo=datetime.timezone.utc)
>>> DateTime(x) # doctest: +NORMALIZE_WHITESPACE
nitypes.bintime.DateTime(2025, 1, 1, 0, 0, 0, 0, 0, 123436417, tzinfo=datetime.timezone.utc)

Rounding
--------

NI-BTF represents fractional seconds as a binary fraction, which is a sum of inverse
powers of 2. Values that are not exactly representable as binary fractions will display
rounding error or "bruising" similar to a floating point number.

For example, it may round 100 microseconds down to 99.9999... microseconds.

>>> x = hightime.datetime(2025, 1, 1, microsecond=100, tzinfo=datetime.timezone.utc)
>>> x
hightime.datetime(2025, 1, 1, 0, 0, 0, 100, tzinfo=datetime.timezone.utc)
>>> DateTime(x) # doctest: +NORMALIZE_WHITESPACE
nitypes.bintime.DateTime(2025, 1, 1, 0, 0, 0, 99, 999999999, 999991239,
    tzinfo=datetime.timezone.utc)

TimeDelta
=========

The :any:`TimeDelta` class represents a NI-BTF time interval as a Python object.

Constructing
------------

You can construct a :any:`TimeDelta` from a number of seconds, expressed as an :any:`int`,
:any:`float`, or :any:`decimal.Decimal`.

>>> TimeDelta(100)
nitypes.bintime.TimeDelta(Decimal('100'))
>>> TimeDelta(100.125)
nitypes.bintime.TimeDelta(Decimal('100.125'))
>>> from decimal import Decimal
>>> TimeDelta(Decimal("100.125"))
nitypes.bintime.TimeDelta(Decimal('100.125'))

:any:`TimeDelta` has the same resolution and rounding behavior as :any:`DateTime`.

>>> TimeDelta(Decimal("100.01234567890123456789"))
nitypes.bintime.TimeDelta(Decimal('100.012345678901234567889'))

Unlike other ``timedelta`` objects, you cannot construct a :any:`TimeDelta` from separate weeks,
days, hours, etc. If you want to do that, construct a :any:`datetime.timedelta` or
:any:`hightime.timedelta` and then use it to construct a :any:`TimeDelta`.

>>> TimeDelta(datetime.timedelta(days=1, microseconds=1))
nitypes.bintime.TimeDelta(Decimal('86400.0000010000000000000'))
>>> TimeDelta(hightime.timedelta(days=1, femtoseconds=1))
nitypes.bintime.TimeDelta(Decimal('86400.0000000000000010000'))

Math Operations
---------------

:any:`DateTime` and :any:`TimeDelta` support the same math operations as :any:`datetime.datetime`
and :any:`datetime.timedelta`.

For example, you can add or subtract :any:`TimeDelta` objects together:

>>> TimeDelta(100.5) + TimeDelta(0.5)
nitypes.bintime.TimeDelta(Decimal('101'))
>>> TimeDelta(100.5) - TimeDelta(0.5)
nitypes.bintime.TimeDelta(Decimal('100'))

Or add/subtract a :any:`DateTime` with a :any:`TimeDelta`, :any:`datetime.timedelta`, or
:any:`hightime.timedelta`:

>>> DateTime(2025, 1, 1, tzinfo=datetime.timezone.utc) + TimeDelta(86400)
nitypes.bintime.DateTime(2025, 1, 2, 0, 0, tzinfo=datetime.timezone.utc)
>>> DateTime(2025, 1, 1, tzinfo=datetime.timezone.utc) + datetime.timedelta(days=1)
nitypes.bintime.DateTime(2025, 1, 2, 0, 0, tzinfo=datetime.timezone.utc)
>>> DateTime(2025, 1, 1, tzinfo=datetime.timezone.utc) + hightime.timedelta(femtoseconds=1)
nitypes.bintime.DateTime(2025, 1, 1, 0, 0, 0, 0, 1, 13873, tzinfo=datetime.timezone.utc)

NI-BTF NumPy Structured Data Types
==================================

:any:`CVIAbsoluteTimeDType` and :any:`CVITimeIntervalDType` are NumPy structured data type objects
representing the ``CVIAbsoluteTime`` and ``CVITimeInterval`` C structs. These structured data types
can be used to efficiently represent NI-BTF time values in NumPy arrays or pass them to/from C DLLs.

.. warning::
    :any:`CVIAbsoluteTimeDType` and :any:`CVITimeIntervalDType` have the same layout and field
    names, so NumPy and type checkers such as Mypy currently treat them as the same type.

NI-BTF versus ``hightime``
==========================

NI also provides the ``hightime`` Python package, which extends the standard Python :mod:`datetime`
module to support up to yoctosecond precision.

``nitypes.bintime`` is not a replacement for ``hightime``. The two time formats have different
strengths and weaknesses.

* ``hightime`` supports local time zones and time-zone-naive times. ``bintime`` only supports UTC.
* ``hightime`` classes supports the same operations as the standard ``datetime`` classes.
  ``bintime`` classes support a subset of the standard ``datetime`` operations.
* ``hightime`` has a larger memory footprint than NI-BTF. ``hightime`` objects are separately
  allocated from the heap. ``bintime`` offers the choice of separately allocated Python objects or
  a more compact NumPy representation that can store multiple timestamps in a single block of
  memory.
* ``hightime`` requires conversion to/from NI-BTF when calling the NI driver C APIs from Python.
  ``nitypes.bintime`` includes reusable conversion routines for NI driver Python APIs to use.

NI-BTF versus :any:`numpy.datetime64`
=====================================

NumPy provides the :any:`numpy.datetime64` data type, which is even more compact than NI-BTF.
However, it has lower resolution than NI-BTF and is not interoperable with NI driver C APIs that use
NI-BTF.
"""

from __future__ import annotations

from nitypes.bintime._datetime import DateTime
from nitypes.bintime._dtypes import (
    CVIAbsoluteTimeBase,
    CVIAbsoluteTimeDType,
    CVITimeIntervalBase,
    CVITimeIntervalDType,
)
from nitypes.bintime._time_value_tuple import TimeValueTuple
from nitypes.bintime._timedelta import TimeDelta

__all__ = [
    "DateTime",
    "CVIAbsoluteTimeBase",
    "CVIAbsoluteTimeDType",
    "CVITimeIntervalBase",
    "CVITimeIntervalDType",
    "TimeDelta",
    "TimeValueTuple",
]

# Hide that it was defined in a helper file
DateTime.__module__ = __name__
TimeDelta.__module__ = __name__
TimeValueTuple.__module__ = __name__
