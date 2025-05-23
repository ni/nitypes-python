r"""Binary time data types for NI Python APIs.

=====================
NI Binary Time Format
=====================

This module implements the NI Binary Time Format (`NI-BTF`_), a high-resolution time format used by
NI software.

An NI-BTF time value is a 128-bit fixed point number consisting of a 64-bit whole seconds part and
a 64-bit fractional seconds part. There are two types of NI-BTF time values:

* An NI-BTF absolute time represents a point in time as the number of seconds after midnight,
  January 1, 1904, UTC.
* An NI-BTF time interval represents a difference between two points in time.

NI-BTF time types are also supported in LabVIEW, LabWindows/CVI, and .NET. You can use NI-BTF time
to efficiently share high-resolution date-time information with other NI application development
environments.

.. _ni-btf: https://www.ni.com/docs/en-US/bundle/labwindows-cvi/page/cvi/libref/ni-btf.htm

NI-BTF versus ``hightime``
--------------------------

NI also provides the ``hightime`` Python package, which extends the standard Python :mod:`datetime`
module to support up to yoctosecond precision.

``nitypes.bintime`` is not a replacement for ``hightime``. The two time formats have different
strengths and weaknesses.

* ``hightime`` supports local time zones. NI-BTF only supports UTC.
* ``hightime`` supports the same operations as the standard ``datetime`` module. NI-BTF supports a
  limited set of operations.
* ``hightime`` has a larger memory footprint than NI-BTF. Each ``hightime`` object is separately
  allocated from the heap. NI-BTF has a compact, 128-bit representation. You can allocate a NumPy
  array containing many NI-BTF time values in a single block of memory.
* ``hightime`` requires conversion to/from NI-BTF when calling the NI driver C APIs from Python.
  ``nitypes.bintime`` includes reusable conversion routines for NI driver Python APIs to use.

NI-BTF versus :any:`numpy.datetime64`
-------------------------------------

NumPy provides the :any:`numpy.datetime64` data type, which is even more compact than NI-BTF.
However, it has lower resolution than NI-BTF and is not interoperable with NI driver C APIs that
use NI-BTF.

============
AbsoluteTime
============


============
TimeInterval
============

==================================
NI-BTF NumPy Structured Data Types
==================================

:any:`CVIAbsoluteTimeDType` and :any:`CVITimeIntervalDType` are NumPy structured data type objects
representing the ``CVIAbsoluteTime`` and ``CVITimeInterval`` C structs. These structured data types
can be used to efficiently represent NI-BTF time values in NumPy arrays and to pass NI-BTF time
values to/from NI driver APIs.

.. warning::
    :any:`CVIAbsoluteTimeDType` and :any:`CVITimeIntervalDType` have the same layout and field
    names, so NumPy treats them as the same data type.

"""

from __future__ import annotations

from nitypes.bintime._absolutetime import AbsoluteTime
from nitypes.bintime._dtypes import (
    CVIAbsoluteTimeBase,
    CVIAbsoluteTimeDType,
    CVITimeIntervalBase,
    CVITimeIntervalDType,
)
from nitypes.bintime._timevalue import TimeValue

__all__ = [
    "CVIAbsoluteTimeBase",
    "CVIAbsoluteTimeDType",
    "CVITimeIntervalBase",
    "CVITimeIntervalDType",
    "AbsoluteTime",
    "TimeValue",
]

# Hide that it was defined in a helper file
AbsoluteTime.__module__ = __name__
TimeValue.__module__ = __name__
