"""Scalar data types for NI Python APIs.

Scalar Data
================

A scalar data object represents a single scalar value with units information.
Valid types for the scalar value are bool, int32, double, and str

Constructing scalar data objects
-----------------------------

To construct a scalar data object, use the :any:`ScalarData` class:

>>> ScalarData(False)
nitypes.scalar.ScalarData(False, "")
>>> ScalarData(0)
nitypes.scalar.ScalarData(0, "")
>>> ScalarData(5.0, "volts")
nitypes.scalar.ScalarData(5.0, "volts")
>>> ScalarData("value", "volts")
nitypes.scalar.ScalarData("value", "volts")
"""  # noqa: W505 - doc line too long

from nitypes.scalar._scalar_data import ScalarData

__all__ = ["ScalarData"]
__doctest_requires__ = {".": "numpy>=2.0"}


# Hide that it was defined in a helper file
ScalarData.__module__ = __name__
