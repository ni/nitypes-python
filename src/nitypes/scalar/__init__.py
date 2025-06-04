"""Scalar data types for NI Python APIs.

Scalar
================

A scalar data object represents a single scalar value with units information.
Valid types for the scalar value are bool, int32, double, and str

Constructing scalar data objects
-----------------------------

To construct a scalar data object, use the :any:`Scalar` class:

>>> Scalar(False)
nitypes.scalar.Scalar(False, "")
>>> Scalar(0)
nitypes.scalar.Scalar(0, "")
>>> Scalar(5.0, "volts")
nitypes.scalar.Scalar(5.0, "volts")
>>> Scalar("value", "volts")
nitypes.scalar.Scalar("value", "volts")
"""  # noqa: W505 - doc line too long

from nitypes.scalar._scalar import Scalar

__all__ = ["Scalar"]
__doctest_requires__ = {".": "numpy>=2.0"}


# Hide that it was defined in a helper file
Scalar.__module__ = __name__
