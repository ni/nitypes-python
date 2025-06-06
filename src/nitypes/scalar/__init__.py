"""Scalar data types for NI Python APIs.

Scalar
================

A scalar data object represents a single scalar value with units information.
Valid types for the scalar value are :any:`bool`, :any:`int`, :any:`float`, and :any:`str`.

Constructing scalar data objects
--------------------------------

To construct a scalar data object, use the :any:`Scalar` class:

>>> Scalar(False)
nitypes.scalar.Scalar(value=False, units='')
>>> Scalar(0)
nitypes.scalar.Scalar(value=0, units='')
>>> Scalar(5.0, 'volts')
nitypes.scalar.Scalar(value=5.0, units='volts')
>>> Scalar("value", "volts")
nitypes.scalar.Scalar(value='value', units='volts')
"""

from nitypes.scalar._scalar import Scalar

__all__ = ["Scalar"]

# Hide that it was defined in a helper file
Scalar.__module__ = __name__
