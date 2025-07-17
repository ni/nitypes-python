"""Scalar data types for NI Python APIs.

Scalar Data Types
=================

* :class:`Scalar`: A scalar data object represents a single scalar value with units information.
  Valid types for the scalar value are :any:`bool`, :any:`int`, :any:`float`, and :any:`str`.
"""

from nitypes.scalar._scalar import Scalar

__all__ = ["Scalar"]

# Hide that it was defined in a helper file
Scalar.__module__ = __name__
