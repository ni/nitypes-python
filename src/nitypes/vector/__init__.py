"""Vector data type for NI Python APIs.

Vector Data Type
=================

* :class:`Vector`: A vector data object represents an array of scalar values with units information.
  Valid types for the scalar value are :any:`bool`, :any:`int`, :any:`float`, and :any:`str`.
"""

from nitypes.vector._vector import Vector

__all__ = ["Vector"]

# Hide that it was defined in a helper file
Vector.__module__ = __name__
