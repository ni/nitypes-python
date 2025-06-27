"""Scalar data types for NI Python APIs."""

from nitypes.scalar._scalar import Scalar

__all__ = ["Scalar"]

# Hide that it was defined in a helper file
Scalar.__module__ = __name__
