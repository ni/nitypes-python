"""Complex number data types for NI Python APIs."""

from nitypes.complex._complexint import ComplexInt
from nitypes.complex._complexint32 import ComplexInt32, _ComplexInt32Scalar

__all__ = ["ComplexInt", "ComplexInt32", "_ComplexInt32Scalar"]

# Hide that it was defined in a helper file
ComplexInt.__module__ = __name__
ComplexInt32.__module__ = __name__
_ComplexInt32Scalar.__module__ = __name__
