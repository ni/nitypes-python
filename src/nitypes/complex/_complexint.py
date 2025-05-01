from __future__ import annotations

import operator
from math import sqrt
from numbers import Complex
from typing import SupportsIndex

from nitypes._typing import Self

# This subclasses tuple so that it can be used to initialize NumPy structured arrays. Unfortunately,
# tuple and Complex have different ideas about how __add__ and __mul__ should be typed.
#
# Why not NamedTuple? Because Complex is a subclass of Generic, and NamedTuple and Generic have
# different metaclasses.
#
# TODO: can __array__ convert to a structured array?


class ComplexInt(tuple[int, int], Complex):
    """A complex number composed of two integers."""

    __slots__ = ()

    def __new__(cls, real: SupportsIndex = 0, imag: SupportsIndex = 0) -> Self:
        """Construct a ComplexInt."""
        real = operator.index(real)
        imag = operator.index(imag)
        return super().__new__(cls, (real, imag))

    def __complex__(self) -> complex:
        """Return self as a floating-point complex number."""
        return complex(self.real, self.imag)

    @property
    def real(self) -> int:
        """The real part of the complex number."""
        return self[0]

    @property
    def imag(self) -> int:
        """The imaginary part of the complex number."""
        return self[1]

    def __add__(self, other: Self, /) -> Self:  # type: ignore[override]
        """Return self + other."""
        return self.__class__(self.real + other.real, self.imag + other.imag)

    __radd__ = __add__

    def __neg__(self) -> Self:
        """Return -self."""
        return self.__class__(-self.real, -self.imag)

    def __pos__(self) -> Self:
        """Return +self."""
        return self

    def __sub__(self, other: Self, /) -> Self:
        """Return self - other."""
        return self.__class__(self.real - other.real, self.imag - other.imag)

    def __rsub__(self, other: Self, /) -> Self:
        """Return other - self."""
        return self.__class__(other.real - self.real, other.imag - self.imag)

    def __mul__(self, other: Self, /) -> Self:  # type: ignore[override]
        """Return self * other."""
        real = self.real * other.real - self.imag * other.imag
        imag = self.real * other.imag + self.imag * other.real
        return self.__class__(real, imag)

    __rmul__ = __mul__  # type: ignore[assignment]

    def __truediv__(self, other: Self, /) -> complex:
        """Return self / other."""
        return complex(self) / complex(other)

    def __rtruediv__(self, other: Self, /) -> complex:
        """Return other / self."""
        return complex(other) / complex(self)

    def __floordiv__(self, other: Self, /) -> Self:
        """Return self // other."""
        return self.__class__._floordiv(self, other)

    def __rfloordiv__(self, other: Self, /) -> Self:
        """Return other // self."""
        return self.__class__._floordiv(other, self)

    @classmethod
    def _floordiv(cls, left: Self, right: Self) -> Self:
        denominator = right.real * right.real + right.imag * right.imag
        real = (left.real * right.real + left.imag * right.imag) // denominator
        imag = (left.imag * right.real - left.real * right.imag) // denominator
        return cls(real, imag)

    def __pow__(self, exponent: Self) -> complex:
        """Return pow(self, exponent)."""
        return pow(complex(self), complex(exponent))

    def __rpow__(self, base: Self) -> complex:
        """Return pow(base, self)."""
        return pow(complex(base), complex(self))

    def __abs__(self) -> float:
        """Return abs(self)."""
        return sqrt(self.real * self.real + self.imag * self.imag)

    def conjugate(self) -> Self:
        """Return the complex conjugate."""
        return self.__class__(self.real, -self.imag)

    def __eq__(self, other: object, /) -> bool:
        """Return self == other."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.real == other.real and self.imag == other.imag

    def __str__(self) -> str:
        """Return str(self)."""
        return f"{self.real}+{self.imag}j"

    def __repr__(self) -> str:
        """Return repr(self)."""
        return f"{self.__class__.__module__}.{self.__class__.__name__}({self.real}, {self.imag})"

    def __hash__(self) -> int:  # type: ignore[override]
        """Return hash(self)."""
        return super().__hash__()
