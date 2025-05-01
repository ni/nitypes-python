from __future__ import annotations

import operator
from collections.abc import Iterator, Sequence
from math import sqrt
from numbers import Complex
from typing import SupportsIndex

from nitypes._typing import Self


# Implementing Sequence allows conversion to NumPy arrays.
class ComplexInt(Complex, Sequence[int]):
    """A complex number composed of two integers."""

    __slots__ = ["_real", "_imag"]

    def __init__(self, real: SupportsIndex = 0, imag: SupportsIndex = 0) -> None:
        """Construct a ComplexInt."""
        self._real = operator.index(real)
        self._imag = operator.index(imag)

    def __complex__(self) -> complex:
        """Return self as a floating-point complex number."""
        return complex(self._real, self._imag)

    @property
    def real(self) -> int:
        """The real part of the complex number."""
        return self._real

    @property
    def imag(self) -> int:
        """The imaginary part of the complex number."""
        return self._imag

    def __add__(self, other: Self, /) -> Self:
        """Return self + other."""
        return self.__class__(self._real + other._real, self._imag + other._imag)

    __radd__ = __add__

    def __neg__(self) -> Self:
        """Return -self."""
        return self.__class__(-self._real, -self._imag)

    def __pos__(self) -> Self:
        """Return +self."""
        return self

    def __sub__(self, other: Self, /) -> Self:
        """Return self - other."""
        return self.__class__(self._real - other._real, self._imag - other._imag)

    def __rsub__(self, other: Self, /) -> Self:
        """Return other - self."""
        return self.__class__(other._real - self._real, other._imag - self._imag)

    def __mul__(self, other: Self, /) -> Self:
        """Return self * other."""
        real = self._real * other._real - self._imag * other._imag
        imag = self._real * other._imag + self._imag * other._real
        return self.__class__(real, imag)

    __rmul__ = __mul__

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
        return sqrt(self._real * self._real + self._imag * self._imag)

    def conjugate(self) -> Self:
        """Return the complex conjugate."""
        return self.__class__(self._real, -self._imag)

    def __eq__(self, other: object, /) -> bool:
        """Return self == other."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._real == other._real and self._imag == other._imag

    def __str__(self) -> str:
        """Return str(self)."""
        return f"{self._real}+{self._imag}j"

    def __repr__(self) -> str:
        """Return repr(self)."""
        return f"{self.__class__.__module__}.{self.__class__.__name__}({self._real}, {self._imag})"

    def __hash__(self) -> int:  # type: ignore[override]
        """Return hash(self)."""
        return hash(self._as_tuple())

    def __len__(self) -> int:
        """Return len(self)."""
        return 2

    def __iter__(self) -> Iterator[int]:
        """Return iter(self)."""
        return iter(self._as_tuple())

    def __contains__(self, value: int) -> bool:
        """Return value in self."""
        return value in self._as_tuple()

    def __getitem__(self, index: int) -> int:
        """Return self[index]."""
        return self._as_tuple()[index]

    def _as_tuple(self) -> tuple[int, int]:
        return (self._real, self._imag)
