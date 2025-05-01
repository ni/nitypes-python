from __future__ import annotations

from typing import ClassVar, NewType, SupportsIndex, cast

import numpy as np

from nitypes._typing import Self
from nitypes.complex._complexint import ComplexInt

_ComplexInt32Scalar = NewType("_ComplexInt32Scalar", np.void)


class ComplexInt32(ComplexInt):
    """A complex number composed of two 16-bit integers.

    ======================
    NumPy Interoperability
    ======================

    You can use the ComplexInt32 class in place of a NumPy dtype object to construct NumPy arrays
    representing ComplexInt32 data.

    For example, you can convert a list of ComplexInt32 objects to the corresonding NumPy array:

    >>> import numpy as np
    >>> np.array([ComplexInt32(1, 2), ComplexInt32(3, 4)], dtype=ComplexInt32)
    array([(1, 2), (3, 4)], dtype=[('real', '<i2'), ('imag', '<i2')])

    You can also use dtype=ComplexInt32 with ordinary tuples:

    >>> np.array([(1, 2), (3, 4)], dtype=ComplexInt32)
    array([(1, 2), (3, 4)], dtype=[('real', '<i2'), ('imag', '<i2')])

    However, you cannot omit the dtype argument, or else NumPy will treat the ComplexInt32 objects
    as ordinary tuples and promote their contents to np.int64.

    >>> x = np.array([ComplexInt32(1, 2), ComplexInt32(3, 4)])
    >>> x
    array([[1, 2],
           [3, 4]])
    >>> x.dtype
    dtype('int64')

    The array elements are NumPy structured arrays with "real" and "imag" fields, not instances of
    the ComplexInt32 class.

    >>> x = np.array([[(1, 2), (3, 4)], [(5, 6), (7, 8)]], dtype=ComplexInt32)
    >>> x[0, 0]
    np.void((1, 2), dtype=[('real', '<i2'), ('imag', '<i2')])
    >>> x[1, 0]['real']
    np.int16(5)
    >>> x[1, 0]['imag']
    np.int16(6)
    >>> x[:, 0]
    array([(1, 2), (5, 6)], dtype=[('real', '<i2'), ('imag', '<i2')])

    You can also use the "real" and "imag" fields on the array to slice the real and imaginary
    parts.

    >>> x['real']
    array([[1, 3],
           [5, 7]], dtype=int16)
    >>> x['imag']
    array([[2, 4],
           [6, 8]], dtype=int16)
    """

    def __new__(cls, real: SupportsIndex = 0, imag: SupportsIndex = 0) -> Self:
        """Construct a ComplexInt32."""
        self = super().__new__(cls, real, imag)
        if not (-32768 <= self.real <= 32767):
            raise OverflowError("The real part must be between -32768 and 32767.")
        if not (-32768 <= self.imag <= 32767):
            raise OverflowError("The imaginary part must be between -32768 and 32767.")
        return self

    dtype: ClassVar[np.dtype[_ComplexInt32Scalar]]
    """A NumPy dtype object representing a ComplexInt32 as a structured array."""


ComplexInt32.dtype = cast(
    np.dtype[_ComplexInt32Scalar], np.dtype([("real", np.int16), ("imag", np.int16)])
)
