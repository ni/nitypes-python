"""Complex number data types for NI Python APIs.

================
Complex Integers
================

Some NI driver APIs (such as NI-FGEN, NI-SCOPE, NI-RFSA, and NI-RFSG) use complex numbers to
represent I/Q data. Python and NumPy have native support for complex floating-point numbers, but
not complex integers, so the ``nityeps.complex`` submodule provides a NumPy representation of
complex integers.

``ComplexInt32DType`` is a NumPy structured data type object representing a complex integer with
16-bit ``real`` and ``imag`` fields. This structured data type has the same memory layout as the
``NIComplexI16`` C struct used by NI driver APIs.

For more information about NumPy structured data types, see the
:ref:`NumPy documentation on structured arrays <numpy:structured_arrays>`.

.. note::
   In ``NIComplexI16``, the number 16 refers to the number of bits in each field. In
   ``ComplexInt32DType``, the number 32 refers to the total number of bits, following the precedent
   set by NumPy's other complex types. For example, ``np.complex128`` contains 64-bit ``real`` and
   ``imag`` fields.

Constructing arrays of complex integers
---------------------------------------

You can construct an array of complex integers from a sequence of tuples using ``np.array``:

>>> import numpy as np
>>> np.array([(1, 2), (3, 4)], dtype=ComplexInt32DType)
array([(1, 2), (3, 4)], dtype=[('real', '<i2'), ('imag', '<i2')])

Likewise, you can construct an array of complex integer zeros using ``np.zeros``:

>>> np.zeros(3, dtype=ComplexInt32DType)
array([(0, 0), (0, 0), (0, 0)], dtype=[('real', '<i2'), ('imag', '<i2')])

Indexing and slicing
--------------------

Indexing the array gives you a complex integer structured scalar:

>>> x = np.array([(1, 2), (3, 4), (5, 6)], dtype=ComplexInt32DType)
>>> x[0]
np.void((1, 2), dtype=[('real', '<i2'), ('imag', '<i2')])
>>> x[1]
np.void((3, 4), dtype=[('real', '<i2'), ('imag', '<i2')])

.. note:
    NumPy displays ``np.void`` because the ``ComplexInt32DType`` has a base type of ``np.void``.
    Defining a structured array with a different base type such as ``np.int32`` would have benefits,
    such as making it easier to convert array elements to/from ``np.int32``, but it would also have
    drawbacks, such as making it harder to initialize the array using a sequence of tuples.

You can index a complex integer structured scalar to get the real and imaginary parts:

>>> x[0][0]
np.int16(1)
>>> x[0][1]
np.int16(2)

You can also index by field names ``real`` and ``imag``:

>>> x[0]['real']
np.int16(1)
>>> x[0]['imag']
np.int16(2)

Or you can index the entire array by field names:

>>> x['real']
array([1, 3, 5], dtype=int16)
>>> x['imag']
array([2, 4, 6], dtype=int16)

Arrays of complex integers support slicing and negative indices like any other array:

>>> x[0:2]
array([(1, 2), (3, 4)], dtype=[('real', '<i2'), ('imag', '<i2')])
>>> x[1:]
array([(3, 4), (5, 6)], dtype=[('real', '<i2'), ('imag', '<i2')])
>>> x[-1]
np.void((5, 6), dtype=[('real', '<i2'), ('imag', '<i2')])

Conversion
----------

To convert a complex integer structured scalar to a tuple, use the ``item`` method:

>>> x[0].item()
(1, 2)
>>> [y.item() for y in x]
[(1, 2), (3, 4), (5, 6)]

To convert NumPy arrays between between different complex number data types, use the
`convert_complex` function:

>>> convert_complex(np.complex128, x)
array([1.+2.j, 3.+4.j, 5.+6.j])
>>> convert_complex(ComplexInt32DType, np.array([1.23+4.56j]))
array([(1, 4)], dtype=[('real', '<i2'), ('imag', '<i2')])

You can also use `convert_complex` with NumPy scalars:

>>> convert_complex(np.complex128, x[0])
np.complex128(1+2j)
>>> convert_complex(ComplexInt32DType, np.complex128(3+4j))
np.void((3, 4), dtype=[('real', '<i2'), ('imag', '<i2')])

Mathematical operations
-----------------------

Structured arrays of complex integers do not support mathematical operations. Convert
them to arrays of complex floating-point numbers before doing any sort of math or analysis.
"""

from nitypes.complex._conversion import convert_complex
from nitypes.complex._dtypes import ComplexInt32Base, ComplexInt32DType

__all__ = ["convert_complex", "ComplexInt32DType", "ComplexInt32Base"]
