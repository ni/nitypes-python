from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing_extensions import assert_type

from nitypes.complex import ComplexInt32Base, ComplexInt32DType, convert_complex


def test___complexint32_to_complex64___convert_complex_array___converts_array() -> None:
    value_in = np.array([(1, 2), (3, -4), (-5, 6), (-7, -8)], ComplexInt32DType)

    value_out = convert_complex(np.complex64, value_in)

    assert_type(value_out, npt.NDArray[np.complex64])
    assert isinstance(value_out, np.ndarray) and value_out.dtype == np.complex64
    assert list(value_out) == [1 + 2j, 3 - 4j, -5 + 6j, -7 - 8j]


def test___complexint32_to_complex128___convert_complex_array___converts_array() -> None:
    value_in = np.array([(1, 2), (3, -4), (-5, 6), (-7, -8)], ComplexInt32DType)

    value_out = convert_complex(np.complex128, value_in)

    assert_type(value_out, npt.NDArray[np.complex128])
    assert isinstance(value_out, np.ndarray) and value_out.dtype == np.complex128
    assert list(value_out) == [1 + 2j, 3 - 4j, -5 + 6j, -7 - 8j]


def test___complexint32_to_complexint32___convert_complex_array___returns_original_array() -> None:
    value_in = np.array([(1, 2), (3, -4), (-5, 6), (-7, -8)], ComplexInt32DType)

    value_out = convert_complex(ComplexInt32DType, value_in)

    assert_type(value_out, npt.NDArray[ComplexInt32Base])
    assert value_out is value_in


def test___complex64_to_complexint32___convert_complex_array___converts_array() -> None:
    value_in = np.array([1 + 2j, 3 - 4j, -5 + 6j, -7 - 8j], np.complex64)

    value_out = convert_complex(ComplexInt32DType, value_in)

    assert_type(value_out, npt.NDArray[ComplexInt32Base])
    assert isinstance(value_out, np.ndarray) and value_out.dtype == ComplexInt32DType
    assert [x.item() for x in value_out] == [(1, 2), (3, -4), (-5, 6), (-7, -8)]


def test___complex64_to_complex64___convert_complex_array___returns_original_array() -> None:
    value_in = np.array([1 + 2j, 3 - 4j, -5 + 6j, -7 - 8j], np.complex64)

    value_out = convert_complex(np.complex64, value_in)

    assert_type(value_out, npt.NDArray[np.complex64])
    assert value_out is value_in


def test___complex64_to_complex128___convert_complex_array___converts_array() -> None:
    value_in = np.array([1.23 + 4.56j, 6.78 - 9.01j], np.complex64)

    value_out = convert_complex(np.complex128, value_in)

    assert_type(value_out, npt.NDArray[np.complex128])
    assert isinstance(value_out, np.ndarray) and value_out.dtype == np.complex128
    # complex64 bruises the numbers (e.g. np.complex128(1.2300000190734863+4.559999942779541j)) so
    # round to 3 decimal places.
    assert list(np.round(value_out, 3)) == [1.23 + 4.56j, 6.78 - 9.01j]


def test___complex128_to_complexint32___convert_complex_array___converts_array() -> None:
    value_in = np.array([1 + 2j, 3 - 4j, -5 + 6j, -7 - 8j], np.complex128)

    value_out = convert_complex(ComplexInt32DType, value_in)

    assert_type(value_out, npt.NDArray[ComplexInt32Base])
    assert isinstance(value_out, np.ndarray) and value_out.dtype == ComplexInt32DType
    assert [x.item() for x in value_out] == [(1, 2), (3, -4), (-5, 6), (-7, -8)]


def test___complex128_to_complex64___convert_complex_array___converts_array() -> None:
    value_in = np.array([1.23 + 4.56j, 6.78 - 9.01j], np.complex128)

    value_out = convert_complex(np.complex64, value_in)

    assert_type(value_out, npt.NDArray[np.complex64])
    assert isinstance(value_out, np.ndarray) and value_out.dtype == np.complex64
    assert list(value_out) == [1.23 + 4.56j, 6.78 - 9.01j]


def test___complex128_to_complex128___convert_complex_array___returns_original_array() -> None:
    value_in = np.array([1.23 + 4.56j, 6.78 - 9.01j], np.complex128)

    value_out = convert_complex(np.complex128, value_in)

    assert_type(value_out, npt.NDArray[np.complex128])
    assert value_out is value_in
