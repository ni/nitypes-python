from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from typing_extensions import assert_type

from nitypes.complex import ComplexInt32Base, ComplexInt32DType, convert_complex


###############################################################################
# convert arrays
###############################################################################
def test___complexint32_array_to_complex64_array___convert_complex___converts_array() -> None:
    value_in = np.array([(1, 2), (3, -4), (-5, 6), (-7, -8)], ComplexInt32DType)

    value_out = convert_complex(np.complex64, value_in)

    assert_type(value_out, npt.NDArray[np.complex64])
    assert isinstance(value_out, np.ndarray) and value_out.dtype == np.complex64
    assert list(value_out) == [1 + 2j, 3 - 4j, -5 + 6j, -7 - 8j]


def test___complexint32_array_to_complex128_array___convert_complex___converts_array() -> None:
    value_in = np.array([(1, 2), (3, -4), (-5, 6), (-7, -8)], ComplexInt32DType)

    value_out = convert_complex(np.complex128, value_in)

    assert_type(value_out, npt.NDArray[np.complex128])
    assert isinstance(value_out, np.ndarray) and value_out.dtype == np.complex128
    assert list(value_out) == [1 + 2j, 3 - 4j, -5 + 6j, -7 - 8j]


def test___complexint32_array_to_complexint32_array___convert_complex___returns_original_array() -> (
    None
):
    value_in = np.array([(1, 2), (3, -4), (-5, 6), (-7, -8)], ComplexInt32DType)

    value_out = convert_complex(ComplexInt32DType, value_in)

    assert_type(value_out, npt.NDArray[ComplexInt32Base])
    assert value_out is value_in


def test___complex64_array_to_complexint32_array___convert_complex___converts_array() -> None:
    value_in = np.array([1 + 2j, 3 - 4j, -5 + 6j, -7 - 8j], np.complex64)

    value_out = convert_complex(ComplexInt32DType, value_in)

    assert_type(value_out, npt.NDArray[ComplexInt32Base])
    assert isinstance(value_out, np.ndarray) and value_out.dtype == ComplexInt32DType
    assert [x.item() for x in value_out] == [(1, 2), (3, -4), (-5, 6), (-7, -8)]


def test___complex64_array_to_complex64_array___convert_complex___returns_original_array() -> None:
    value_in = np.array([1 + 2j, 3 - 4j, -5 + 6j, -7 - 8j], np.complex64)

    value_out = convert_complex(np.complex64, value_in)

    assert_type(value_out, npt.NDArray[np.complex64])
    assert value_out is value_in


def test___complex64_array_to_complex128_array___convert_complex___converts_array() -> None:
    value_in = np.array([1.23 + 4.56j, 6.78 - 9.01j], np.complex64)

    value_out = convert_complex(np.complex128, value_in)

    assert_type(value_out, npt.NDArray[np.complex128])
    assert isinstance(value_out, np.ndarray) and value_out.dtype == np.complex128
    assert list(value_out) == pytest.approx([1.23 + 4.56j, 6.78 - 9.01j])


def test___complex128_array_to_complexint32_array___convert_complex___converts_array() -> None:
    value_in = np.array([1 + 2j, 3 - 4j, -5 + 6j, -7 - 8j], np.complex128)

    value_out = convert_complex(ComplexInt32DType, value_in)

    assert_type(value_out, npt.NDArray[ComplexInt32Base])
    assert isinstance(value_out, np.ndarray) and value_out.dtype == ComplexInt32DType
    assert [x.item() for x in value_out] == [(1, 2), (3, -4), (-5, 6), (-7, -8)]


def test___complex128_array_to_complex64_array___convert_complex___converts_array() -> None:
    value_in = np.array([1.23 + 4.56j, 6.78 - 9.01j], np.complex128)

    value_out = convert_complex(np.complex64, value_in)

    assert_type(value_out, npt.NDArray[np.complex64])
    assert isinstance(value_out, np.ndarray) and value_out.dtype == np.complex64
    assert list(value_out) == pytest.approx([1.23 + 4.56j, 6.78 - 9.01j])


def test___complex128_array_to_complex128_array___convert_complex___returns_original_array() -> (
    None
):
    value_in = np.array([1.23 + 4.56j, 6.78 - 9.01j], np.complex128)

    value_out = convert_complex(np.complex128, value_in)

    assert_type(value_out, npt.NDArray[np.complex128])
    assert value_out is value_in


def test___2d_complexint32_array_to_complex128_array___convert_complex___preserves_shape() -> None:
    value_in = np.array(
        [[(1, 2), (3, -4)], [(-5, 6), (-7, -8)], [(9, 10), (11, 12)]], ComplexInt32DType
    )

    value_out = convert_complex(np.complex128, value_in)

    # Use npt.NDArray because np.array() can't infer the array shape when type checking.
    assert_type(value_out, npt.NDArray[np.complex128])
    assert isinstance(value_out, np.ndarray) and value_out.shape == (3, 2)
    assert [list(x) for x in value_out] == [
        [1 + 2j, 3 - 4j],
        [-5 + 6j, -7 - 8j],
        [9 + 10j, 11 + 12j],
    ]


def test___2d_complex64_array_to_complex128_array___convert_complex___preserves_shape() -> None:
    value_in = np.array([[1 + 2j, 3 - 4j], [-5 + 6j, -7 - 8j], [9 + 10j, 11 + 12j]], np.complex64)

    value_out = convert_complex(np.complex128, value_in)

    # Use npt.NDArray because np.array() can't infer the array shape when type checking.
    assert_type(value_out, npt.NDArray[np.complex128])
    assert isinstance(value_out, np.ndarray) and value_out.shape == (3, 2)
    assert [list(x) for x in value_out] == [
        [1 + 2j, 3 - 4j],
        [-5 + 6j, -7 - 8j],
        [9 + 10j, 11 + 12j],
    ]


def test___arrays_with_static_shape___convert_complex___preserves_static_and_runtime_shape() -> (
    None
):
    # np.zeros() can infer the array shape when type checking because it takes a shape argument.
    value_in_1d = np.zeros(3, np.complex64)
    value_in_2d = np.zeros((4, 5), np.complex128)
    value_in_3d = np.zeros((6, 7, 8), ComplexInt32DType)

    value_out_1d = convert_complex(np.complex128, value_in_1d)
    value_out_2d = convert_complex(ComplexInt32DType, value_in_2d)
    value_out_3d = convert_complex(np.complex64, value_in_3d)

    assert_type(value_out_1d, np.ndarray[tuple[int], np.dtype[np.complex128]])
    assert_type(value_out_2d, np.ndarray[tuple[int, int], np.dtype[ComplexInt32Base]])
    assert_type(value_out_3d, np.ndarray[tuple[int, int, int], np.dtype[np.complex64]])
    assert isinstance(value_out_1d, np.ndarray) and value_out_1d.shape == (3,)
    assert isinstance(value_out_2d, np.ndarray) and value_out_2d.shape == (4, 5)
    assert isinstance(value_out_3d, np.ndarray) and value_out_3d.shape == (6, 7, 8)


###############################################################################
# convert scalars
###############################################################################
def test___complexint32_scalar_to_complex128_scalar___convert_complex___converts_scalar() -> None:
    value_in = np.array([(1, 2)], ComplexInt32DType)[0]
    assert_type(value_in, Any)  # ¯\_(ツ)_/¯

    value_out = convert_complex(np.complex128, value_in)

    # Mypy infers np.ndarray[Any, Any], which seems wrong.
    # Pyright infers np.ndarray[Any, np.dtype[np.complex128]], which seems right.
    assert_type(value_out, np.ndarray[Any, np.dtype[np.complex128]])  # type: ignore[assert-type]
    assert isinstance(value_out, np.complex128)
    assert value_out == (1 + 2j)


def test___complex128_scalar_to_complexint32_scalar___convert_complex___converts_scalar() -> None:
    value_in = np.complex128(1 + 2j)

    value_out = convert_complex(ComplexInt32DType, value_in)

    assert_type(value_out, np.ndarray[tuple[()], np.dtype[ComplexInt32Base]])
    assert isinstance(value_out, ComplexInt32Base)
    assert value_out.item() == (1, 2)


def test___complexint32_scalar_to_complexint32_scalar___convert_complex___returns_original_scalar() -> (
    None
):
    value_in = np.array([(1, 2)], ComplexInt32DType)[0]
    assert_type(value_in, Any)  # ¯\_(ツ)_/¯

    value_out = convert_complex(ComplexInt32DType, value_in)

    # Mypy infers np.ndarray[Any, Any], which seems wrong.
    # Pyright infers np.ndarray[Any, np.dtype[np.void]], which seems right.
    assert_type(value_out, np.ndarray[Any, np.dtype[np.void]])  # type: ignore[assert-type]
    assert value_out is value_in


def test___complex64_scalar_to_complex128_scalar___convert_complex___converts_scalar() -> None:
    value_in = np.complex64(1.23 + 4.56j)

    value_out = convert_complex(np.complex128, value_in)

    assert_type(value_out, np.ndarray[tuple[()], np.dtype[np.complex128]])
    assert isinstance(value_out, np.complex128)
    assert value_out == pytest.approx((1.23 + 4.56j))
