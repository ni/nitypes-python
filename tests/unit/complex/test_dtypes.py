from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from typing_extensions import assert_type

from nitypes.complex import ComplexInt32Base, ComplexInt32DType


def test___complexint32_dtype___np_array___constructs_array_with_dtype() -> None:
    value = np.array([(1, 2), (3, -4), (-5, 6), (-7, -8)], ComplexInt32DType)

    assert_type(value, npt.NDArray[ComplexInt32Base])
    assert isinstance(value, np.ndarray) and value.dtype == ComplexInt32DType
    assert [x.item() for x in value] == [(1, 2), (3, -4), (-5, 6), (-7, -8)]


def test___complexint32_dtype___np_zeros___constructs_array_with_dtype() -> None:
    value = np.zeros(3, ComplexInt32DType)

    assert_type(value, np.ndarray[tuple[int], np.dtype[ComplexInt32Base]])
    assert isinstance(value, np.ndarray) and value.dtype == ComplexInt32DType
    assert [x.item() for x in value] == [(0, 0), (0, 0), (0, 0)]


def test___complexint32_array___index___returns_complexint32_scalar() -> None:
    array = np.array([(1, 2), (3, -4)], ComplexInt32DType)

    value = array[1]

    assert_type(value, Any)  # ¯\_(ツ)_/¯
    assert isinstance(value, ComplexInt32Base)  # alias for np.void
    assert value["real"] == 3
    assert value["imag"] == -4


def test___complexint32_arrays___add___raises_type_error() -> None:
    left = np.array([(1, 2), (3, -4)], ComplexInt32DType)
    right = np.array([(-5, 6), (-7, -8)], ComplexInt32DType)

    with pytest.raises(TypeError):
        _ = left + right  # type: ignore[operator]


def test___complexint32_array_and_int16_array___add___raises_type_error() -> None:
    left = np.array([(1, 2), (3, -4)], ComplexInt32DType)
    right = np.array([5, -6], np.int16)

    with pytest.raises(TypeError):
        _ = left + right  # type: ignore[operator]


def test___unknown_structured_dtype___equality___not_equal() -> None:
    dtype = np.dtype([("a", np.int16), ("b", np.int16)])

    assert dtype != ComplexInt32DType
