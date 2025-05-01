from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from nitypes._typing import assert_type
from nitypes.complex import ComplexInt32, _ComplexInt32Scalar


def test___construct___sets_fields() -> None:
    value = ComplexInt32(1, 2)

    assert_type(value, ComplexInt32)
    assert isinstance(value, ComplexInt32)
    assert value.real == 1
    assert value.imag == 2


@pytest.mark.parametrize("real, imag", [(-32769, 0), (32768, 0), (0, -32769), (0, 32768)])
def test___out_of_range___construct___raises_overflow_error(real: int, imag: int) -> None:
    with pytest.raises(OverflowError):
        _ = ComplexInt32(real, imag)


def test___tuple_list_and_complexint32_dtype___np_array___constructs_ndarray() -> None:
    array = np.array([(1, 2), (3, 4)], ComplexInt32)

    assert_type(array, npt.NDArray[_ComplexInt32Scalar])
    assert isinstance(array, np.ndarray)
    assert array.dtype == ComplexInt32
    assert array.shape == (2,)
    assert array[0].item() == (1, 2)
    assert array[1].item() == (3, 4)


def test___complexint32_list_and_complexint32_dtype___np_array___constructs_ndarray() -> None:
    array = np.array([ComplexInt32(1, 2), ComplexInt32(3, 4)], ComplexInt32)

    assert_type(array, npt.NDArray[_ComplexInt32Scalar])
    assert isinstance(array, np.ndarray)
    assert array.dtype == ComplexInt32
    assert array.shape == (2,)
    assert array[0].item() == (1, 2)
    assert array[1].item() == (3, 4)
