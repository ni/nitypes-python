from __future__ import annotations

from typing import Any, TypeVar, overload

import numpy as np
import numpy.typing as npt

from nitypes._arguments import validate_dtype
from nitypes._exceptions import unsupported_dtype
from nitypes.complex._dtypes import ComplexInt32DType

_ScalarType = TypeVar("_ScalarType", bound=np.generic)

_COMPLEX_DTYPES = (
    np.complex64,
    np.complex128,
    ComplexInt32DType,
)

_FIELD_DTYPE = {
    np.dtype(np.complex64): np.float32,
    np.dtype(np.complex128): np.float64,
    ComplexInt32DType: np.int16,
}


@overload
def convert_complex(
    requested_dtype: type[_ScalarType] | np.dtype[_ScalarType], value: npt.NDArray[Any]
) -> npt.NDArray[_ScalarType]: ...


@overload
def convert_complex(
    requested_dtype: npt.DTypeLike, value: npt.NDArray[Any]
) -> npt.NDArray[Any]: ...


def convert_complex(requested_dtype: npt.DTypeLike, value: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Convert a NumPy array of complex numbers to the specified dtype.

    Args:
        requested_dtype: The NumPy data type to convert to. This must be a complex number data type.
        value: The NumPy array or scalar to convert.
    """
    validate_dtype(requested_dtype, _COMPLEX_DTYPES)
    if requested_dtype == value.dtype:
        return value
    elif requested_dtype == ComplexInt32DType or value.dtype == ComplexInt32DType:
        if value.shape == ():
            return _convert_complexint32_scalar(requested_dtype, value)
        else:
            return _convert_complexint32_array(requested_dtype, value)
    else:
        return value.astype(requested_dtype)


def _convert_complexint32_scalar(
    requested_dtype: npt.DTypeLike | type[_ScalarType] | np.dtype[_ScalarType],
    value: npt.NDArray[Any],
) -> npt.NDArray[_ScalarType]:
    # ndarray.view on scalars requires the source and destination types to have the same size, so
    # reshape the scalar into an 1-element array before converting and index it afterwards.
    # Mypy currently thinks that the index operator returns Any.
    return _convert_complexint32_array(requested_dtype, value.reshape(1))[0]  # type: ignore[no-any-return]


def _convert_complexint32_array(
    requested_dtype: npt.DTypeLike | type[_ScalarType] | np.dtype[_ScalarType],
    value: npt.NDArray[Any],
) -> npt.NDArray[_ScalarType]:
    if not isinstance(requested_dtype, np.dtype):
        requested_dtype = np.dtype(requested_dtype)

    requested_field_dtype = _FIELD_DTYPE.get(requested_dtype)
    if requested_field_dtype is None:
        raise unsupported_dtype("requested data type", requested_dtype, _COMPLEX_DTYPES)

    value_field_dtype = _FIELD_DTYPE.get(value.dtype)
    if value_field_dtype is None:
        raise unsupported_dtype("array data type", value.dtype, _COMPLEX_DTYPES)

    return value.view(value_field_dtype).astype(requested_field_dtype).view(requested_dtype)
