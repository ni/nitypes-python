from __future__ import annotations

import operator
from typing import SupportsIndex

import numpy as np
import numpy.typing as npt


def arg_to_int(arg_description: str, value: SupportsIndex | None, default_value: int = 0) -> int:
    """Convert an argument to a signed integer."""
    if value is None:
        return default_value
    return operator.index(value)


def arg_to_uint(arg_description: str, value: SupportsIndex | None, default_value: int = 0) -> int:
    """Convert an argument to an unsigned integer."""
    value = arg_to_int(arg_description, value, default_value)
    if value < 0:
        raise ValueError(
            f"The {arg_description} must be a non-negative integer.\n\nProvided value: {value}"
        )
    return value


def validate_dtype(dtype: npt.DTypeLike, supported_dtypes: tuple[npt.DTypeLike, ...]) -> None:
    """Validate a dtype-like object against a tuple of supported dtype-like objects.

    >>> validate_dtype(np.float64, (np.float64, np.int32,))
    >>> validate_dtype("float64", (np.float64, np.int32,))
    >>> validate_dtype(np.int8, (np.float64, np.int32,))
    Traceback (most recent call last):
    ...
    TypeError: The requested data type is not supported.
    <BLANKLINE>
    Data type: int8
    Supported data types: float64, int32
    """
    if not isinstance(dtype, (type, np.dtype)):
        dtype = np.dtype(dtype)
    if not np.isdtype(dtype, supported_dtypes):
        raise TypeError(
            "The requested data type is not supported.\n\n"
            f"Data type: {np.dtype(dtype)}\n"
            f"Supported data types: {', '.join(str(np.dtype(d)) for d in supported_dtypes)}"
        )
