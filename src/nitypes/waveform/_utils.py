from __future__ import annotations

import operator
import sys
from typing import SupportsIndex

import numpy as np
import numpy.typing as npt


def add_note(exception: Exception, note: str) -> None:
    """Add a note to an exception.

    >>> try:
    ...     raise ValueError("Oh no")
    ... except Exception as e:
    ...     add_note(e, "p.s. This is bad")
    ...     raise
    Traceback (most recent call last):
    ...
    ValueError: Oh no
    p.s. This is bad
    """
    if sys.version_info >= (3, 11):
        exception.add_note(note)
    else:
        message = exception.args[0] + "\n" + note
        exception.args = (message,) + exception.args[1:]


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
            f"The {arg_description} must be a non-negative integer.\n\n" f"Provided value: {value}"
        )
    return value


def validate_dtype(dtype: npt.DTypeLike, supported_dtypes: tuple[npt.DTypeLike, ...]) -> None:
    """Validate a dtype-like object against a tuple of supported dtype-like objects.

    >>> validate_dtype(np.float64, (np.float64, np.intc, np.long,))
    >>> validate_dtype("float64", (np.float64, np.intc, np.long,))
    >>> validate_dtype(np.float64, (np.byte, np.short, np.intc, np.int_, np.long, np.longlong))
    Traceback (most recent call last):
    ...
    TypeError: The requested data type is not supported.
    <BLANKLINE>
    Data type: float64
    Supported data types: int8, int16, int32, int64
    """
    if not isinstance(dtype, (type, np.dtype)):
        dtype = np.dtype(dtype)
    if not np.isdtype(dtype, supported_dtypes):
        # Remove duplicate names because distinct types (e.g. int vs. long) may have the same name
        # ("int32").
        supported_dtype_names = {np.dtype(d).name: None for d in supported_dtypes}.keys()
        raise TypeError(
            "The requested data type is not supported.\n\n"
            f"Data type: {np.dtype(dtype)}\n"
            f"Supported data types: {', '.join(supported_dtype_names)}"
        )
