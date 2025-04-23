from __future__ import annotations

import operator
import sys
from typing import SupportsFloat, SupportsIndex

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


def arg_to_float(
    arg_description: str, value: SupportsFloat | None, default_value: float | None = None
) -> float:
    """Convert an argument to a float.

    >>> arg_to_float("xyz", 1.234)
    1.234
    >>> arg_to_float("xyz", 1234)
    1234.0
    >>> arg_to_float("xyz", np.float64(1.234))
    np.float64(1.234)
    >>> arg_to_float("xyz", np.float32(1.234))  # doctest: +ELLIPSIS
    1.233999...
    >>> arg_to_float("xyz", 1.234, 5.0)
    1.234
    >>> arg_to_float("xyz", None, 5.0)
    5.0
    >>> arg_to_float("xyz", None)
    Traceback (most recent call last):
    ...
    TypeError: The xyz must be a floating point number.
    <BLANKLINE>
    Provided value: None
    >>> arg_to_float("xyz", "1.234")
    Traceback (most recent call last):
    ...
    TypeError: The xyz must be a floating point number.
    <BLANKLINE>
    Provided value: '1.234'
    """
    if value is None:
        if default_value is None:
            raise TypeError(
                f"The {arg_description} must be a floating point number.\n\n"
                f"Provided value: {value!r}"
            )
        value = default_value

    if not isinstance(value, float):
        try:
            # Use value.__float__() because float(value) also accepts strings.
            return value.__float__()
        except Exception:
            raise TypeError(
                f"The {arg_description} must be a floating point number.\n\n"
                f"Provided value: {value!r}"
            ) from None

    return value


def arg_to_int(
    arg_description: str, value: SupportsIndex | None, default_value: int | None = None
) -> int:
    """Convert an argument to a signed integer.

    >>> arg_to_int("xyz", 1234)
    1234
    >>> arg_to_int("xyz", 1234, -1)
    1234
    >>> arg_to_int("xyz", None, -1)
    -1
    >>> arg_to_int("xyz", None)
    Traceback (most recent call last):
    ...
    TypeError: The xyz must be an integer.
    <BLANKLINE>
    Provided value: None
    >>> arg_to_int("xyz", 1.234)
    Traceback (most recent call last):
    ...
    TypeError: The xyz must be an integer.
    <BLANKLINE>
    Provided value: 1.234
    >>> arg_to_int("xyz", "1234")
    Traceback (most recent call last):
    ...
    TypeError: The xyz must be an integer.
    <BLANKLINE>
    Provided value: '1234'
    """
    if value is None:
        if default_value is None:
            raise TypeError(
                f"The {arg_description} must be an integer.\n\n" f"Provided value: {value!r}"
            )
        value = default_value

    if not isinstance(value, int):
        try:
            return operator.index(value)
        except Exception:
            raise TypeError(
                f"The {arg_description} must be an integer.\n\n" f"Provided value: {value!r}"
            ) from None

    return value


def arg_to_uint(
    arg_description: str, value: SupportsIndex | None, default_value: int | None = None
) -> int:
    """Convert an argument to an unsigned integer.

    >>> arg_to_uint("xyz", 1234)
    1234
    >>> arg_to_uint("xyz", 1234, 5000)
    1234
    >>> arg_to_uint("xyz", None, 5000)
    5000
    >>> arg_to_uint("xyz", -1234)
    Traceback (most recent call last):
    ...
    ValueError: The xyz must be a non-negative integer.
    <BLANKLINE>
    Provided value: -1234
    >>> arg_to_uint("xyz", "1234")
    Traceback (most recent call last):
    ...
    TypeError: The xyz must be an integer.
    <BLANKLINE>
    Provided value: '1234'
    """
    value = arg_to_int(arg_description, value, default_value)
    if value < 0:
        raise ValueError(
            f"The {arg_description} must be a non-negative integer.\n\n"
            f"Provided value: {value!r}"
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
