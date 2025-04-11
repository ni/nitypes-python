from __future__ import annotations

import operator
from typing import SupportsIndex


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
