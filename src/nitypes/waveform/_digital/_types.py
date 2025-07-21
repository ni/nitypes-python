from __future__ import annotations

from typing import TypeVar, Union

import numpy as np
from typing_extensions import TypeAlias

from nitypes._numpy import bool as _np_bool

__all__ = [
    "AnyDigitalPort",
    "AnyDigitalState",
    "TDigitalState",
    "TOtherDigitalState",
    "DIGITAL_PORT_DTYPES",
    "DIGITAL_STATE_DTYPES",
]

AnyDigitalPort: TypeAlias = Union[np.uint8, np.uint16, np.uint32]
"""Type alias for any digital port data type.

This type alias is a union of the following types:

* :class:`numpy.uint8`
* :class:`numpy.uint16`
* :class:`numpy.uint32`
"""

# np.byte == np.int8, np.ubyte == np.uint8
AnyDigitalState: TypeAlias = Union[_np_bool, np.int8, np.uint8]
"""Type alias for any digital state data type.

This type alias is a union of the following types:

* :class:`numpy.bool` (NumPy 2.x) or :class:`numpy.bool_` (NumPy 1.x)
* :class:`numpy.int8`
* :class:`numpy.uint8`
"""

TDigitalState = TypeVar("TDigitalState", bound=AnyDigitalState)
"""Type variable with a bound of :any:`AnyDigitalState`."""

TOtherDigitalState = TypeVar("TOtherDigitalState", bound=AnyDigitalState)
"""Another type variable with a bound of :any:`AnyDigitalState`."""

DIGITAL_PORT_DTYPES = (np.uint8, np.uint16, np.uint32)
"""Tuple of types corresponding to :any:`AnyDigitalPort`."""

DIGITAL_STATE_DTYPES = (_np_bool, np.int8, np.uint8)
"""Tuple of types corresponding to :any:`AnyDigitalState`."""
