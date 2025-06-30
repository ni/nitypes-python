from __future__ import annotations

from typing import TypeVar, Union

import numpy as np
from typing_extensions import TypeAlias

from nitypes._numpy import bool as _np_bool

__all__ = [
    "_AnyPort",
    "_AnyState",
    "_TState",
    "_TOtherState",
    "_DIGITAL_PORT_DTYPES",
    "_DIGITAL_STATE_DTYPES",
]

_AnyPort: TypeAlias = Union[np.uint8, np.uint16, np.uint32]

# np.byte == np.int8, np.ubyte == np.uint8
_AnyState: TypeAlias = Union[_np_bool, np.int8, np.uint8]
_TState = TypeVar("_TState", bound=_AnyState)
_TOtherState = TypeVar("_TOtherState", bound=_AnyState)

_DIGITAL_PORT_DTYPES = (np.uint8, np.uint16, np.uint32)
_DIGITAL_STATE_DTYPES = (_np_bool, np.int8, np.uint8)
