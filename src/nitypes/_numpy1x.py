"""NumPy 1.x compatibility shim implementations."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def asarray(  # noqa: D103 - missing docstring in public function
    a: npt.ArrayLike, dtype: npt.DTypeLike = None, *, copy: bool | None = None
) -> npt.NDArray[Any]:
    b = np.asarray(a, dtype)
    made_copy = b is not a and b.base is None
    if copy is True and not made_copy:
        b = np.copy(b)
    if copy is False and made_copy:
        raise ValueError("Unable to avoid copy while creating an array as requested.")
    return b


def isdtype(  # noqa: D103 - missing docstring in public function
    dtype: type[Any] | np.dtype[Any], kind: npt.DTypeLike | tuple[npt.DTypeLike, ...]
) -> bool:
    if isinstance(kind, tuple):
        return any(dtype == k for k in kind)
    else:
        return dtype == kind
