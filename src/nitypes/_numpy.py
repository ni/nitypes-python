"""NumPy 1.x compatibility shims."""

from __future__ import annotations

import sys

import numpy as np

from nitypes._version import parse_version

numpy_version_info = parse_version(np.__version__)
"""The NumPy version as a tuple."""

if numpy_version_info >= (2, 0, 0):
    # In NumPy 2.x, np.long and np.ulong are equivalent to long and unsigned long in C, following
    # the platform's data model.
    from numpy import asarray, isdtype, long, ulong
else:
    # In NumPy 1.x, np.long is an alias for int and np.ulong does not exist.
    # https://numpy.org/doc/1.22/release/1.20.0-notes.html#using-the-aliases-of-builtin-types-like-np-int-is-deprecated
    if sys.platform == "win32":
        # 32-bit Windows has an ILP32 data model and 64-bit Windows has an LLP64 data model, so
        # long is 32-bit.
        from numpy import (  # type: ignore[assignment,unused-ignore]
            int32 as long,
            uint32 as ulong,
        )
    else:
        # Assume other 32-bit platforms have an ILP32 data model and other 64-bit platforms have an
        # LP64 data model, so long is pointer-sized.
        from numpy import intp as long, uintp as ulong

    from nitypes._numpy1x import asarray, isdtype  # type: ignore[no-redef]


__all__ = [
    "asarray",
    "isdtype",
    "long",
    "ulong",
]
