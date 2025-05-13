"""NumPy 1.x compatibility shims."""

from __future__ import annotations

import numpy as np

from nitypes._version import parse_version

numpy_version_info = parse_version(np.__version__)
"""The NumPy version as a tuple."""

if numpy_version_info >= (2, 0, 0):
    # In NumPy 2.x, np.long is an alias for np.int64 and np.ulong is an alias for np.uint64, at
    # least on 64-bit platforms.
    from numpy import asarray, isdtype, long, ulong
else:
    # In NumPy 1.x, np.long is an alias for int and np.ulong does not exist.
    # https://numpy.org/doc/1.22/release/1.20.0-notes.html#using-the-aliases-of-builtin-types-like-np-int-is-deprecated
    from numpy import int_ as long, uint as ulong  # type: ignore[assignment]

    from nitypes._numpy1x import asarray, isdtype  # type: ignore[no-redef]


__all__ = [
    "asarray",
    "isdtype",
    "long",
    "ulong",
]
