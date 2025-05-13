"""NumPy 1.x compatibility shims."""

from __future__ import annotations

import numpy as np

from nitypes._version import parse_version

numpy_version_info = parse_version(np.__version__)
"""The NumPy version as a tuple."""

if numpy_version_info >= (2, 0, 0):
    from numpy import asarray, isdtype, long, ulong
else:
    from nitypes._numpy1x import asarray, isdtype, long, ulong  # type: ignore[no-redef]


__all__ = [
    "asarray",
    "isdtype",
    "long",
    "numpy_version_info",
    "ulong",
]
