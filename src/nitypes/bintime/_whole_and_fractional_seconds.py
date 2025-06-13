from __future__ import annotations

from typing import NamedTuple


class WholeAndFractionalSeconds(NamedTuple):
    """A named tuple containing 64bit ints for whole and fractional seconds."""

    whole_seconds: int
    fractional_seconds: int
