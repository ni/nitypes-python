from __future__ import annotations

from typing import NamedTuple


class TimeValueTuple(NamedTuple):
    """A named tuple containing the whole seconds and fractional seconds parts of a time value."""

    whole_seconds: int
    """The whole seconds portion of a binary time value. This should be an int64."""

    fractional_seconds: int
    """The fractional seconds portion of a binary time value. This should be a uint64."""
