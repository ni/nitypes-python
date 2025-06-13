from __future__ import annotations

from typing import Any

from nitypes.waveform import Timing


def assert_deep_copy(value: Timing[Any, Any, Any], other: Timing[Any, Any, Any]) -> None:
    """Assert that value is a deep copy of other."""
    assert value == other
    assert value is not other
    if other._timestamp is not None:
        assert value._timestamp is not other._timestamp
    if other._time_offset is not None:
        assert value._time_offset is not other._time_offset
    if other._sample_interval is not None:
        assert value._sample_interval is not other._sample_interval
    if other._timestamps is not None:
        assert value._timestamps is not other._timestamps


def assert_shallow_copy(value: Timing[Any, Any, Any], other: Timing[Any, Any, Any]) -> None:
    """Assert that value is a shallow copy of other."""
    assert value == other
    assert value is not other
    assert value._timestamp is other._timestamp
    assert value._time_offset is other._time_offset
    assert value._sample_interval is other._sample_interval
    assert value._timestamps is other._timestamps
