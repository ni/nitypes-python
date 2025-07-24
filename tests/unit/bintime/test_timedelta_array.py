from __future__ import annotations

from typing_extensions import assert_type

from nitypes.bintime import TimeDelta
from nitypes.bintime import TimeDeltaArray


def test___no_args___construct___returns_empty_array() -> None:
    value = TimeDeltaArray()

    assert_type(value, TimeDeltaArray)
    assert isinstance(value, TimeDeltaArray)
    assert len(value._array) == 0


def test___list_arg___construct___returns_matching_array() -> None:
    arg = [TimeDelta(1), TimeDelta(20), TimeDelta(500)]
    value = TimeDeltaArray(arg)

    assert_type(value, TimeDeltaArray)
    assert isinstance(value, TimeDeltaArray)
    assert len(value._array) == len(arg)
    assert tuple(value._array[0]) == TimeDelta(1).to_tuple()
    assert tuple(value._array[1]) == TimeDelta(20).to_tuple()
    assert tuple(value._array[2]) == TimeDelta(500).to_tuple()
