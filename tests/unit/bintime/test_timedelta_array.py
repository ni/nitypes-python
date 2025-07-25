from __future__ import annotations

import pytest
from typing_extensions import assert_type

from nitypes.bintime import TimeDelta
from nitypes.bintime import TimeDeltaArray


def test___no_args___construct___returns_empty_array() -> None:
    value = TimeDeltaArray()

    assert_type(value, TimeDeltaArray)
    assert isinstance(value, TimeDeltaArray)
    assert len(value._array) == 0


def test___list_arg___construct___returns_matching_array() -> None:
    arg = [TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)]
    value = TimeDeltaArray(arg)

    assert_type(value, TimeDeltaArray)
    assert isinstance(value, TimeDeltaArray)
    assert len(value._array) == len(arg)
    assert (value._array[0]["msb"], value._array[0]["lsb"]) == TimeDelta(-1).to_tuple()
    assert (value._array[1]["msb"], value._array[1]["lsb"]) == TimeDelta(20.26).to_tuple()
    assert (value._array[2]["msb"], value._array[2]["lsb"]) == TimeDelta(500).to_tuple()


@pytest.mark.parametrize(
    "timedelta_list,exception",
    (
        (None, IndexError()),
        ([TimeDelta(3.14)], None),
        ([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)], None),
    ),
)
def test__timedelta_array___index_first___returns_timedelta(
    timedelta_list: list[TimeDelta], exception: BaseException | None
) -> None:
    value = TimeDeltaArray(timedelta_list)

    if exception:
        with pytest.raises(type(exception)):
            entry = value[0]
    else:
        entry = value[0]
        assert entry == timedelta_list[0]
