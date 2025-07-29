from __future__ import annotations

from typing import Any

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
    ("timedelta_list", "expected_length"),
    (
        (None, 0),
        ([TimeDelta(3.14)], 1),
        ([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)], 3),
    ),
)
def test___timedelta_array___get_len___returns_length(
    timedelta_list: list[TimeDelta], expected_length: int
) -> None:
    value = TimeDeltaArray(timedelta_list)

    length = len(value)

    assert length == expected_length


@pytest.mark.parametrize(
    ("timedelta_list", "indexer", "raised_exception"),
    (
        # First index
        (None, 0, IndexError()),
        ([TimeDelta(3.14)], 0, None),
        ([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)], 0, None),
        # Last index
        (None, -1, IndexError()),
        ([TimeDelta(3.14)], -1, None),
        ([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)], -1, None),
        # Out of bounds index
        (None, 10, IndexError()),
        ([TimeDelta(3.14)], 10, IndexError()),
        ([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)], 10, IndexError()),
    ),
)
def test__timedelta_array___index___returns_timedelta(
    timedelta_list: list[TimeDelta], indexer: int, raised_exception: BaseException | None
) -> None:
    value = TimeDeltaArray(timedelta_list)

    if raised_exception:
        with pytest.raises(type(raised_exception)):
            entry = value[indexer]
    else:
        entry = value[indexer]
        assert entry == timedelta_list[indexer]


def test___timedelta_array___slice___returns_slice() -> None:
    value = TimeDeltaArray(
        [
            TimeDelta(-1),
            TimeDelta(3.14),
            TimeDelta(20.26),
            TimeDelta(500),
            TimeDelta(0x12345678_90ABCDEF),
        ]
    )

    selected = value[::2]

    expected = TimeDeltaArray(
        [
            TimeDelta(-1),
            TimeDelta(20.26),
            TimeDelta(0x12345678_90ABCDEF),
        ]
    )
    assert all(selected._array == expected._array)


@pytest.mark.parametrize(
    "indexer",
    (
        "0",
        1.0,
        True,
        None,
        [1, 2, 3],
    ),
)
def test___timedelta_array___index_unsupported___raises(indexer: Any) -> None:
    value = TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)])

    with pytest.raises(TypeError):
        _ = value[indexer]
