from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from typing_extensions import assert_type

from nitypes.bintime import TimeDelta
from nitypes.bintime import TimeDeltaArray


#############
# Constructor
#############


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


#######
# len()
#######


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


###############
# __getitem__()
###############


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
    assert np.array_equal(selected._array, expected._array)


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


###############
# __setitem__()
###############


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
def test___timedelta_array___set_by_index___updates_array(
    timedelta_list: list[TimeDelta], indexer: int, raised_exception: BaseException | None
) -> None:
    value = TimeDeltaArray(timedelta_list)
    new_entry = TimeDelta(-123.456)

    if raised_exception:
        with pytest.raises(type(raised_exception)):
            value[indexer] = new_entry
    else:
        value[indexer] = new_entry
        assert value[indexer] == new_entry


@pytest.mark.parametrize(
    ("indexer", "new_entries", "expected_result"),
    (
        (
            slice(1, 4),
            [TimeDelta(0), TimeDelta(1), TimeDelta(2)],
            TimeDeltaArray(
                [
                    TimeDelta(-1),
                    TimeDelta(0),
                    TimeDelta(1),
                    TimeDelta(2),
                    TimeDelta(0x12345678_90ABCDEF),
                ]
            ),
        ),
        (
            slice(None, None, 2),
            [TimeDelta(0), TimeDelta(1), TimeDelta(2)],
            TimeDeltaArray(
                [TimeDelta(0), TimeDelta(3.14), TimeDelta(1), TimeDelta(500), TimeDelta(2)]
            ),
        ),
    ),
)
def test___timedelta_array___set_by_slice___updates_array(
    indexer: slice, new_entries: list[TimeDelta], expected_result: TimeDeltaArray
) -> None:
    value = TimeDeltaArray(
        [
            TimeDelta(-1),
            TimeDelta(3.14),
            TimeDelta(20.26),
            TimeDelta(500),
            TimeDelta(0x12345678_90ABCDEF),
        ]
    )

    value[indexer] = new_entries

    assert np.array_equal(value._array, expected_result._array)


@pytest.mark.parametrize(
    ("indexer"),
    (
        "0",
        1.0,
        True,
        None,
        [1, 2, 3],
    ),
)
def test___timedelta_array___set_unsupported_index___raises(indexer: Any) -> None:
    value = TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)])
    new_entry = TimeDelta(-100)

    with pytest.raises(TypeError):
        value[indexer] = new_entry


@pytest.mark.parametrize(
    "new_entry",
    (
        "0",
        1.0,
        True,
        None,
        [1, 2, 3],
    ),
)
def test___timedelta_array___set_unsupported_value___raises(new_entry: Any) -> None:
    value = TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)])

    with pytest.raises(TypeError):
        value[0] = new_entry


def test___timedelta_array___set_with_not_iterable___raises() -> None:
    value = TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)])

    with pytest.raises(TypeError):
        value[1:] = TimeDelta(-100)  # type: ignore # validating a type-error case


@pytest.mark.parametrize(
    ("indexer"),
    (
        slice(1, None, None),  # Slice is too long for set values
        slice(None, None, 4),  # Slice is too short for set values
    ),
)
def test___timedelta_array___set_slice_wrong_length___raises(indexer: slice) -> None:
    value = TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500), TimeDelta(3000.125)])

    with pytest.raises(ValueError):
        value[indexer] = [TimeDelta(0), TimeDelta(10)]


@pytest.mark.parametrize(
    ("new_entries"),
    (
        ["ab", "cd"],
        [1.0, 2.0],
        [True, False],
        [None, None],
        [TimeDelta(0), TimeDelta(15).to_tuple()],
    ),
)
def test___timedelta_array___set_mixed_slice___raises(new_entries: list[Any]) -> None:
    value = TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500), TimeDelta(3000.125)])

    with pytest.raises(TypeError):
        value[::2] = new_entries


###############
# __delitem__()
###############


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
def test___timedelta_array___delete_by_index___removes_item(
    timedelta_list: list[TimeDelta], indexer: int, raised_exception: BaseException | None
) -> None:
    value = TimeDeltaArray(timedelta_list)

    if raised_exception:
        with pytest.raises(type(raised_exception)):
            del value[indexer]
    else:
        modified = timedelta_list.copy()
        del modified[indexer]
        del value[indexer]
        expected = TimeDeltaArray(modified)
        assert np.array_equal(value._array, expected._array)


@pytest.mark.parametrize(
    ("indexer", "expected_result"),
    (
        (
            slice(1, 4),
            TimeDeltaArray(
                [
                    TimeDelta(-1),
                    TimeDelta(0x12345678_90ABCDEF),
                ]
            ),
        ),
        (
            slice(None, None, 2),
            TimeDeltaArray(
                [
                    TimeDelta(3.14),
                    TimeDelta(500),
                ]
            ),
        ),
    ),
)
def test___timedelta_array___delete_by_slice___removes_items(
    indexer: slice, expected_result: TimeDeltaArray
) -> None:
    value = TimeDeltaArray(
        [
            TimeDelta(-1),
            TimeDelta(3.14),
            TimeDelta(20.26),
            TimeDelta(500),
            TimeDelta(0x12345678_90ABCDEF),
        ]
    )

    del value[indexer]

    assert np.array_equal(value._array, expected_result._array)


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
def test___timedelta_array___delete_unsupported_index___raises(indexer: Any) -> None:
    value = TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)])

    with pytest.raises(TypeError):
        del value[indexer]
