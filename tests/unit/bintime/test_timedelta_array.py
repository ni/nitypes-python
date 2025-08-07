from __future__ import annotations

import copy
import pickle
from typing import Any, Sequence

import numpy as np
import pytest
from typing_extensions import assert_type

from nitypes.bintime import TimeDelta
from nitypes.bintime import TimeDeltaArray


###############################################################################
# Constructor
###############################################################################
def test___no_args___construct___returns_empty_array() -> None:
    value = TimeDeltaArray()

    assert_type(value, TimeDeltaArray)
    assert isinstance(value, TimeDeltaArray)
    assert len(value._array) == 0


@pytest.mark.parametrize(
    "constructor_arg",
    (
        ([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)]),
        (TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)])),
    ),
)
def test___sequence_arg___construct___returns_matching_array(
    constructor_arg: Sequence[TimeDelta],
) -> None:
    value = TimeDeltaArray(constructor_arg)

    assert_type(value, TimeDeltaArray)
    assert isinstance(value, TimeDeltaArray)
    assert len(value._array) == len(constructor_arg)
    assert (value._array[0]["msb"], value._array[0]["lsb"]) == TimeDelta(-1).to_tuple()
    assert (value._array[1]["msb"], value._array[1]["lsb"]) == TimeDelta(20.26).to_tuple()
    assert (value._array[2]["msb"], value._array[2]["lsb"]) == TimeDelta(500).to_tuple()


@pytest.mark.parametrize(
    "constructor_arg",
    (
        ([TimeDelta(0), TimeDelta(15).to_tuple()]),
        ([True, False]),
        ([1, 2]),
        ([10.0, 20.0]),
        (["abc", "xyz"]),
    ),
)
def test___mixed_arg___construct___raises(constructor_arg: list[Any]) -> None:
    with pytest.raises(TypeError):
        _ = TimeDeltaArray(constructor_arg)


###############################################################################
# len
###############################################################################
@pytest.mark.parametrize(
    "timedelta_list, expected_length",
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


###############################################################################
# __getitem__
###############################################################################
@pytest.mark.parametrize(
    "timedelta_list, indexer, raised_exception",
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
def test___timedelta_array___index___returns_timedelta(
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
        None,
        [1, 2, 3],
    ),
)
def test___timedelta_array___invalid_index___raises(indexer: Any) -> None:
    value = TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)])

    with pytest.raises(TypeError):
        _ = value[indexer]


###############################################################################
# __setitem__
###############################################################################
@pytest.mark.parametrize(
    "timedelta_list, indexer, raised_exception",
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
    "indexer, new_entries, expected_result",
    (
        (  # Replaces one-for-one from list
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
        (  # Replaces one from length-1 list
            slice(4, None),
            [TimeDelta(0)],
            TimeDeltaArray(
                [TimeDelta(-1), TimeDelta(3.14), TimeDelta(20.26), TimeDelta(500), TimeDelta(0)]
            ),
        ),
        (  # Replaces many from length-1 list
            slice(1, 4),
            [TimeDelta(0)],
            TimeDeltaArray([TimeDelta(-1), TimeDelta(0), TimeDelta(0x12345678_90ABCDEF)]),
        ),
        (  # Deletes when assigning empty list
            slice(1, 4),
            [],
            TimeDeltaArray([TimeDelta(-1), TimeDelta(0x12345678_90ABCDEF)]),
        ),
        (  # Replaces one-for-one with step from same-sized list
            slice(None, None, 2),
            [TimeDelta(0), TimeDelta(1), TimeDelta(2)],
            TimeDeltaArray(
                [TimeDelta(0), TimeDelta(3.14), TimeDelta(1), TimeDelta(500), TimeDelta(2)]
            ),
        ),
    ),
)
def test___timedelta_array___set_by_slice___updates_array(
    indexer: slice, new_entries: Sequence[TimeDelta], expected_result: TimeDeltaArray
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
    "indexer, new_entries, raised_exception",
    (
        (  # Slice is too long for incoming values
            slice(None, None, 2),
            [TimeDelta(0)],
            ValueError(),
        ),
        (  # Slice is too short for incoming values
            slice(None, None, 4),
            [TimeDelta(0), TimeDelta(10)],
            ValueError(),
        ),
        (  # Assigning empty requires step == 1
            slice(None, None, 2),
            [],
            ValueError(),
        ),
        (  # Cannot assign from scalar
            slice(None, None, 2),
            TimeDelta(0),
            TypeError(),
        ),
        (
            slice(1, None),
            TimeDelta(0),
            TypeError(),
        ),
    ),
)
def test___timedelta_array___set_slice_wrong_value___raises(
    indexer: slice, new_entries: Sequence[TimeDelta] | TimeDelta, raised_exception: BaseException
) -> None:
    value = TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500), TimeDelta(3000.125)])

    with pytest.raises(type(raised_exception)):
        value[indexer] = new_entries  # type: ignore # validating incompatible types


@pytest.mark.parametrize(
    "indexer",
    (
        "0",
        1.0,
        None,
        [1, 2, 3],
    ),
)
def test___timedelta_array___set_invalid_index___raises(indexer: Any) -> None:
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
def test___timedelta_array___set_invalid_value___raises(new_entry: Any) -> None:
    value = TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)])

    with pytest.raises(TypeError):
        value[0] = new_entry


@pytest.mark.parametrize(
    "new_entries",
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


###############################################################################
# __delitem__
###############################################################################
@pytest.mark.parametrize(
    "timedelta_list, indexer, raised_exception",
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
    "indexer, expected_result",
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
        None,
        [1, 2, 3],
    ),
)
def test___timedelta_array___delete_invalid_index___raises(indexer: Any) -> None:
    value = TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)])

    with pytest.raises(TypeError):
        del value[indexer]


###############################################################################
# insert
###############################################################################
@pytest.mark.parametrize(
    "initial_value, index",
    (
        # Empty array
        (None, 0),
        (None, 1),
        (None, 3),
        (None, 10),
        (None, -1),
        (None, -2),
        (None, -10),
        # Existing entries
        ([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)], 0),
        ([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)], 1),
        ([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)], 3),
        ([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)], 10),
        ([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)], -1),
        ([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)], -3),
        ([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)], -10),
    ),
)
def test___timedelta_array___insert_value___inserts(
    initial_value: list[TimeDelta], index: int
) -> None:
    value = TimeDeltaArray(initial_value)
    inserted_value = TimeDelta(0)

    value.insert(index, inserted_value)

    expected_value = initial_value.copy() if initial_value else []
    expected_value.insert(index, inserted_value)
    expected = TimeDeltaArray(expected_value)
    assert np.array_equal(value._array, expected._array)


@pytest.mark.parametrize(
    "index",
    (
        "0",
        1.0,
        None,
        [1, 2, 3],
    ),
)
def test___timedelta_array___insert_invalid_index___raises(index: int) -> None:
    value = TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)])

    with pytest.raises(TypeError):
        value.insert(index, TimeDelta(0))


@pytest.mark.parametrize(
    "value",
    (
        "0",
        1.0,
        True,
        None,
        [1, 2, 3],
    ),
)
def test___timedelta_array___insert_invalid_value___raises(value: Any) -> None:
    value = TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)])

    with pytest.raises(TypeError):
        value.insert(0, value)


###############################################################################
# MutableSequence
###############################################################################
@pytest.mark.parametrize(
    "array, item, expected_count",
    (
        (TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)]), TimeDelta(12.34), 0),
        (TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)]), TimeDelta(-1), 1),
        (TimeDeltaArray([TimeDelta(20.26), TimeDelta(20.26), TimeDelta(500)]), TimeDelta(20.26), 2),
    ),
)
def test___timedelta_array___count___returns_matching_count(
    array: TimeDeltaArray, item: TimeDelta, expected_count: int
) -> None:
    item_count = array.count(item)

    assert item_count == expected_count


@pytest.mark.parametrize(
    "array, item, expected_index",
    (
        (TimeDeltaArray([TimeDelta(20.26), TimeDelta(20.26), TimeDelta(500)]), TimeDelta(20.26), 0),
        (TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)]), TimeDelta(20.26), 1),
    ),
)
def test___timedelta_array___index___returns_matching_index(
    array: TimeDeltaArray, item: TimeDelta, expected_index: int
) -> None:
    item_index = array.index(item)

    assert item_index == expected_index


def test___timedelta_array_no_item___index___raises() -> None:
    array = TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)])

    with pytest.raises(ValueError):
        _ = array.index(TimeDelta(12.34))


def test___timedelta_array___append___adds_to_end() -> None:
    array = TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)])
    new_entry = TimeDelta(12.34)

    array.append(new_entry)

    expected = TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500), TimeDelta(12.34)])
    assert np.array_equal(array._array, expected._array)


@pytest.mark.parametrize(
    "new_entry",
    (
        (),
        (True),
        (13),
        (12.34),
        ("abc"),
        ([]),
        ([TimeDelta(-100)]),
    ),
)
def test___timedelta_array___append_invalid_value___raises(new_entry: Any) -> None:
    array = TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)])

    with pytest.raises(TypeError):
        array.append(new_entry)


@pytest.mark.parametrize(
    "new_entries",
    (
        (),
        ([]),
        ([TimeDelta(12.34)]),
        ([TimeDelta(12.34), TimeDelta(55.77)]),
        (TimeDeltaArray()),
        (TimeDeltaArray([TimeDelta(12.34)])),
        (TimeDeltaArray([TimeDelta(12.34), TimeDelta(55.77)])),
    ),
)
def test___timedelta_array___extend___adds_to_end(new_entries: Sequence[TimeDelta]) -> None:
    original_items = [TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)]
    array = TimeDeltaArray(original_items)

    array.extend(new_entries)

    assert len(array) == len(original_items) + len(new_entries)
    separate_items = [*original_items, *new_entries]
    expected_array = TimeDeltaArray(separate_items)
    assert np.array_equal(array._array, expected_array._array)


@pytest.mark.parametrize(
    "new_entries",
    (
        (),
        ([]),
        ([TimeDelta(12.34)]),
        ([TimeDelta(12.34), TimeDelta(55.77)]),
        (TimeDeltaArray()),
        (TimeDeltaArray([TimeDelta(12.34)])),
        (TimeDeltaArray([TimeDelta(12.34), TimeDelta(55.77)])),
    ),
)
def test___timedelta_array___plus_equals___adds_to_end(new_entries: Sequence[TimeDelta]) -> None:
    original_items = [TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)]
    array = TimeDeltaArray(original_items)

    array += new_entries

    assert len(array) == len(original_items) + len(new_entries)
    separate_items = [*original_items, *new_entries]
    expected_array = TimeDeltaArray(separate_items)
    assert np.array_equal(array._array, expected_array._array)


@pytest.mark.parametrize(
    "new_entries",
    (
        (None),
        (True),
        (13),
        (12.34),
        ("abc"),
        ([None]),
        ([True]),
        ([13]),
        ([12.34]),
        (["abc"]),
    ),
)
def test___timedelta_array___extend_invalid_values___raises(new_entries: Any) -> None:
    array = TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)])

    with pytest.raises(TypeError):
        array.extend(new_entries)


@pytest.mark.parametrize(
    "new_entries",
    (
        (None),
        (True),
        (13),
        (12.34),
        ("abc"),
        ([None]),
        ([True]),
        ([13]),
        ([12.34]),
        (["abc"]),
    ),
)
def test___timedelta_array___plus_equals_invalid_values___raises(new_entries: Any) -> None:
    array = TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)])

    with pytest.raises(TypeError):
        array += new_entries


def test___timedelta_array___remove___removes_first_match() -> None:
    array = TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(20.26)])

    array.remove(TimeDelta(20.26))

    assert len(array) == 2
    expected_array = TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26)])
    assert np.array_equal(array._array, expected_array._array)


@pytest.mark.parametrize(
    "item_to_remove",
    (
        (),
        (None),
        (True),
        (13),
        (12.34),
        ("abc"),
        (TimeDelta(0)),
    ),
)
def test___timedelta_array_no_item___remove___raises(item_to_remove: Any) -> None:
    array = TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)])

    with pytest.raises(ValueError):
        array.remove(item_to_remove)


def test___timedelta_array___pop___removes_from_location() -> None:
    original_values = [TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)]
    array = TimeDeltaArray(original_values)

    popped = array.pop()
    assert popped == original_values[-1]
    assert len(array) == 2
    assert np.array_equal(array._array, TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26)])._array)

    popped = array.pop(0)
    assert popped == TimeDelta(-1)
    assert len(array) == 1
    assert np.array_equal(array._array, TimeDeltaArray([TimeDelta(20.26)])._array)


def test___empty_timedelta_array___pop___raises() -> None:
    array = TimeDeltaArray()

    with pytest.raises(IndexError):
        array.pop()


def test___timedelta_array___pop_out_of_bounds___raises() -> None:
    array = TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)])

    with pytest.raises(IndexError):
        array.pop(10)


def test___timedelta_array___reverse___reverses() -> None:
    original_values = [TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)]
    array = TimeDeltaArray(original_values)

    array.reverse()

    expected_order = original_values.copy()
    expected_order.reverse()
    expected_array = TimeDeltaArray(expected_order)
    assert np.array_equal(array._array, expected_array._array)


def test___timedelta_array___clear___empties_array() -> None:
    array = TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)])

    array.clear()

    assert len(array) == 0


def test___timedelta_array___iterate___visits_entries() -> None:
    original_values = [TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)]
    array = TimeDeltaArray(original_values)

    iterated = list(iter(array))

    assert original_values == iterated


def test___timedelta_array___contains___returns_presence() -> None:
    array = TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)])

    assert TimeDelta(20.26) in array
    assert TimeDelta(12.34) not in array


###############################################################################
# Builtins
###############################################################################
@pytest.mark.parametrize(
    "left, right",
    [
        (
            TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)]),
            TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)]),
        ),
    ],
)
def test___same_value___equality___equal(left: TimeDeltaArray, right: TimeDeltaArray) -> None:
    assert left == right
    assert right == left


@pytest.mark.parametrize(
    "left, right",
    [
        (
            TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)]),
            TimeDeltaArray([TimeDelta(-10), TimeDelta(200.26), TimeDelta(1500)]),
        ),
    ],
)
def test___different_value___equality___not_equal(
    left: TimeDeltaArray, right: TimeDeltaArray
) -> None:
    assert left != right


def test___timedelta_array___min___returns_minimum() -> None:
    array = TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)])

    assert min(array) == TimeDelta(-1)


def test___timedelta_array___max___returns_maximum() -> None:
    array = TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)])

    assert max(array) == TimeDelta(500)


def test___timedelta_array___copy___returns_copy() -> None:
    array = TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)])

    copied = copy.copy(array)

    assert array == copied
    assert array is not copied
    assert array._array is not copied._array
    for original_entry, copied_entry in zip(array, copied):
        assert original_entry is not copied_entry


def test___timedelta_array___deepcopy___returns_deepcopy() -> None:
    array = TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)])

    copied = copy.deepcopy(array)

    assert array == copied
    assert array is not copied
    assert array._array is not copied._array
    for original_entry, copied_entry in zip(array, copied):
        assert original_entry is not copied_entry


@pytest.mark.parametrize(
    "value, expected_str",
    (
        (TimeDeltaArray(), "[]"),
        (TimeDeltaArray([TimeDelta(-1)]), "[-1 day, 23:59:59]"),
        (
            TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26)]),
            "[-1 day, 23:59:59; 0:00:20.260000000000001563]",
        ),
        (
            TimeDeltaArray([TimeDelta(20.26), TimeDelta(-1), TimeDelta(500)]),
            "[0:00:20.260000000000001563; -1 day, 23:59:59; 0:08:20]",
        ),
    ),
)
def test___timedelta_array___str___looks_ok(value: TimeDeltaArray, expected_str: str) -> None:
    assert str(value) == expected_str


@pytest.mark.parametrize(
    "value, expected_repr",
    (
        (TimeDeltaArray(), "nitypes.bintime.TimeDeltaArray([])"),
        (
            TimeDeltaArray([TimeDelta(-1)]),
            "nitypes.bintime.TimeDeltaArray([nitypes.bintime.TimeDelta(Decimal('-1'))])",
        ),
        (
            TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26)]),
            "nitypes.bintime.TimeDeltaArray([nitypes.bintime.TimeDelta(Decimal('-1')), nitypes.bintime.TimeDelta(Decimal('20.2600000000000015631940'))])",
        ),
        (
            TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26), TimeDelta(500)]),
            "nitypes.bintime.TimeDeltaArray([nitypes.bintime.TimeDelta(Decimal('-1')), nitypes.bintime.TimeDelta(Decimal('20.2600000000000015631940')), nitypes.bintime.TimeDelta(Decimal('500'))])",
        ),
    ),
)
def test___timedelta_array___repr___looks_ok(value: TimeDeltaArray, expected_repr: str) -> None:
    assert repr(value) == expected_repr


def test___timedelta_array___pickle___references_public_modules() -> None:
    value = TimeDeltaArray([TimeDelta(20.26), TimeDelta(-1), TimeDelta(500)])

    pickled = pickle.dumps(value)

    assert b"nitypes.bintime" in pickled
    assert b"nitypes.bintime._timedelta_array" not in pickled


@pytest.mark.parametrize(
    "value",
    (
        (TimeDeltaArray()),
        (TimeDeltaArray([TimeDelta(-1)])),
        (TimeDeltaArray([TimeDelta(-1), TimeDelta(20.26)])),
        (TimeDeltaArray([TimeDelta(20.26), TimeDelta(-1), TimeDelta(500)])),
    ),
)
def test___timedelta_array___pickle_unpickle___makes_copy(value: TimeDeltaArray) -> None:
    new_value: TimeDeltaArray = pickle.loads(pickle.dumps(value))
    assert new_value == value
    assert new_value is not value
    assert new_value._array is not value._array
