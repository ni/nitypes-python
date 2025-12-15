from __future__ import annotations

import copy
import datetime as dt
import pickle
from typing import Any, Sequence

import numpy as np
import pytest
from typing_extensions import assert_type

from nitypes.bintime import DateTime, DateTimeArray


###############################################################################
# Constructor
###############################################################################
def test___no_args___construct___returns_empty_array() -> None:
    value = DateTimeArray()

    assert_type(value, DateTimeArray)
    assert isinstance(value, DateTimeArray)
    assert len(value._array) == 0


@pytest.mark.parametrize(
    "constructor_arg",
    (
        (
            [
                DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
                DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
            ]
        ),
        (
            DateTimeArray(
                [
                    DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
                ]
            )
        ),
    ),
)
def test___sequence_arg___construct___returns_matching_array(
    constructor_arg: Sequence[DateTime],
) -> None:
    value = DateTimeArray(constructor_arg)

    assert_type(value, DateTimeArray)
    assert isinstance(value, DateTimeArray)
    assert len(value._array) == len(constructor_arg)
    assert (value._array[0]["msb"], value._array[0]["lsb"]) == constructor_arg[0].to_tuple()
    assert (value._array[1]["msb"], value._array[1]["lsb"]) == constructor_arg[1].to_tuple()


@pytest.mark.parametrize(
    "constructor_arg",
    (
        (
            [
                DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
                DateTime(2025, 1, 2, tzinfo=dt.timezone.utc).to_tuple(),
            ]
        ),
        ([True, False]),
        ([1, 2]),
        ([10.0, 20.0]),
        (["abc", "xyz"]),
    ),
)
def test___mixed_arg___construct___raises(constructor_arg: list[Any]) -> None:
    with pytest.raises(TypeError):
        _ = DateTimeArray(constructor_arg)


###############################################################################
# len
###############################################################################
@pytest.mark.parametrize(
    "datetime_list, expected_length",
    (
        (None, 0),
        ([DateTime(2025, 1, 1, tzinfo=dt.timezone.utc)], 1),
        (
            [
                DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
                DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
            ],
            2,
        ),
    ),
)
def test___datetime_array___get_len___returns_length(
    datetime_list: list[DateTime] | None, expected_length: int
) -> None:
    value = DateTimeArray(datetime_list)

    length = len(value)

    assert length == expected_length


###############################################################################
# __getitem__
###############################################################################
@pytest.mark.parametrize(
    "datetime_list, indexer, raised_exception",
    (
        # First index
        (None, 0, IndexError()),
        ([DateTime(2025, 1, 1, tzinfo=dt.timezone.utc)], 0, None),
        (
            [
                DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
                DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
            ],
            0,
            None,
        ),
        # Last index
        (None, -1, IndexError()),
        ([DateTime(2025, 1, 1, tzinfo=dt.timezone.utc)], -1, None),
        (
            [
                DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
                DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
            ],
            -1,
            None,
        ),
        # Out of bounds index
        (None, 10, IndexError()),
        ([DateTime(2025, 1, 1, tzinfo=dt.timezone.utc)], 10, IndexError()),
        (
            [
                DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
                DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
            ],
            10,
            IndexError(),
        ),
    ),
)
def test___datetime_array___index___returns_datetime(
    datetime_list: list[DateTime], indexer: int, raised_exception: BaseException | None
) -> None:
    value = DateTimeArray(datetime_list)

    if raised_exception:
        with pytest.raises(type(raised_exception)):
            _ = value[indexer]
    else:
        entry = value[indexer]
        assert entry == datetime_list[indexer]


def test___datetime_array___slice___returns_slice() -> None:
    value = DateTimeArray(
        [
            DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 3, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 4, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 5, tzinfo=dt.timezone.utc),
        ]
    )

    selected = value[::2]

    expected = DateTimeArray(
        [
            DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 3, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 5, tzinfo=dt.timezone.utc),
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
def test___datetime_array___invalid_index___raises(indexer: Any) -> None:
    value = DateTimeArray(
        [DateTime(2025, 1, 1, tzinfo=dt.timezone.utc), DateTime(2025, 1, 2, tzinfo=dt.timezone.utc)]
    )

    with pytest.raises(TypeError):
        _ = value[indexer]


###############################################################################
# __setitem__
###############################################################################
@pytest.mark.parametrize(
    "datetime_list, indexer, raised_exception",
    (
        # First index
        (None, 0, IndexError()),
        ([DateTime(2025, 1, 1, tzinfo=dt.timezone.utc)], 0, None),
        (
            [
                DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
                DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
            ],
            0,
            None,
        ),
        # Last index
        (None, -1, IndexError()),
        ([DateTime(2025, 1, 1, tzinfo=dt.timezone.utc)], -1, None),
        (
            [
                DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
                DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
            ],
            -1,
            None,
        ),
        # Out of bounds index
        (None, 10, IndexError()),
        ([DateTime(2025, 1, 1, tzinfo=dt.timezone.utc)], 10, IndexError()),
        (
            [
                DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
                DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
            ],
            10,
            IndexError(),
        ),
    ),
)
def test___datetime_array___set_by_index___updates_array(
    datetime_list: list[DateTime] | None, indexer: int, raised_exception: BaseException | None
) -> None:
    value = DateTimeArray(datetime_list)
    new_entry = DateTime(1990, 1, 1, tzinfo=dt.timezone.utc)

    if raised_exception:
        with pytest.raises(type(raised_exception)):
            value[indexer] = new_entry
    else:
        value[indexer] = new_entry
        assert value[indexer] == new_entry


@pytest.mark.parametrize(
    "indexer, new_entries, expected_result",
    (
        # len(selected entries) == len(incoming entries)
        (  # Replaces one-for-one from list
            slice(1, 4),
            [
                DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
                DateTime(1991, 1, 1, tzinfo=dt.timezone.utc),
                DateTime(1992, 1, 1, tzinfo=dt.timezone.utc),
            ],
            DateTimeArray(
                [
                    DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(1991, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(1992, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(2025, 1, 5, tzinfo=dt.timezone.utc),
                ]
            ),
        ),
        (  # Replaces one from length-1 list
            slice(3, 4),
            [DateTime(1990, 1, 1, tzinfo=dt.timezone.utc)],
            DateTimeArray(
                [
                    DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
                    DateTime(2025, 1, 3, tzinfo=dt.timezone.utc),
                    DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(2025, 1, 5, tzinfo=dt.timezone.utc),
                ]
            ),
        ),
        (  # With strided selection, replaces one-for-one from same-sized list
            slice(None, None, 2),
            [
                DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
                DateTime(1991, 1, 1, tzinfo=dt.timezone.utc),
                DateTime(1992, 1, 1, tzinfo=dt.timezone.utc),
            ],
            DateTimeArray(
                [
                    DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
                    DateTime(1991, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(2025, 1, 4, tzinfo=dt.timezone.utc),
                    DateTime(1992, 1, 1, tzinfo=dt.timezone.utc),
                ]
            ),
        ),
        # len(selected entries) > len(incoming entries)
        (  # Shrinks array, replacing many from length-2 list
            slice(1, 4),
            [
                DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
                DateTime(1991, 1, 1, tzinfo=dt.timezone.utc),
            ],
            DateTimeArray(
                [
                    DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(1991, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(2025, 1, 5, tzinfo=dt.timezone.utc),
                ]
            ),
        ),
        (  # Shrinks array, replacing many from length-1 list
            slice(1, 4),
            [DateTime(1990, 1, 1, tzinfo=dt.timezone.utc)],
            DateTimeArray(
                [
                    DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(2025, 1, 5, tzinfo=dt.timezone.utc),
                ]
            ),
        ),
        (  # Shrinks array, deleting when assigning empty list
            slice(1, 4),
            [],
            DateTimeArray(
                [
                    DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(2025, 1, 5, tzinfo=dt.timezone.utc),
                ]
            ),
        ),
        # len(selected entries) < len(incoming entries)
        (  # Grows array, replacing then inserting when slice is too short for incoming values
            slice(1, 2),
            [
                DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
                DateTime(1991, 1, 1, tzinfo=dt.timezone.utc),
                DateTime(1992, 1, 1, tzinfo=dt.timezone.utc),
            ],
            DateTimeArray(
                [
                    DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(1991, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(1992, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(2025, 1, 3, tzinfo=dt.timezone.utc),
                    DateTime(2025, 1, 4, tzinfo=dt.timezone.utc),
                    DateTime(2025, 1, 5, tzinfo=dt.timezone.utc),
                ]
            ),
        ),
        (  # Grows array, inserting when slice is empty
            slice(1, 1),
            [
                DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
                DateTime(1991, 1, 1, tzinfo=dt.timezone.utc),
                DateTime(1992, 1, 1, tzinfo=dt.timezone.utc),
            ],
            DateTimeArray(
                [
                    DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(1991, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(1992, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
                    DateTime(2025, 1, 3, tzinfo=dt.timezone.utc),
                    DateTime(2025, 1, 4, tzinfo=dt.timezone.utc),
                    DateTime(2025, 1, 5, tzinfo=dt.timezone.utc),
                ]
            ),
        ),
    ),
)
def test___datetime_array___set_by_slice___updates_array(
    indexer: slice, new_entries: Sequence[DateTime], expected_result: DateTimeArray
) -> None:
    value = DateTimeArray(
        [
            DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 3, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 4, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 5, tzinfo=dt.timezone.utc),
        ]
    )

    value[indexer] = new_entries

    assert np.array_equal(value._array, expected_result._array)


@pytest.mark.parametrize(
    "indexer, new_entries, raised_exception",
    (
        (  # Strided slice is too long for incoming values
            slice(None, None, 2),
            [DateTime(1990, 1, 1, tzinfo=dt.timezone.utc)],
            ValueError(),
        ),
        (  # Strided slice is too short for incoming values
            slice(None, None, 4),
            [
                DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
                DateTime(1991, 1, 1, tzinfo=dt.timezone.utc),
            ],
            ValueError(),
        ),
        (  # Assigning empty requires step == 1
            slice(None, None, 2),
            [],
            ValueError(),
        ),
        (  # Cannot assign from scalar
            slice(None, None, 2),
            DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
            TypeError(),
        ),
        (
            slice(1, None),
            DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
            TypeError(),
        ),
    ),
)
def test___datetime_array___set_slice_wrong_value___raises(
    indexer: slice, new_entries: Sequence[DateTime] | DateTime, raised_exception: BaseException
) -> None:
    value = DateTimeArray(
        [
            DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 3, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 4, tzinfo=dt.timezone.utc),
        ]
    )

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
def test___datetime_array___set_invalid_index___raises(indexer: Any) -> None:
    value = DateTimeArray(
        [DateTime(2025, 1, 1, tzinfo=dt.timezone.utc), DateTime(2025, 1, 2, tzinfo=dt.timezone.utc)]
    )
    new_entry = DateTime(1999, 1, 1, tzinfo=dt.timezone.utc)

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
def test___datetime_array___set_invalid_value___raises(new_entry: Any) -> None:
    value = DateTimeArray(
        [DateTime(2025, 1, 1, tzinfo=dt.timezone.utc), DateTime(2025, 1, 2, tzinfo=dt.timezone.utc)]
    )

    with pytest.raises(TypeError):
        value[0] = new_entry


@pytest.mark.parametrize(
    "new_entries",
    (
        ["ab", "cd"],
        [1.0, 2.0],
        [True, False],
        [None, None],
        [
            DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 2, tzinfo=dt.timezone.utc).to_tuple(),
        ],
    ),
)
def test___datetime_array___set_mixed_slice___raises(new_entries: list[Any]) -> None:
    value = DateTimeArray(
        [
            DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 3, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 4, tzinfo=dt.timezone.utc),
        ]
    )

    with pytest.raises(TypeError):
        value[::2] = new_entries


###############################################################################
# __delitem__
###############################################################################
@pytest.mark.parametrize(
    "datetime_list, indexer, raised_exception",
    (
        # First index
        (None, 0, IndexError()),
        ([DateTime(2025, 1, 1, tzinfo=dt.timezone.utc)], 0, None),
        (
            [
                DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
                DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
            ],
            0,
            None,
        ),
        # Last index
        (None, -1, IndexError()),
        ([DateTime(2025, 1, 1, tzinfo=dt.timezone.utc)], -1, None),
        (
            [
                DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
                DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
            ],
            -1,
            None,
        ),
        # Out of bounds index
        (None, 10, IndexError()),
        ([DateTime(2025, 1, 1, tzinfo=dt.timezone.utc)], 10, IndexError()),
        (
            [
                DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
                DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
            ],
            10,
            IndexError(),
        ),
    ),
)
def test___datetime_array___delete_by_index___removes_item(
    datetime_list: list[DateTime] | None, indexer: int, raised_exception: BaseException | None
) -> None:
    value = DateTimeArray(datetime_list)

    if raised_exception:
        with pytest.raises(type(raised_exception)):
            del value[indexer]
    else:
        modified = datetime_list.copy() if datetime_list else []
        del modified[indexer]
        del value[indexer]
        expected = DateTimeArray(modified)
        assert np.array_equal(value._array, expected._array)


@pytest.mark.parametrize(
    "indexer, expected_result",
    (
        (
            slice(1, 4),
            DateTimeArray(
                [
                    DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(2025, 1, 5, tzinfo=dt.timezone.utc),
                ]
            ),
        ),
        (
            slice(None, None, 2),
            DateTimeArray(
                [
                    DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
                    DateTime(2025, 1, 4, tzinfo=dt.timezone.utc),
                ]
            ),
        ),
    ),
)
def test___datetime_array___delete_by_slice___removes_items(
    indexer: slice, expected_result: DateTimeArray
) -> None:
    value = DateTimeArray(
        [
            DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 3, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 4, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 5, tzinfo=dt.timezone.utc),
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
def test___datetime_array___delete_invalid_index___raises(indexer: Any) -> None:
    value = DateTimeArray(
        [DateTime(2025, 1, 1, tzinfo=dt.timezone.utc), DateTime(2025, 1, 2, tzinfo=dt.timezone.utc)]
    )

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
        (
            [
                DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
                DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
                DateTime(2025, 1, 3, tzinfo=dt.timezone.utc),
            ],
            0,
        ),
        (
            [
                DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
                DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
                DateTime(2025, 1, 3, tzinfo=dt.timezone.utc),
            ],
            1,
        ),
        (
            [
                DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
                DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
                DateTime(2025, 1, 3, tzinfo=dt.timezone.utc),
            ],
            3,
        ),
        (
            [
                DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
                DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
                DateTime(2025, 1, 3, tzinfo=dt.timezone.utc),
            ],
            10,
        ),
        (
            [
                DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
                DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
                DateTime(2025, 1, 3, tzinfo=dt.timezone.utc),
            ],
            -1,
        ),
        (
            [
                DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
                DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
                DateTime(2025, 1, 3, tzinfo=dt.timezone.utc),
            ],
            -3,
        ),
        (
            [
                DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
                DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
                DateTime(2025, 1, 3, tzinfo=dt.timezone.utc),
            ],
            -10,
        ),
    ),
)
def test___datetime_array___insert_value___inserts(
    initial_value: list[DateTime] | None, index: int
) -> None:
    value = DateTimeArray(initial_value)
    inserted_value = DateTime(1990, 1, 1, tzinfo=dt.timezone.utc)

    value.insert(index, inserted_value)

    expected_value = initial_value.copy() if initial_value else []
    expected_value.insert(index, inserted_value)
    expected = DateTimeArray(expected_value)
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
def test___datetime_array___insert_invalid_index___raises(index: int) -> None:
    value = DateTimeArray(
        [DateTime(2025, 1, 1, tzinfo=dt.timezone.utc), DateTime(2025, 1, 2, tzinfo=dt.timezone.utc)]
    )

    with pytest.raises(TypeError):
        value.insert(index, DateTime(1990, 1, 1, tzinfo=dt.timezone.utc))


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
def test___datetime_array___insert_invalid_value___raises(value: Any) -> None:
    value = DateTimeArray(
        [DateTime(2025, 1, 1, tzinfo=dt.timezone.utc), DateTime(2025, 1, 2, tzinfo=dt.timezone.utc)]
    )

    with pytest.raises(TypeError):
        value.insert(0, value)


###############################################################################
# MutableSequence
###############################################################################
@pytest.mark.parametrize(
    "array, item, expected_count",
    (
        (
            DateTimeArray(
                [
                    DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
                ]
            ),
            DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
            0,
        ),
        (
            DateTimeArray(
                [
                    DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
                ]
            ),
            DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
            1,
        ),
        (
            DateTimeArray(
                [
                    DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
                    DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
                ]
            ),
            DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
            2,
        ),
    ),
)
def test___datetime_array___count___returns_matching_count(
    array: DateTimeArray, item: DateTime, expected_count: int
) -> None:
    item_count = array.count(item)

    assert item_count == expected_count


@pytest.mark.parametrize(
    "array, item, expected_index",
    (
        (
            DateTimeArray(
                [
                    DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
                    DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
                ]
            ),
            DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
            0,
        ),
        (
            DateTimeArray(
                [
                    DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
                ]
            ),
            DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
            1,
        ),
    ),
)
def test___datetime_array___index___returns_matching_index(
    array: DateTimeArray, item: DateTime, expected_index: int
) -> None:
    item_index = array.index(item)

    assert item_index == expected_index


def test___datetime_array_no_item___index___raises() -> None:
    array = DateTimeArray(
        [DateTime(1990, 1, 1, tzinfo=dt.timezone.utc), DateTime(2025, 1, 2, tzinfo=dt.timezone.utc)]
    )

    with pytest.raises(ValueError):
        _ = array.index(DateTime(2000, 1, 1, tzinfo=dt.timezone.utc))


def test___datetime_array___append___adds_to_end() -> None:
    array = DateTimeArray(
        [DateTime(1990, 1, 1, tzinfo=dt.timezone.utc), DateTime(2025, 1, 2, tzinfo=dt.timezone.utc)]
    )
    new_entry = DateTime(2000, 1, 1, tzinfo=dt.timezone.utc)

    array.append(new_entry)

    expected = DateTimeArray(
        [
            DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
            DateTime(2000, 1, 1, tzinfo=dt.timezone.utc),
        ]
    )
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
        ([DateTime(1990, 1, 1, tzinfo=dt.timezone.utc)]),
    ),
)
def test___datetime_array___append_invalid_value___raises(new_entry: Any) -> None:
    array = DateTimeArray(
        [DateTime(1990, 1, 1, tzinfo=dt.timezone.utc), DateTime(2025, 1, 2, tzinfo=dt.timezone.utc)]
    )

    with pytest.raises(TypeError):
        array.append(new_entry)


@pytest.mark.parametrize(
    "new_entries",
    (
        (),
        ([]),
        ([DateTime(2000, 1, 1, tzinfo=dt.timezone.utc)]),
        (
            [
                DateTime(2000, 1, 1, tzinfo=dt.timezone.utc),
                DateTime(2001, 1, 1, tzinfo=dt.timezone.utc),
            ]
        ),
        (DateTimeArray()),
        (DateTimeArray([DateTime(2000, 1, 1, tzinfo=dt.timezone.utc)])),
        (
            DateTimeArray(
                [
                    DateTime(2000, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(2001, 1, 1, tzinfo=dt.timezone.utc),
                ]
            )
        ),
    ),
)
def test___datetime_array___extend___adds_to_end(new_entries: Sequence[DateTime]) -> None:
    original_items = [
        DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
        DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
    ]
    array = DateTimeArray(original_items)

    array.extend(new_entries)

    assert len(array) == len(original_items) + len(new_entries)
    separate_items = [*original_items, *new_entries]
    expected_array = DateTimeArray(separate_items)
    assert np.array_equal(array._array, expected_array._array)


@pytest.mark.parametrize(
    "new_entries",
    (
        (),
        ([]),
        ([DateTime(2000, 1, 1, tzinfo=dt.timezone.utc)]),
        (
            [
                DateTime(2000, 1, 1, tzinfo=dt.timezone.utc),
                DateTime(2001, 1, 1, tzinfo=dt.timezone.utc),
            ]
        ),
        (DateTimeArray()),
        (DateTimeArray([DateTime(2000, 1, 1, tzinfo=dt.timezone.utc)])),
        (
            DateTimeArray(
                [
                    DateTime(2000, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(2001, 1, 1, tzinfo=dt.timezone.utc),
                ]
            )
        ),
    ),
)
def test___datetime_array___plus_equals___adds_to_end(new_entries: Sequence[DateTime]) -> None:
    original_items = [
        DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
        DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
    ]
    array = DateTimeArray(original_items)

    array += new_entries

    assert len(array) == len(original_items) + len(new_entries)
    separate_items = [*original_items, *new_entries]
    expected_array = DateTimeArray(separate_items)
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
def test___datetime_array___extend_invalid_values___raises(new_entries: Any) -> None:
    array = DateTimeArray(
        [DateTime(1990, 1, 1, tzinfo=dt.timezone.utc), DateTime(2025, 1, 2, tzinfo=dt.timezone.utc)]
    )

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
def test___datetime_array___plus_equals_invalid_values___raises(new_entries: Any) -> None:
    array = DateTimeArray(
        [DateTime(1990, 1, 1, tzinfo=dt.timezone.utc), DateTime(2025, 1, 2, tzinfo=dt.timezone.utc)]
    )

    with pytest.raises(TypeError):
        array += new_entries


def test___datetime_array___remove___removes_first_match() -> None:
    array = DateTimeArray(
        [
            DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
        ]
    )

    array.remove(DateTime(2025, 1, 2, tzinfo=dt.timezone.utc))

    assert len(array) == 2
    expected_array = DateTimeArray(
        [DateTime(1990, 1, 1, tzinfo=dt.timezone.utc), DateTime(2025, 1, 2, tzinfo=dt.timezone.utc)]
    )
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
        (DateTime(2025, 1, 1, tzinfo=dt.timezone.utc)),
    ),
)
def test___datetime_array_no_item___remove___raises(item_to_remove: Any) -> None:
    array = DateTimeArray(
        [
            DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 3, tzinfo=dt.timezone.utc),
        ]
    )

    with pytest.raises(ValueError):
        array.remove(item_to_remove)


def test___datetime_array___pop___removes_from_location() -> None:
    original_values = [
        DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
        DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
        DateTime(2025, 1, 3, tzinfo=dt.timezone.utc),
    ]
    array = DateTimeArray(original_values)

    popped = array.pop()
    assert popped == original_values[-1]
    assert len(array) == 2
    assert np.array_equal(
        array._array,
        DateTimeArray(
            [
                DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
                DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
            ]
        )._array,
    )

    popped = array.pop(0)
    assert popped == DateTime(1990, 1, 1, tzinfo=dt.timezone.utc)
    assert len(array) == 1
    assert np.array_equal(
        array._array, DateTimeArray([DateTime(2025, 1, 2, tzinfo=dt.timezone.utc)])._array
    )


def test___empty_datetime_array___pop___raises() -> None:
    array = DateTimeArray()

    with pytest.raises(IndexError):
        array.pop()


def test___datetime_array___pop_out_of_bounds___raises() -> None:
    array = DateTimeArray(
        [DateTime(1990, 1, 1, tzinfo=dt.timezone.utc), DateTime(2025, 1, 2, tzinfo=dt.timezone.utc)]
    )

    with pytest.raises(IndexError):
        array.pop(10)


def test___datetime_array___reverse___reverses() -> None:
    original_values = [
        DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
        DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
        DateTime(2025, 1, 3, tzinfo=dt.timezone.utc),
    ]
    array = DateTimeArray(original_values)

    array.reverse()

    expected_order = original_values.copy()
    expected_order.reverse()
    expected_array = DateTimeArray(expected_order)
    assert np.array_equal(array._array, expected_array._array)


def test___datetime_array___clear___empties_array() -> None:
    array = DateTimeArray(
        [DateTime(1990, 1, 1, tzinfo=dt.timezone.utc), DateTime(2025, 1, 2, tzinfo=dt.timezone.utc)]
    )

    array.clear()

    assert len(array) == 0


def test___datetime_array___iterate___visits_entries() -> None:
    original_values = [
        DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
        DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
    ]
    array = DateTimeArray(original_values)

    iterated = list(iter(array))

    assert original_values == iterated


def test___datetime_array___contains___returns_presence() -> None:
    array = DateTimeArray(
        [
            DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 2, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 3, tzinfo=dt.timezone.utc),
        ]
    )

    assert DateTime(2025, 1, 2, tzinfo=dt.timezone.utc) in array
    assert DateTime(2000, 1, 1, tzinfo=dt.timezone.utc) not in array


###############################################################################
# Builtins
###############################################################################
@pytest.mark.parametrize(
    "left, right",
    [
        (
            DateTimeArray(
                [
                    DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(2000, 1, 1, tzinfo=dt.timezone.utc),
                ]
            ),
            DateTimeArray(
                [
                    DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(2000, 1, 1, tzinfo=dt.timezone.utc),
                ]
            ),
        ),
    ],
)
def test___same_value___equality___equal(left: DateTimeArray, right: DateTimeArray) -> None:
    assert left == right
    assert right == left


@pytest.mark.parametrize(
    "left, right",
    [
        (
            DateTimeArray(
                [
                    DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(2000, 1, 1, tzinfo=dt.timezone.utc),
                ]
            ),
            DateTimeArray(
                [
                    DateTime(1991, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(2001, 1, 1, tzinfo=dt.timezone.utc),
                ]
            ),
        ),
    ],
)
def test___different_value___equality___not_equal(
    left: DateTimeArray, right: DateTimeArray
) -> None:
    assert left != right


def test___datetime_array___min___returns_minimum() -> None:
    array = DateTimeArray(
        [DateTime(1990, 1, 1, tzinfo=dt.timezone.utc), DateTime(2025, 1, 2, tzinfo=dt.timezone.utc)]
    )

    assert min(array) == DateTime(1990, 1, 1, tzinfo=dt.timezone.utc)


def test___datetime_array___max___returns_maximum() -> None:
    array = DateTimeArray(
        [DateTime(1990, 1, 1, tzinfo=dt.timezone.utc), DateTime(2025, 1, 2, tzinfo=dt.timezone.utc)]
    )

    assert max(array) == DateTime(2025, 1, 2, tzinfo=dt.timezone.utc)


def test___datetime_array___copy___returns_copy() -> None:
    array = DateTimeArray(
        [DateTime(1990, 1, 1, tzinfo=dt.timezone.utc), DateTime(2025, 1, 2, tzinfo=dt.timezone.utc)]
    )

    copied = copy.copy(array)

    assert array == copied
    assert array is not copied
    assert array._array is not copied._array
    for original_entry, copied_entry in zip(array, copied):
        assert original_entry is not copied_entry


def test___datetime_array___deepcopy___returns_deepcopy() -> None:
    array = DateTimeArray(
        [DateTime(1990, 1, 1, tzinfo=dt.timezone.utc), DateTime(2025, 1, 2, tzinfo=dt.timezone.utc)]
    )

    copied = copy.deepcopy(array)

    assert array == copied
    assert array is not copied
    assert array._array is not copied._array
    for original_entry, copied_entry in zip(array, copied):
        assert original_entry is not copied_entry


@pytest.mark.parametrize(
    "value, expected_str",
    (
        (DateTimeArray(), "[]"),
        (
            DateTimeArray([DateTime(1990, 1, 1, tzinfo=dt.timezone.utc)]),
            "[1990-01-01 00:00:00+00:00]",
        ),
        (
            DateTimeArray(
                [
                    DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(2000, 1, 1, tzinfo=dt.timezone.utc),
                ]
            ),
            "[1990-01-01 00:00:00+00:00; 2000-01-01 00:00:00+00:00]",
        ),
        (
            DateTimeArray(
                [
                    DateTime(2000, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
                ]
            ),
            "[2000-01-01 00:00:00+00:00; 1990-01-01 00:00:00+00:00; 2025-01-01 00:00:00+00:00]",
        ),
    ),
)
def test___datetime_array___str___looks_ok(value: DateTimeArray, expected_str: str) -> None:
    assert str(value) == expected_str


@pytest.mark.parametrize(
    "value, expected_repr",
    (
        (DateTimeArray(), "nitypes.bintime.DateTimeArray([])"),
        (
            DateTimeArray([DateTime(1990, 1, 1, tzinfo=dt.timezone.utc)]),
            "nitypes.bintime.DateTimeArray([nitypes.bintime.DateTime(1990, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)])",
        ),
        (
            DateTimeArray(
                [
                    DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(2000, 1, 1, tzinfo=dt.timezone.utc),
                ]
            ),
            "nitypes.bintime.DateTimeArray([nitypes.bintime.DateTime(1990, 1, 1, 0, 0, tzinfo=datetime.timezone.utc), nitypes.bintime.DateTime(2000, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)])",
        ),
        (
            DateTimeArray(
                [
                    DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(2000, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
                ]
            ),
            "nitypes.bintime.DateTimeArray([nitypes.bintime.DateTime(1990, 1, 1, 0, 0, tzinfo=datetime.timezone.utc), nitypes.bintime.DateTime(2000, 1, 1, 0, 0, tzinfo=datetime.timezone.utc), nitypes.bintime.DateTime(2025, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)])",
        ),
    ),
)
def test___datetime_array___repr___looks_ok(value: DateTimeArray, expected_repr: str) -> None:
    assert repr(value) == expected_repr


def test___datetime_array___pickle___references_public_modules() -> None:
    value = DateTimeArray(
        [
            DateTime(2000, 1, 1, tzinfo=dt.timezone.utc),
            DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
            DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
        ]
    )

    pickled = pickle.dumps(value)

    assert b"nitypes.bintime" in pickled
    assert b"nitypes.bintime._datetime_array" not in pickled


@pytest.mark.parametrize(
    "pickled_value, expected",
    [
        # nitypes 1.0.0
        (
            b"\x80\x04\x95\xb1\x00\x00\x00\x00\x00\x00\x00\x8c\x0fnitypes.bintime\x94\x8c\rDateTimeArray\x94\x93\x94]\x94(\x8c\x08builtins\x94\x8c\x07getattr\x94\x93\x94h\x00\x8c\x08DateTime\x94\x93\x94\x8c\nfrom_ticks\x94\x86\x94R\x94\x8a\r\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf4\x92\xb4\x00\x85\x94R\x94h\x06h\x08h\t\x86\x94R\x94\x8a\r\x00\x00\x00\x00\x00\x00\x00\x00\x00N\xc4\xa1\x00\x85\x94R\x94h\x06h\x08h\t\x86\x94R\x94\x8a\r\x00\x00\x00\x00\x00\x00\x00\x00\x006\x9a\xe3\x00\x85\x94R\x94e\x85\x94R\x94.",
            DateTimeArray(
                [
                    DateTime(2000, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
                ]
            ),
        ),
    ],
)
def test___pickled_value___unpickle___is_compatible(
    pickled_value: bytes, expected: DateTimeArray
) -> None:
    new_value = pickle.loads(pickled_value)
    assert new_value == expected


@pytest.mark.parametrize(
    "value",
    (
        (DateTimeArray()),
        (DateTimeArray([DateTime(1990, 1, 1, tzinfo=dt.timezone.utc)])),
        (
            DateTimeArray(
                [
                    DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(2000, 1, 1, tzinfo=dt.timezone.utc),
                ]
            )
        ),
        (
            DateTimeArray(
                [
                    DateTime(2000, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(1990, 1, 1, tzinfo=dt.timezone.utc),
                    DateTime(2025, 1, 1, tzinfo=dt.timezone.utc),
                ]
            )
        ),
    ),
)
def test___datetime_array___pickle_unpickle___makes_copy(value: DateTimeArray) -> None:
    new_value: DateTimeArray = pickle.loads(pickle.dumps(value))
    assert new_value == value
    assert new_value is not value
    assert new_value._array is not value._array
