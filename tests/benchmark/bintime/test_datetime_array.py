from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import nitypes.bintime as bt


LIST_10: list[bt.DateTime] = [
    bt.DateTime.from_offset(bt.TimeDelta(float(offset))) for offset in np.arange(0, 10, 0.3)
]
# LIST_100: list[bt.DateTime] = [
#     bt.DateTime.from_offset(bt.TimeDelta(float(offset))) for offset in np.arange(0, 100, 0.3)
# ]
# LIST_1000: list[bt.DateTime] = [
#     bt.DateTime.from_offset(bt.TimeDelta(float(offset))) for offset in np.arange(0, 1000, 0.3)
# ]
# LIST_10000: list[bt.DateTime] = [
#     bt.DateTime.from_offset(bt.TimeDelta(float(offset))) for offset in np.arange(0, 10000, 0.3)
# ]


@pytest.mark.benchmark(group="datetime_array_construct", min_rounds=1, max_time=0.5)
@pytest.mark.parametrize("constructor_list", (LIST_10))
def test___bt_datetime_array___construct(
    benchmark: BenchmarkFixture,
    constructor_list: list[bt.DateTime],
) -> None:
    benchmark(bt.DateTimeArray, constructor_list)


@pytest.mark.benchmark(group="datetime_array_extend", min_rounds=1, max_time=0.5)
@pytest.mark.parametrize("extend_list", (LIST_10))
def test___bt_datetime_array___extend(
    benchmark: BenchmarkFixture,
    extend_list: list[bt.DateTime],
) -> None:
    empty_array = bt.DateTimeArray()
    benchmark(empty_array.extend, extend_list)
