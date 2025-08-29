from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import nitypes.bintime as bt


LIST_1 = [bt.DateTime.from_offset(bt.TimeDelta(0.3))]
LIST_10 = [bt.DateTime.from_offset(bt.TimeDelta(float(offset))) for offset in np.arange(0, 10, 0.3)]
LIST_100 = [
    bt.DateTime.from_offset(bt.TimeDelta(float(offset))) for offset in np.arange(0, 100, 0.3)
]
LIST_1000 = [
    bt.DateTime.from_offset(bt.TimeDelta(float(offset))) for offset in np.arange(0, 1000, 0.3)
]
LIST_10000 = [
    bt.DateTime.from_offset(bt.TimeDelta(float(offset))) for offset in np.arange(0, 10000, 0.3)
]

FAST_CASES = (LIST_1, LIST_10, LIST_100)
BIG_O_CASES = (LIST_1, LIST_10, LIST_100, LIST_1000, LIST_10000)


@pytest.mark.benchmark(group="datetime_array_construct")
@pytest.mark.parametrize("constructor_list", FAST_CASES)
def test___bt_datetime_array___construct(
    benchmark: BenchmarkFixture,
    constructor_list: list[bt.DateTime],
) -> None:
    benchmark(bt.DateTimeArray, constructor_list)


@pytest.mark.benchmark(group="datetime_array_extend")
@pytest.mark.parametrize("extend_list", FAST_CASES)
def test___bt_datetime_array___extend(
    benchmark: BenchmarkFixture,
    extend_list: list[bt.DateTime],
) -> None:
    empty_array = bt.DateTimeArray()
    benchmark(empty_array.extend, extend_list)
