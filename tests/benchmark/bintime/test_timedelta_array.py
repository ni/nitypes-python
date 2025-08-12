from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import nitypes.bintime as bt


LIST_10: list[bt.TimeDelta] = [bt.TimeDelta(float(value)) for value in np.arange(-10, 10, 0.3)]
LIST_100: list[bt.TimeDelta] = [bt.TimeDelta(float(value)) for value in np.arange(-100, 100, 0.3)]
LIST_1000: list[bt.TimeDelta] = [
    bt.TimeDelta(float(value)) for value in np.arange(-1000, 1000, 0.3)
]
LIST_10000: list[bt.TimeDelta] = [
    bt.TimeDelta(float(value)) for value in np.arange(-10000, 10000, 0.3)
]


@pytest.mark.benchmark(group="timedelta_array_construct", min_rounds=100)
@pytest.mark.parametrize("constructor_list", (LIST_10, LIST_100, LIST_1000, LIST_10000))
def test___bt_timedelta_array___construct(
    benchmark: BenchmarkFixture,
    constructor_list: list[bt.TimeDelta],
) -> None:
    benchmark(bt.TimeDeltaArray, constructor_list)


@pytest.mark.benchmark(group="timedelta_array_extend", min_rounds=100)
@pytest.mark.parametrize("extend_list", (LIST_10, LIST_100, LIST_1000, LIST_10000))
def test___bt_timedelta_array___extend(
    benchmark: BenchmarkFixture,
    extend_list: list[bt.TimeDelta],
) -> None:
    empty_array = bt.TimeDeltaArray()
    benchmark(empty_array.extend, extend_list)
