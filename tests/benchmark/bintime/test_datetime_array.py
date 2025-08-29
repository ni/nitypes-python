from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import nitypes.bintime as bt


benchmark_options: dict[str, Any] = {}
match sys.implementation.name:
    case "pypy":
        # See #182 -- PR/CI workflows spend too much time on PyPy benchmarks
        benchmark_options["warmup"] = False
        benchmark_options["min_rounds"] = 1
        benchmark_options["max_time"] = 0.5
    case _:
        pass


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

FAST_CASES = (LIST_1,)
BIG_O_CASES = (LIST_1, LIST_10, LIST_100, LIST_1000, LIST_10000)


@pytest.mark.benchmark(group="datetime_array_construct", **benchmark_options)
@pytest.mark.parametrize("constructor_list", FAST_CASES)
def test___bt_datetime_array___construct(
    benchmark: BenchmarkFixture,
    constructor_list: list[bt.DateTime],
) -> None:
    benchmark(bt.DateTimeArray, constructor_list)


@pytest.mark.benchmark(group="datetime_array_extend", **benchmark_options)
@pytest.mark.parametrize("extend_list", FAST_CASES)
def test___bt_datetime_array___extend(
    benchmark: BenchmarkFixture,
    extend_list: list[bt.DateTime],
) -> None:
    empty_array = bt.DateTimeArray()
    benchmark(empty_array.extend, extend_list)
