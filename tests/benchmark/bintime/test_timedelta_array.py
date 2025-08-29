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


LIST_1 = [bt.TimeDelta(0.3)]
LIST_10 = [bt.TimeDelta(float(value)) for value in np.arange(-10, 10, 0.3)]
LIST_100 = [bt.TimeDelta(float(value)) for value in np.arange(-100, 100, 0.3)]
LIST_1000 = [bt.TimeDelta(float(value)) for value in np.arange(-1000, 1000, 0.3)]
LIST_10000 = [bt.TimeDelta(float(value)) for value in np.arange(-10000, 10000, 0.3)]

FAST_CASES = (LIST_1, LIST_10)
BIG_O_CASES = (LIST_1, LIST_10, LIST_100, LIST_1000, LIST_10000)  # Useful for local measurements


@pytest.mark.benchmark(group="timedelta_array_construct", **benchmark_options)
@pytest.mark.parametrize("constructor_list", FAST_CASES)
def test___bt_timedelta_array___construct(
    benchmark: BenchmarkFixture,
    constructor_list: list[bt.TimeDelta],
) -> None:
    benchmark(bt.TimeDeltaArray, constructor_list)


@pytest.mark.benchmark(group="timedelta_array_extend", **benchmark_options)
@pytest.mark.parametrize("extend_list", FAST_CASES)
def test___bt_timedelta_array___extend(
    benchmark: BenchmarkFixture,
    extend_list: list[bt.TimeDelta],
) -> None:
    empty_array = bt.TimeDeltaArray()
    benchmark(empty_array.extend, extend_list)
