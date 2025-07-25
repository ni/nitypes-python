from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import nitypes.bintime as bt

pytestmark = pytest.mark.benchmark

SHORT_LIST: list[bt.TimeDelta] = [bt.TimeDelta(float(value)) for value in np.arange(-2, 2, 0.5)]
LONG_LIST: list[bt.TimeDelta] = [bt.TimeDelta(float(value)) for value in np.arange(-100, 100, 0.3)]


@pytest.mark.benchmark(group="timedelta_array_construct")
def test___bt_timedelta_array___construct_short(
    benchmark: BenchmarkFixture,
) -> None:
    benchmark(bt.TimeDeltaArray, SHORT_LIST)


@pytest.mark.benchmark(group="timedelta_array_construct")
def test___bt_timedelta_array___construct_long(
    benchmark: BenchmarkFixture,
) -> None:
    benchmark(bt.TimeDeltaArray, LONG_LIST)
