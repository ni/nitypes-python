from __future__ import annotations  # noqa: D100 - Missing docstring in public module

import datetime as dt
import operator

import hightime as ht
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import nitypes.bintime as bt


@pytest.mark.benchmark(group="timedelta_construct")
def test___bt_timedelta___construct(  # noqa: D103 - Missing docstring in public function
    benchmark: BenchmarkFixture,
) -> None:
    benchmark(bt.TimeDelta, 1e-3)


@pytest.mark.benchmark(group="timedelta_construct")
def test___dt_timedelta___construct(  # noqa: D103 - Missing docstring in public function
    benchmark: BenchmarkFixture,
) -> None:
    benchmark(dt.timedelta, milliseconds=1)


@pytest.mark.benchmark(group="timedelta_construct")
def test___ht_timedelta___construct(  # noqa: D103 - Missing docstring in public function
    benchmark: BenchmarkFixture,
) -> None:
    benchmark(ht.timedelta, milliseconds=1)


@pytest.mark.benchmark(group="timedelta_construct")
def test___bt_timedelta___from_ticks(  # noqa: D103 - Missing docstring in public function
    benchmark: BenchmarkFixture,
) -> None:
    benchmark(bt.TimeDelta.from_ticks, 1 << 60 | 1 << 31)


@pytest.mark.benchmark(group="timedelta_construct")
def test___bt_timedelta___from_tuple(  # noqa: D103 - Missing docstring in public function
    benchmark: BenchmarkFixture,
) -> None:
    benchmark(bt.TimeDelta.from_tuple, bt.TimeValueTuple(1 << 28, 1 << 31))


@pytest.mark.benchmark(group="timedelta_eq")
def test___bt_timedelta___eq(  # noqa: D103 - Missing docstring in public function
    benchmark: BenchmarkFixture,
) -> None:
    t1 = bt.TimeDelta(1e-3)
    t2 = bt.TimeDelta(2e-3)
    benchmark(operator.eq, t1, t2)


@pytest.mark.benchmark(group="timedelta_eq")
def test___dt_timedelta___eq(  # noqa: D103 - Missing docstring in public function
    benchmark: BenchmarkFixture,
) -> None:
    t1 = dt.timedelta(milliseconds=1)
    t2 = dt.timedelta(milliseconds=2)
    benchmark(operator.eq, t1, t2)


@pytest.mark.benchmark(group="timedelta_eq")
def test___ht_timedelta___eq(  # noqa: D103 - Missing docstring in public function
    benchmark: BenchmarkFixture,
) -> None:
    t1 = ht.timedelta(milliseconds=1)
    t2 = ht.timedelta(milliseconds=2)
    benchmark(operator.lt, t1, t2)


@pytest.mark.benchmark(group="timedelta_lt")
def test___bt_timedelta___lt(  # noqa: D103 - Missing docstring in public function
    benchmark: BenchmarkFixture,
) -> None:
    t1 = bt.TimeDelta(1e-3)
    t2 = bt.TimeDelta(2e-3)
    benchmark(operator.lt, t1, t2)


@pytest.mark.benchmark(group="timedelta_lt")
def test___dt_timedelta___lt(  # noqa: D103 - Missing docstring in public function
    benchmark: BenchmarkFixture,
) -> None:
    t1 = dt.timedelta(milliseconds=1)
    t2 = dt.timedelta(milliseconds=2)
    benchmark(operator.lt, t1, t2)


@pytest.mark.benchmark(group="timedelta_lt")
def test___ht_timedelta___lt(  # noqa: D103 - Missing docstring in public function
    benchmark: BenchmarkFixture,
) -> None:
    t1 = ht.timedelta(milliseconds=1)
    t2 = ht.timedelta(milliseconds=2)
    benchmark(operator.lt, t1, t2)


@pytest.mark.benchmark(group="timedelta_add")
def test___bt_timedelta___add(  # noqa: D103 - Missing docstring in public function
    benchmark: BenchmarkFixture,
) -> None:
    t1 = bt.TimeDelta(1e-3)
    t2 = bt.TimeDelta(2e-3)
    benchmark(operator.add, t1, t2)


@pytest.mark.benchmark(group="timedelta_add")
def test___dt_timedelta___add(  # noqa: D103 - Missing docstring in public function
    benchmark: BenchmarkFixture,
) -> None:
    t1 = dt.timedelta(milliseconds=1)
    t2 = dt.timedelta(milliseconds=2)
    benchmark(operator.add, t1, t2)


@pytest.mark.benchmark(group="timedelta_add")
def test___ht_timedelta___add(  # noqa: D103 - Missing docstring in public function
    benchmark: BenchmarkFixture,
) -> None:
    t1 = ht.timedelta(milliseconds=1)
    t2 = ht.timedelta(milliseconds=2)
    benchmark(operator.add, t1, t2)


@pytest.mark.benchmark(group="timedelta_mul")
def test___bt_timedelta___mul(  # noqa: D103 - Missing docstring in public function
    benchmark: BenchmarkFixture,
) -> None:
    t1 = bt.TimeDelta(1e-3)
    t2 = 0.1
    benchmark(operator.mul, t1, t2)


@pytest.mark.benchmark(group="timedelta_mul")
def test___dt_timedelta___mul(  # noqa: D103 - Missing docstring in public function
    benchmark: BenchmarkFixture,
) -> None:
    t1 = dt.timedelta(milliseconds=1)
    t2 = 0.1
    benchmark(operator.mul, t1, t2)


@pytest.mark.benchmark(group="timedelta_mul")
def test___ht_timedelta___mul(  # noqa: D103 - Missing docstring in public function
    benchmark: BenchmarkFixture,
) -> None:
    t1 = ht.timedelta(milliseconds=1)
    t2 = 0.1
    benchmark(operator.mul, t1, t2)


@pytest.mark.benchmark(group="timedelta_total_seconds")
def test___bt_timedelta___total_seconds(  # noqa: D103 - Missing docstring in public function
    benchmark: BenchmarkFixture,
) -> None:
    t = bt.TimeDelta(1e-3)
    benchmark(t.total_seconds)


@pytest.mark.benchmark(group="timedelta_total_seconds")
def test___dt_timedelta___total_seconds(  # noqa: D103 - Missing docstring in public function
    benchmark: BenchmarkFixture,
) -> None:
    t = dt.timedelta(milliseconds=1)
    benchmark(t.total_seconds)


@pytest.mark.benchmark(group="timedelta_total_seconds")
def test___ht_timedelta___total_seconds(  # noqa: D103 - Missing docstring in public function
    benchmark: BenchmarkFixture,
) -> None:
    t = ht.timedelta(milliseconds=1)
    benchmark(t.total_seconds)


@pytest.mark.benchmark(group="timedelta_total_seconds")
def test___bt_timedelta___precision_total_seconds(  # noqa: D103 - Missing docstring in public function
    benchmark: BenchmarkFixture,
) -> None:
    t = bt.TimeDelta(1e-3)
    benchmark(t.precision_total_seconds)


@pytest.mark.benchmark(group="timedelta_total_seconds")
def test___ht_timedelta___precision_total_seconds(  # noqa: D103 - Missing docstring in public function
    benchmark: BenchmarkFixture,
) -> None:
    t = ht.timedelta(milliseconds=1)
    benchmark(t.precision_total_seconds)
