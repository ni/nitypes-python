from __future__ import annotations

import datetime as dt
import operator

import hightime as ht
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import nitypes.bintime as bt


# Note: constructing bt.DateTime from a fixed date is slow because it creates a
# ht.datetime behind the scenes and then converts it to ticks.
@pytest.mark.benchmark(group="datetime_construct")
def test___bt_datetime___construct(
    benchmark: BenchmarkFixture,
) -> None:
    benchmark(bt.DateTime, 2025, 1, 1, tzinfo=dt.timezone.utc)


@pytest.mark.benchmark(group="datetime_construct")
def test___dt_datetime___construct(
    benchmark: BenchmarkFixture,
) -> None:
    benchmark(dt.datetime, 2025, 1, 1, tzinfo=dt.timezone.utc)


@pytest.mark.benchmark(group="datetime_construct")
def test___ht_datetime___construct(
    benchmark: BenchmarkFixture,
) -> None:
    benchmark(ht.datetime, 2025, 1, 1, tzinfo=dt.timezone.utc)


@pytest.mark.benchmark(group="datetime_construct")
def test___bt_datetime___from_ticks(
    benchmark: BenchmarkFixture,
) -> None:
    benchmark(bt.DateTime.from_ticks, 1 << 60 | 1 << 31)


@pytest.mark.benchmark(group="datetime_construct")
def test___bt_datetime___from_tuple(
    benchmark: BenchmarkFixture,
) -> None:
    benchmark(bt.DateTime.from_tuple, bt.TimeValueTuple(1 << 28, 1 << 31))


@pytest.mark.benchmark(group="datetime_now")
def test___bt_datetime___now(
    benchmark: BenchmarkFixture,
) -> None:
    benchmark(bt.DateTime.now, dt.timezone.utc)


@pytest.mark.benchmark(group="datetime_now")
def test___dt_datetime___now(
    benchmark: BenchmarkFixture,
) -> None:
    benchmark(dt.datetime.now, dt.timezone.utc)


@pytest.mark.benchmark(group="datetime_now")
def test___ht_datetime___now(
    benchmark: BenchmarkFixture,
) -> None:
    benchmark(ht.datetime.now, dt.timezone.utc)


@pytest.mark.benchmark(group="datetime_lt")
def test___bt_datetime___lt(
    benchmark: BenchmarkFixture,
) -> None:
    t1 = bt.DateTime.now(dt.timezone.utc)
    t2 = bt.DateTime.now(dt.timezone.utc)
    benchmark(operator.lt, t1, t2)


@pytest.mark.benchmark(group="datetime_lt")
def test___dt_datetime___lt(
    benchmark: BenchmarkFixture,
) -> None:
    t1 = dt.datetime.now(dt.timezone.utc)
    t2 = dt.datetime.now(dt.timezone.utc)
    benchmark(operator.lt, t1, t2)


@pytest.mark.benchmark(group="datetime_lt")
def test___ht_datetime___lt(
    benchmark: BenchmarkFixture,
) -> None:
    t1 = ht.datetime.now(dt.timezone.utc)
    t2 = ht.datetime.now(dt.timezone.utc)
    benchmark(operator.lt, t1, t2)


@pytest.mark.benchmark(group="datetime_add")
def test___bt_datetime___add(
    benchmark: BenchmarkFixture,
) -> None:
    t1 = bt.DateTime.now(dt.timezone.utc)
    t2 = bt.TimeDelta(1e-3)
    benchmark(operator.add, t1, t2)


@pytest.mark.benchmark(group="datetime_add")
def test___dt_datetime___add(
    benchmark: BenchmarkFixture,
) -> None:
    t1 = dt.datetime.now(dt.timezone.utc)
    t2 = dt.timedelta(milliseconds=1)
    benchmark(operator.add, t1, t2)


@pytest.mark.benchmark(group="datetime_add")
def test___ht_datetime___add(
    benchmark: BenchmarkFixture,
) -> None:
    t1 = ht.datetime.now(dt.timezone.utc)
    t2 = ht.timedelta(milliseconds=1)
    benchmark(operator.add, t1, t2)
