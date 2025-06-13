from __future__ import annotations

import datetime as dt

import hightime as ht
import pytest
from typing_extensions import assert_type

import nitypes.bintime as bt
from nitypes.waveform import SampleIntervalMode, Timing


def test___bt_to_bt___convert_timing___returns_original_object() -> None:
    value_in = Timing.create_with_regular_interval(
        bt.TimeDelta(1e-3), bt.DateTime(2025, 1, 1, tzinfo=dt.timezone.utc), bt.TimeDelta(1)
    )

    value_out = value_in.to_bintime()

    assert_type(value_out, Timing[bt.DateTime, bt.TimeDelta, bt.TimeDelta])
    assert value_out is value_in


def test___dt_to_dt___convert_timing___returns_original_object() -> None:
    value_in = Timing.create_with_regular_interval(
        dt.timedelta(milliseconds=1), dt.datetime(2025, 1, 1), dt.timedelta(seconds=1)
    )

    value_out = value_in.to_datetime()

    assert_type(value_out, Timing[dt.datetime, dt.timedelta, dt.timedelta])
    assert value_out is value_in


def test___ht_to_ht___convert_timing___returns_original_object() -> None:
    value_in = Timing.create_with_regular_interval(
        ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)
    )

    value_out = value_in.to_hightime()

    assert_type(value_out, Timing[ht.datetime, ht.timedelta, ht.timedelta])

    assert value_out is value_in


def test___empty_to_bt___convert_timing___returns_equivalent_timing() -> None:
    value_in = Timing.empty

    value_out = value_in.to_bintime()

    assert_type(value_out, Timing[bt.DateTime, bt.TimeDelta, bt.TimeDelta])
    # Can't check runtime field type because they are all None.
    assert value_out == value_in


def test___empty_to_dt___convert_timing___returns_original_object() -> None:
    value_in = Timing.empty

    value_out = value_in.to_datetime()

    assert_type(value_out, Timing[dt.datetime, dt.timedelta, dt.timedelta])
    # Can't check runtime field type because they are all None.
    assert value_out is value_in


def test___empty_to_ht___convert_timing___returns_equivalent_timing() -> None:
    value_in = Timing.empty

    value_out = value_in.to_hightime()

    assert_type(value_out, Timing[ht.datetime, ht.timedelta, ht.timedelta])
    # Can't check runtime field type because they are all None.
    assert value_out == value_in


def test___dt_to_bt_regular_interval___convert_timing___returns_equivalent_timing() -> None:
    value_in = Timing.create_with_regular_interval(
        dt.timedelta(milliseconds=1),
        dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc),
        dt.timedelta(seconds=1),
    )

    value_out = value_in.to_bintime()

    assert_type(value_out, Timing[bt.DateTime, bt.TimeDelta, bt.TimeDelta])
    assert isinstance(value_out.timestamp, bt.DateTime)
    assert isinstance(value_out.time_offset, bt.TimeDelta)
    assert isinstance(value_out.sample_interval, bt.TimeDelta)
    assert value_out.sample_interval_mode == SampleIntervalMode.REGULAR
    assert value_out.sample_interval.total_seconds() == pytest.approx(1e-3)  # rounding
    assert value_out.timestamp == bt.DateTime(2025, 1, 1, tzinfo=dt.timezone.utc)
    assert value_out.time_offset == bt.TimeDelta(1)


def test___dt_to_ht_regular_interval___convert_timing___returns_equivalent_timing() -> None:
    value_in = Timing.create_with_regular_interval(
        dt.timedelta(milliseconds=1), dt.datetime(2025, 1, 1), dt.timedelta(seconds=1)
    )

    value_out = value_in.to_hightime()

    assert_type(value_out, Timing[ht.datetime, ht.timedelta, ht.timedelta])
    assert isinstance(value_out.timestamp, ht.datetime)
    assert isinstance(value_out.time_offset, ht.timedelta)
    assert isinstance(value_out.sample_interval, ht.timedelta)
    assert value_out.sample_interval_mode == SampleIntervalMode.REGULAR
    assert value_out.sample_interval == ht.timedelta(milliseconds=1)
    assert value_out.timestamp == ht.datetime(2025, 1, 1)
    assert value_out.time_offset == ht.timedelta(seconds=1)


def test___ht_to_dt_regular_interval___convert_timing___returns_equivalent_timing() -> None:
    value_in = Timing.create_with_regular_interval(
        ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)
    )

    value_out = value_in.to_datetime()

    assert_type(value_out, Timing[dt.datetime, dt.timedelta, dt.timedelta])
    assert isinstance(value_out.timestamp, dt.datetime)
    assert isinstance(value_out.time_offset, dt.timedelta)
    assert isinstance(value_out.sample_interval, dt.timedelta)
    assert value_out.sample_interval_mode == SampleIntervalMode.REGULAR
    assert value_out.sample_interval == dt.timedelta(milliseconds=1)
    assert value_out.timestamp == dt.datetime(2025, 1, 1)
    assert value_out.time_offset == dt.timedelta(seconds=1)


def test___dt_to_ht_irregular_interval___convert_timing___returns_equivalent_timing() -> None:
    value_in = Timing.create_with_irregular_interval(
        [dt.datetime(2025, 1, 1), dt.datetime(2025, 1, 2)]
    )

    value_out = value_in.to_hightime()

    assert_type(value_out, Timing[ht.datetime, ht.timedelta, ht.timedelta])
    assert value_out._timestamps is not None and all(
        isinstance(ts, ht.datetime) for ts in value_out._timestamps
    )
    assert value_out.sample_interval_mode == SampleIntervalMode.IRREGULAR
    assert value_out._timestamps == [ht.datetime(2025, 1, 1), ht.datetime(2025, 1, 2)]


def test___ht_to_dt_irregular_interval___convert_timing___returns_equivalent_timing() -> None:
    value_in = Timing.create_with_irregular_interval(
        [ht.datetime(2025, 1, 1), ht.datetime(2025, 1, 2)]
    )

    value_out = value_in.to_datetime()

    assert_type(value_out, Timing[dt.datetime, dt.timedelta, dt.timedelta])
    assert value_out._timestamps is not None and all(
        isinstance(ts, dt.datetime) for ts in value_out._timestamps
    )
    assert value_out.sample_interval_mode == SampleIntervalMode.IRREGULAR
    assert value_out._timestamps == [dt.datetime(2025, 1, 1), dt.datetime(2025, 1, 2)]
