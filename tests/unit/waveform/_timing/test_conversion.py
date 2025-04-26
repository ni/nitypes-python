from __future__ import annotations

import datetime as dt

import hightime as ht

from nitypes._typing import assert_type
from nitypes.waveform import PrecisionTiming, SampleIntervalMode, Timing
from nitypes.waveform._timing import convert_timing


def test___standard_to_standard___convert_timing___returns_original_object() -> None:
    value_in = Timing.create_with_regular_interval(
        dt.timedelta(milliseconds=1), dt.datetime(2025, 1, 1), dt.timedelta(seconds=1)
    )

    value_out = convert_timing(Timing, value_in)

    assert_type(value_out, Timing)
    assert value_out is value_in


def test___precision_to_precision___convert_timing___returns_original_object() -> None:
    value_in = PrecisionTiming.create_with_regular_interval(
        ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)
    )

    value_out = convert_timing(PrecisionTiming, value_in)

    assert_type(value_out, PrecisionTiming)
    assert value_out is value_in


def test___standard_to_precision_empty___convert_timing___returns_equivalent_timing() -> None:
    value_in = Timing.empty

    value_out = convert_timing(PrecisionTiming, value_in)

    assert_type(value_out, PrecisionTiming)
    assert isinstance(value_out, PrecisionTiming)
    assert value_out == PrecisionTiming.empty


def test___precision_to_standard_empty___convert_timing___returns_equivalent_timing() -> None:
    value_in = PrecisionTiming.empty

    value_out = convert_timing(Timing, value_in)

    assert_type(value_out, Timing)
    assert isinstance(value_out, Timing)
    assert value_out == Timing.empty


def test___standard_to_precision_regular_interval___convert_timing___returns_equivalent_timing() -> (
    None
):
    value_in = Timing.create_with_regular_interval(
        dt.timedelta(milliseconds=1), dt.datetime(2025, 1, 1), dt.timedelta(seconds=1)
    )

    value_out = convert_timing(PrecisionTiming, value_in)

    assert_type(value_out, PrecisionTiming)
    assert isinstance(value_out, PrecisionTiming)
    assert value_out.sample_interval_mode == SampleIntervalMode.REGULAR
    assert value_out.sample_interval == ht.timedelta(milliseconds=1)
    assert value_out.timestamp == ht.datetime(2025, 1, 1)
    assert value_out.time_offset == ht.timedelta(seconds=1)


def test___precision_to_standard_regular_interval___convert_timing___returns_equivalent_timing() -> (
    None
):
    value_in = PrecisionTiming.create_with_regular_interval(
        ht.timedelta(milliseconds=1), ht.datetime(2025, 1, 1), ht.timedelta(seconds=1)
    )

    value_out = convert_timing(Timing, value_in)

    assert_type(value_out, Timing)
    assert isinstance(value_out, Timing)
    assert value_out.sample_interval_mode == SampleIntervalMode.REGULAR
    assert value_out.sample_interval == dt.timedelta(milliseconds=1)
    assert value_out.timestamp == dt.datetime(2025, 1, 1)
    assert value_out.time_offset == dt.timedelta(seconds=1)


def test___standard_to_precision_irregular_interval___convert_timing___returns_equivalent_timing() -> (
    None
):
    value_in = Timing.create_with_irregular_interval(
        [dt.datetime(2025, 1, 1), dt.datetime(2025, 1, 2)]
    )

    value_out = convert_timing(PrecisionTiming, value_in)

    assert_type(value_out, PrecisionTiming)
    assert isinstance(value_out, PrecisionTiming)
    assert value_out.sample_interval_mode == SampleIntervalMode.IRREGULAR
    assert value_out._timestamps == [ht.datetime(2025, 1, 1), ht.datetime(2025, 1, 2)]


def test___precision_to_standard_irregular_interval___convert_timing___returns_equivalent_timing() -> (
    None
):
    value_in = PrecisionTiming.create_with_irregular_interval(
        [ht.datetime(2025, 1, 1), ht.datetime(2025, 1, 2)]
    )

    value_out = convert_timing(Timing, value_in)

    assert_type(value_out, Timing)
    assert isinstance(value_out, Timing)
    assert value_out.sample_interval_mode == SampleIntervalMode.IRREGULAR
    assert value_out._timestamps == [dt.datetime(2025, 1, 1), dt.datetime(2025, 1, 2)]
