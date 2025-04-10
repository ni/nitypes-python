import itertools
from typing import Any, assert_type

import numpy as np

from nitypes.waveform import AnalogWaveform


def test___no_args___create___creates_empty_waveform_with_default_dtype() -> None:
    waveform = AnalogWaveform.create()

    assert waveform.sample_count == waveform.capacity == len(waveform.raw_data) == 0
    assert waveform.raw_data.dtype == np.float64
    assert_type(waveform, AnalogWaveform[np.float64])


def test___sample_count___create___creates_waveform_with_sample_count_and_default_dtype() -> None:
    waveform = AnalogWaveform.create(10)

    assert waveform.sample_count == waveform.capacity == len(waveform.raw_data) == 10
    assert waveform.raw_data.dtype == np.float64
    assert_type(waveform, AnalogWaveform[np.float64])


def test___sample_count_and_dtype___create___creates_waveform_with_sample_count_and_dtype() -> None:
    waveform = AnalogWaveform.create(10, np.int32)

    assert waveform.sample_count == waveform.capacity == len(waveform.raw_data) == 10
    assert waveform.raw_data.dtype == np.int32
    assert_type(waveform, AnalogWaveform[np.int32])


def test___sample_count_and_dtype_str___create___creates_waveform_with_sample_count_and_dtype() -> (
    None
):
    waveform = AnalogWaveform.create(10, "i4")

    assert waveform.sample_count == waveform.capacity == len(waveform.raw_data) == 10
    assert waveform.raw_data.dtype == np.int32
    assert_type(waveform, AnalogWaveform[Any])  # dtype not inferred from string


def test___sample_count_and_dtype_any___create___creates_waveform_with_sample_count_and_dtype() -> (
    None
):
    dtype: np.dtype[Any] = np.dtype(np.int32)
    waveform = AnalogWaveform.create(10, dtype)

    assert waveform.sample_count == waveform.capacity == len(waveform.raw_data) == 10
    assert waveform.raw_data.dtype == np.int32
    assert_type(waveform, AnalogWaveform[Any])  # dtype not inferred from np.dtype[Any]


def test___sample_count_dtype_and_capacity___create___creates_waveform_with_sample_count_dtype_and_capacity() -> (
    None
):
    waveform = AnalogWaveform.create(10, np.int32, 20)

    assert waveform.sample_count == len(waveform.raw_data) == 10
    assert waveform.capacity == 20
    assert waveform.raw_data.dtype == np.int32
    assert_type(waveform, AnalogWaveform[np.int32])


def test___unspecified_dtype___from_iter___creates_waveform_with_default_dtype() -> None:
    waveform = AnalogWaveform.from_iter([1.1, 2.2, 3.3, 4.4, 5.5])

    assert waveform.raw_data.tolist() == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert waveform.raw_data.dtype == np.float64
    assert_type(waveform, AnalogWaveform[np.float64])


def test___dtype___from_iter___creates_waveform_with_specified_dtype() -> None:
    waveform = AnalogWaveform.from_iter([1, 2, 3, 4, 5], np.int32)

    assert waveform.raw_data.tolist() == [1, 2, 3, 4, 5]
    assert waveform.raw_data.dtype == np.int32
    assert_type(waveform, AnalogWaveform[np.int32])


def test___infinite_iterator_and_sample_count___from_iter___creates_waveform_with_specified_sample_count() -> (
    None
):
    waveform = AnalogWaveform.from_iter(itertools.repeat(3), sample_count=5)

    assert waveform.raw_data.tolist() == [3, 3, 3, 3, 3]
    assert waveform.sample_count == 5


def test___float64_array___from_ndarray___creates_waveform_with_float64_dtype() -> None:
    data = np.array([1.1, 2.2, 3.3, 4.4, 5.5], np.float64)

    waveform = AnalogWaveform.from_ndarray(data)

    assert waveform.raw_data.tolist() == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert waveform.raw_data.dtype == np.float64
    assert_type(waveform, AnalogWaveform[np.float64])


def test___int32_array___from_ndarray___creates_waveform_with_int32_dtype() -> None:
    data = np.array([1, 2, 3, 4, 5], np.int32)

    waveform = AnalogWaveform.from_ndarray(data)

    assert waveform.raw_data.tolist() == [1, 2, 3, 4, 5]
    assert waveform.raw_data.dtype == np.int32
    assert_type(waveform, AnalogWaveform[np.int32])


def test___copy___from_ndarray___creates_waveform_with_array_copy() -> None:
    data = np.array([1, 2, 3, 4, 5], np.int32)

    waveform = AnalogWaveform.from_ndarray(data, copy=True)

    assert waveform._data is not data
    assert waveform.raw_data.tolist() == data.tolist()


def test___no_copy___from_ndarray___creates_waveform_with_array_reference() -> None:
    data = np.array([1, 2, 3, 4, 5], np.int32)

    waveform = AnalogWaveform.from_ndarray(data, copy=False)

    assert waveform._data is data
    assert waveform.raw_data.tolist() == data.tolist()


def test___array_subset___from_ndarray___creates_waveform_with_array_subset() -> None:
    data = np.array([1, 2, 3, 4, 5], np.int32)

    waveform = AnalogWaveform.from_ndarray(data, start_index=1, sample_count=3)

    assert waveform.raw_data.tolist() == [2, 3, 4]
