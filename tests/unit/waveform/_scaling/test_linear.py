from __future__ import annotations

import sys

import numpy as np
import numpy.typing as npt
import pytest

from nitypes.waveform import AnalogWaveform, LinearScaleMode

if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type


def test___gain_and_offset___construct___constructs_with_gain_and_offset() -> None:
    scale_mode = LinearScaleMode(1.2345, 0.006789)

    assert scale_mode.gain == 1.2345
    assert scale_mode.offset == 0.006789


def test___empty_waveform___get_scaled_data___returns_empty_scaled_data() -> None:
    waveform = AnalogWaveform()
    scale_mode = LinearScaleMode(3.0, 4.0)

    scaled_data = scale_mode.get_scaled_data(waveform)

    assert_type(scaled_data, npt.NDArray[np.float64])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.float64
    assert list(scaled_data) == []


def test___float64_waveform___get_scaled_data___returns_float64_scaled_data() -> None:
    waveform = AnalogWaveform.from_array_1d([0.0, 1.0, 2.0, 3.0], np.float64)
    scale_mode = LinearScaleMode(3.0, 4.0)

    scaled_data = scale_mode.get_scaled_data(waveform)

    assert_type(scaled_data, npt.NDArray[np.float64])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.float64
    assert list(scaled_data) == [4.0, 7.0, 10.0, 13.0]


def test___int32_waveform___get_scaled_data___returns_float64_scaled_data() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2, 3], np.int32)
    scale_mode = LinearScaleMode(3.0, 4.0)

    scaled_data = scale_mode.get_scaled_data(waveform)

    assert_type(scaled_data, npt.NDArray[np.float64])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.float64
    assert list(scaled_data) == [4.0, 7.0, 10.0, 13.0]


def test___float32_dtype___get_scaled_data___returns_float32_scaled_data() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2, 3], np.int32)
    scale_mode = LinearScaleMode(3.0, 4.0)

    scaled_data = scale_mode.get_scaled_data(waveform, np.float32)

    assert_type(scaled_data, npt.NDArray[np.float32])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.float32
    assert list(scaled_data) == [4.0, 7.0, 10.0, 13.0]


def test___float64_dtype___get_scaled_data___returns_float64_scaled_data() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2, 3], np.int32)
    scale_mode = LinearScaleMode(3.0, 4.0)

    scaled_data = scale_mode.get_scaled_data(waveform, np.float64)

    assert_type(scaled_data, npt.NDArray[np.float64])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.float64
    assert list(scaled_data) == [4.0, 7.0, 10.0, 13.0]


@pytest.mark.parametrize(
    "waveform_dtype",
    [
        np.float32,
        np.float64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ],
)
@pytest.mark.parametrize("scaled_dtype", [np.float32, np.float64])
def test___varying_dtype___get_scaled_data___returns_float64_scaled_data(
    waveform_dtype: npt.DTypeLike, scaled_dtype: npt.DTypeLike
) -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2, 3], waveform_dtype)
    scale_mode = LinearScaleMode(3.0, 4.0)

    scaled_data = scale_mode.get_scaled_data(waveform, scaled_dtype)

    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == scaled_dtype
    assert list(scaled_data) == [4.0, 7.0, 10.0, 13.0]


def test___unsupported_dtype___get_scaled_data___raises_type_error() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2, 3], np.int32)
    scale_mode = LinearScaleMode(3.0, 4.0)

    with pytest.raises(TypeError) as exc:
        _ = scale_mode.get_scaled_data(waveform, np.int32)

    assert exc.value.args[0].startswith("The requested data type is not supported.")
    assert "Supported data types: float32, float64" in exc.value.args[0]


def test___array_subset___get_scaled_data___returns_scaled_data_subset() -> None:
    waveform = AnalogWaveform.from_array_1d([0, 1, 2, 3], np.int32)
    scale_mode = LinearScaleMode(3.0, 4.0)

    scaled_data = scale_mode.get_scaled_data(waveform, start_index=1, sample_count=2)

    assert_type(scaled_data, npt.NDArray[np.float64])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.float64
    assert list(scaled_data) == [7.0, 10.0]
