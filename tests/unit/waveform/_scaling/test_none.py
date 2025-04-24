from __future__ import annotations

import sys

import numpy as np
import numpy.typing as npt

from nitypes.waveform import NO_SCALING, NoneScaleMode

if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type


def test___no_scaling___type_is_none_scale_mode() -> None:
    assert_type(NO_SCALING, NoneScaleMode)
    assert isinstance(NO_SCALING, NoneScaleMode)


def test___empty_ndarray___transform_data___returns_empty_scaled_data() -> None:
    waveform = np.zeros(0, np.float64)

    scaled_data = NO_SCALING._transform_data(waveform)

    assert_type(scaled_data, npt.NDArray[np.float64])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.float64
    assert list(scaled_data) == []


def test___float32_ndarray___transform_data___returns_float32_scaled_data() -> None:
    raw_data = np.array([0, 1, 2, 3], np.float32)

    scaled_data = NO_SCALING._transform_data(raw_data)

    assert_type(scaled_data, npt.NDArray[np.float32])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.float32
    assert list(scaled_data) == [0.0, 1.0, 2.0, 3.0]


def test___float64_ndarray___transform_data___returns_float64_scaled_data() -> None:
    raw_data = np.array([0, 1, 2, 3], np.float64)

    scaled_data = NO_SCALING._transform_data(raw_data)

    assert_type(scaled_data, npt.NDArray[np.float64])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.float64
    assert list(scaled_data) == [0.0, 1.0, 2.0, 3.0]


def test___scale_mode___repr___looks_ok() -> None:
    assert repr(NO_SCALING) == "nitypes.waveform.NoneScaleMode()"
