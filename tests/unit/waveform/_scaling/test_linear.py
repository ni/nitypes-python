from __future__ import annotations

from typing import SupportsFloat

import numpy as np
import numpy.typing as npt
import pytest

from nitypes._typing import assert_type
from nitypes.waveform import LinearScaleMode


@pytest.mark.parametrize(
    "gain, offset",
    [
        (1.0, 0.0),
        (1.2345, 0.006789),
        (3, 4),
        (np.float64(1.2345), np.float64(0.006789)),
        # np.float32 is not a subclass of float, but it supports __float__.
        (np.float32(1.2345), np.float32(0.006789)),
    ],
)
def test___gain_and_offset___construct___constructs_with_gain_and_offset(
    gain: SupportsFloat, offset: SupportsFloat
) -> None:
    scale_mode = LinearScaleMode(gain, offset)

    assert scale_mode.gain == gain
    assert scale_mode.offset == offset


@pytest.mark.parametrize(
    "gain, offset, expected_message",
    [
        (None, 0.0, "The gain must be a floating point number."),
        ("1.0", 0.0, "The gain must be a floating point number."),
        (1.0, "0.0", "The offset must be a floating point number."),
    ],
)
def test__invalid_gain_or_offset___construct___raises_type_error(
    gain: object, offset: object, expected_message: str
) -> None:
    with pytest.raises(TypeError) as exc:
        _ = LinearScaleMode(gain, offset)  # type: ignore[arg-type]

    assert exc.value.args[0].startswith(expected_message)


def test___empty_ndarray___transform_data___returns_empty_scaled_data() -> None:
    waveform = np.zeros(0, np.float64)
    scale_mode = LinearScaleMode(3.0, 4.0)

    scaled_data = scale_mode._transform_data(waveform)

    assert_type(scaled_data, npt.NDArray[np.float64])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.float64
    assert list(scaled_data) == []


def test___float32_ndarray___transform_data___returns_float32_scaled_data() -> None:
    raw_data = np.array([0, 1, 2, 3], np.float32)
    scale_mode = LinearScaleMode(3.0, 4.0)

    scaled_data = scale_mode._transform_data(raw_data)

    assert_type(scaled_data, npt.NDArray[np.float32])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.float32
    assert list(scaled_data) == [4.0, 7.0, 10.0, 13.0]


def test___float64_ndarray___transform_data___returns_float64_scaled_data() -> None:
    raw_data = np.array([0, 1, 2, 3], np.float64)
    scale_mode = LinearScaleMode(3.0, 4.0)

    scaled_data = scale_mode._transform_data(raw_data)

    assert_type(scaled_data, npt.NDArray[np.float64])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.float64
    assert list(scaled_data) == [4.0, 7.0, 10.0, 13.0]


def test___scale_mode___repr___looks_ok() -> None:
    scale_mode = LinearScaleMode(1.2345, 0.006789)

    assert repr(scale_mode) == "nitypes.waveform.LinearScaleMode(1.2345, 0.006789)"
