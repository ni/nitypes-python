from __future__ import annotations

import copy
import pickle

import numpy as np
import numpy.typing as npt
from typing_extensions import assert_type

from nitypes.waveform import NO_SCALING, NoneScaleMode


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


def test___complex64_ndarray___transform_data___returns_complex64_scaled_data() -> None:
    raw_data = np.array([1 + 2j, 3 - 4j], np.complex64)

    scaled_data = NO_SCALING._transform_data(raw_data)

    assert_type(scaled_data, npt.NDArray[np.complex64])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.complex64
    assert list(scaled_data) == [1 + 2j, 3 - 4j]


def test___complex128_ndarray___transform_data___returns_complex128_scaled_data() -> None:
    raw_data = np.array([1 + 2j, 3 - 4j], np.complex128)

    scaled_data = NO_SCALING._transform_data(raw_data)

    assert_type(scaled_data, npt.NDArray[np.complex128])
    assert isinstance(scaled_data, np.ndarray) and scaled_data.dtype == np.complex128
    assert list(scaled_data) == [1 + 2j, 3 - 4j]


def test___scale_mode___repr___looks_ok() -> None:
    assert repr(NO_SCALING) == "nitypes.waveform.NoneScaleMode()"


def test___scale_mode___copy___makes_shallow_copy() -> None:
    new_scale_mode = copy.copy(NO_SCALING)

    assert new_scale_mode == NO_SCALING
    assert new_scale_mode is not NO_SCALING


def test___scale_mode___deepcopy___makes_deep_copy() -> None:
    new_scale_mode = copy.deepcopy(NO_SCALING)

    assert new_scale_mode == NO_SCALING
    assert new_scale_mode is not NO_SCALING


def test___scale_mode___pickle_unpickle___makes_deep_copy() -> None:
    new_scale_mode = pickle.loads(pickle.dumps(NO_SCALING))

    assert new_scale_mode == NO_SCALING
    assert new_scale_mode is not NO_SCALING


def test___scale_mode___pickle___references_public_modules() -> None:
    scale_mode_bytes = pickle.dumps(NO_SCALING)

    assert b"nitypes.waveform" in scale_mode_bytes
    assert b"nitypes.waveform._scaling" not in scale_mode_bytes
