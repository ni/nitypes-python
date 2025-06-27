from __future__ import annotations

import copy
import pickle
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from typing_extensions import assert_type

from nitypes.waveform import (
    DigitalWaveform,
    DigitalWaveformSignal,
    DigitalWaveformSignalCollection,
)


###############################################################################
# signal collection
###############################################################################
def test___waveform___signals___is_signal_collection() -> None:
    waveform = DigitalWaveform(10, 3)

    assert_type(waveform.signals, DigitalWaveformSignalCollection[np.uint8])
    assert isinstance(waveform.signals, DigitalWaveformSignalCollection)


def test___waveform___signals_len___returns_signal_count() -> None:
    waveform = DigitalWaveform(10, 3)

    assert len(waveform.signals) == 3


def test___int_index___signals_getitem___returns_signal() -> None:
    waveform = DigitalWaveform(10, 3)

    assert_type(waveform.signals[0], DigitalWaveformSignal[np.uint8])
    assert waveform.signals[0].signal_index == 0
    assert waveform.signals[1].signal_index == 1
    assert waveform.signals[2].signal_index == 2


def test___negative_int_index___signals_getitem___returns_signal() -> None:
    waveform = DigitalWaveform(10, 3)

    assert waveform.signals[-1].signal_index == 2
    assert waveform.signals[-2].signal_index == 1
    assert waveform.signals[-3].signal_index == 0


def test___str_index___signals_getitem___returns_signal() -> None:
    waveform = DigitalWaveform(
        10, 3, extended_properties={"NI_LineNames": "port0/line0, port0/line1, port0/line2"}
    )

    assert_type(waveform.signals["port0/line0"], DigitalWaveformSignal[np.uint8])
    assert waveform.signals["port0/line0"].signal_index == 0
    assert waveform.signals["port0/line1"].signal_index == 1
    assert waveform.signals["port0/line2"].signal_index == 2


def test___invalid_str_index___signals_getitem___raises_index_error() -> None:
    waveform = DigitalWaveform(
        10, 3, extended_properties={"NI_LineNames": "port0/line0, port0/line1, port0/line2"}
    )

    with pytest.raises(IndexError) as exc:
        _ = waveform.signals["port0/line3"]

    assert exc.value.args[0] == "port0/line3"


def test___slice_index___signals_getitem___returns_signal() -> None:
    waveform = DigitalWaveform(10, 5)

    assert_type(waveform.signals[1:3], Sequence[DigitalWaveformSignal[np.uint8]])
    assert [signal.signal_index for signal in waveform.signals[1:3]] == [1, 2]
    assert [signal.signal_index for signal in waveform.signals[2:]] == [2, 3, 4]
    assert [signal.signal_index for signal in waveform.signals[:3]] == [0, 1, 2]


def test___negative_slice_index___signals_getitem___returns_signal() -> None:
    waveform = DigitalWaveform(10, 5)

    assert [signal.signal_index for signal in waveform.signals[-2:]] == [3, 4]
    assert [signal.signal_index for signal in waveform.signals[:-2]] == [0, 1, 2]
    assert [signal.signal_index for signal in waveform.signals[-3:-1]] == [2, 3]


###############################################################################
# signal name
###############################################################################
def test___signal___set_signal_name___sets_name() -> None:
    waveform = DigitalWaveform(10, 3)

    waveform.signals[0].name = "port0/line0"
    waveform.signals[1].name = "port0/line1"
    waveform.signals[2].name = "port0/line2"

    assert waveform.extended_properties["NI_LineNames"] == "port0/line0, port0/line1, port0/line2"


def test___signal_with_line_names___get_signal_name___returns_line_name() -> None:
    waveform = DigitalWaveform(
        10, 3, extended_properties={"NI_LineNames": "port0/line0, port0/line1, port0/line2"}
    )

    assert waveform.signals[0].name == "port0/line0"
    assert waveform.signals[1].name == "port0/line1"
    assert waveform.signals[2].name == "port0/line2"


def test___signal_with_line_names___set_signal_name___returns_line_name() -> None:
    waveform = DigitalWaveform(
        10, 3, extended_properties={"NI_LineNames": "port0/line0, port0/line1, port0/line2"}
    )

    waveform.signals[1].name = "MySignal"

    assert waveform.extended_properties["NI_LineNames"] == "port0/line0, MySignal, port0/line2"


###############################################################################
# signal data
###############################################################################
def test___waveform___get_signal_data___returns_line_data() -> None:
    waveform = DigitalWaveform.from_lines([[0, 1, 2], [3, 4, 5]], np.uint8)

    assert_type(waveform.signals[0].data, npt.NDArray[np.uint8])
    assert len(waveform.signals) == 3
    assert waveform.signals[0].data.tolist() == [0, 3]
    assert waveform.signals[1].data.tolist() == [1, 4]
    assert waveform.signals[2].data.tolist() == [2, 5]


###############################################################################
# magic methods
###############################################################################
@pytest.mark.parametrize(
    "left, right",
    [
        (
            DigitalWaveform.from_lines([0, 1, 2, 3], np.uint8).signals[0],
            DigitalWaveform.from_lines([0, 1, 2, 3], np.uint8).signals[0],
        ),
        (
            DigitalWaveform.from_lines([False, True, False], np.bool).signals[0],
            DigitalWaveform.from_lines([False, True, False], np.bool).signals[0],
        ),
        # Equality does not take the signal index or signal name into account.
        (
            DigitalWaveform(3, 2, extended_properties={"NI_LineNames": "0, 1"}).signals[0],
            DigitalWaveform(3, 2, extended_properties={"NI_LineNames": "0, 1"}).signals[1],
        ),
    ],
)
def test___same_value___equality___equal(
    left: DigitalWaveformSignal[Any], right: DigitalWaveformSignal[Any]
) -> None:
    assert left == right
    assert not (left != right)


@pytest.mark.parametrize(
    "left, right",
    [
        (
            DigitalWaveform.from_lines([0, 1, 2, 3], np.uint8).signals[0],
            DigitalWaveform.from_lines([0, 1, 4, 3], np.uint8).signals[0],
        ),
        (
            DigitalWaveform.from_lines([False, True, False], np.bool).signals[0],
            DigitalWaveform.from_lines([False, False, False], np.bool).signals[0],
        ),
    ],
)
def test___different_value___equality___not_equal(
    left: DigitalWaveformSignal[Any], right: DigitalWaveformSignal[Any]
) -> None:
    assert not (left == right)
    assert left != right


@pytest.mark.parametrize(
    "value, expected_repr",
    [
        (
            DigitalWaveform(3, 2).signals[0],
            "nitypes.waveform.DigitalWaveformSignal(data=array([0, 0, 0], dtype=uint8))",
        ),
        (
            DigitalWaveform(3, 2, np.bool).signals[0],
            "nitypes.waveform.DigitalWaveformSignal(data=array([False, False, False]))",
        ),
        (
            DigitalWaveform(
                3, 2, extended_properties={"NI_LineNames": "port0/line0, port0/line1"}
            ).signals[1],
            "nitypes.waveform.DigitalWaveformSignal(name='port0/line1', data=array([0, 0, 0], dtype=uint8))",
        ),
    ],
)
def test___various_values___repr___looks_ok(
    value: DigitalWaveformSignal[Any], expected_repr: str
) -> None:
    assert repr(value) == expected_repr


_VARIOUS_VALUES = [
    DigitalWaveform(3, 2).signals[0],
    DigitalWaveform(3, 2, np.bool).signals[0],
    DigitalWaveform(3, 2, extended_properties={"NI_LineNames": "port0/line0, port0/line1"}).signals[
        1
    ],
]


@pytest.mark.parametrize("value", _VARIOUS_VALUES)
def test___various_values___copy___makes_shallow_copy(value: DigitalWaveformSignal[Any]) -> None:
    new_value = copy.copy(value)

    _assert_shallow_copy(new_value, value)


def _assert_shallow_copy(
    value: DigitalWaveformSignal[Any], other: DigitalWaveformSignal[Any]
) -> None:
    assert value == other
    assert value is not other
    assert value._owner is other._owner


@pytest.mark.parametrize("value", _VARIOUS_VALUES)
def test___various_values___deepcopy___makes_deep_copy(
    value: DigitalWaveformSignal[Any],
) -> None:
    new_value = copy.deepcopy(value)

    _assert_deep_copy(new_value, value)


def _assert_deep_copy(value: DigitalWaveformSignal[Any], other: DigitalWaveformSignal[Any]) -> None:
    assert value == other
    assert value is not other
    assert value._owner is not other._owner


@pytest.mark.parametrize("value", _VARIOUS_VALUES)
def test___various_values___pickle_unpickle___makes_deep_copy(
    value: DigitalWaveformSignal[Any],
) -> None:
    new_value = pickle.loads(pickle.dumps(value))

    _assert_deep_copy(new_value, value)


def test___waveform___pickle___references_public_modules() -> None:
    value = DigitalWaveform(3, 2).signals[0]

    value_bytes = pickle.dumps(value)

    assert b"nitypes.waveform" in value_bytes
    assert b"nitypes.waveform._digital" not in value_bytes
