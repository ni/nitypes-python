from __future__ import annotations

import pytest

from nitypes.waveform import DigitalState


@pytest.mark.parametrize(
    "state, expected_char",
    [
        (DigitalState.FORCE_DOWN, "0"),
        (DigitalState.COMPARE_HIGH, "H"),
        (DigitalState.COMPARE_UNKNOWN, "X"),
    ],
)
def test___state___to_char___returns_char(state: DigitalState, expected_char: str) -> None:
    assert DigitalState.to_char(state) == expected_char


@pytest.mark.parametrize("state", [-1, 8, 255])
def test___invalid_state___to_char___returns_question_mark(state: DigitalState) -> None:
    assert DigitalState.to_char(state) == "?"


@pytest.mark.parametrize(
    "char, expected_state",
    [
        ("0", DigitalState.FORCE_DOWN),
        ("H", DigitalState.COMPARE_HIGH),
        ("X", DigitalState.COMPARE_UNKNOWN),
    ],
)
def test___char___from_char___returns_state(char: str, expected_state: DigitalState) -> None:
    assert DigitalState.from_char(char) == expected_state


@pytest.mark.parametrize("char", ["}", "?", "A"])
def test___invalid_char___from_char___raises_key_error(char: str) -> None:
    with pytest.raises(KeyError) as exc:
        _ = DigitalState.from_char(char)

    assert exc.value.args[0] == char


@pytest.mark.parametrize(
    "char1, char2, expected_result",
    [
        ("0", "0", False),
        ("1", "1", False),
        ("1", "0", True),
        ("1", "X", False),
        ("0", "X", False),
        ("H", "L", True),
    ],
)
def test___states___test___returns_pass_fail(char1: str, char2: str, expected_result: bool) -> None:
    state1 = DigitalState.from_char(char1)
    state2 = DigitalState.from_char(char2)

    assert DigitalState.test(state1, state2) == expected_result
