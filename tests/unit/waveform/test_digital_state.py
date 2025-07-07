from __future__ import annotations

import pytest

from nitypes.waveform import DigitalState


@pytest.mark.parametrize(
    "state, expected_char",
    [
        (DigitalState.FORCE_DOWN, "0"),
        (DigitalState.COMPARE_HIGH, "H"),
        (DigitalState.COMPARE_UNKNOWN, "X"),
        (DigitalState.EQUAL_1_H, "1H"),
        (DigitalState.NOT_EQUAL_0_1_L_H, "N01LH"),
    ],
)
def test___state___to_char___returns_char(state: DigitalState, expected_char: str) -> None:
    assert DigitalState.to_char(state) == expected_char


@pytest.mark.parametrize("state", [-1, 12, 255])
def test___invalid_state___to_char___raises_key_error(state: DigitalState) -> None:
    with pytest.raises(KeyError) as exc:
        _ = DigitalState.to_char(state)

    assert exc.value.args[0] == state


@pytest.mark.parametrize("state", [-1, 12, 255])
def test___invalid_state_errors_replace___to_char___returns_question_mark(
    state: DigitalState,
) -> None:
    assert DigitalState.to_char(state, errors="replace") == "?"


@pytest.mark.parametrize(
    "char, expected_state",
    [
        ("0", DigitalState.FORCE_DOWN),
        ("H", DigitalState.COMPARE_HIGH),
        ("X", DigitalState.COMPARE_UNKNOWN),
        ("1H", DigitalState.EQUAL_1_H),
        ("N01LH", DigitalState.NOT_EQUAL_0_1_L_H),
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
        ("0", "L", False),
        ("0", "H", True),
        ("1", "1", False),
        ("1", "H", False),
        ("1", "L", True),
        ("1", "0", True),
        ("1", "X", False),
        ("0", "X", False),
        ("H", "L", True),
        ("H", "1H", False),
        ("0", "0L", False),
        ("H", "0L", True),
        ("H", "N01", False),
        ("Z", "N01LH", False),
    ],
)
def test___states___test___returns_pass_fail(char1: str, char2: str, expected_result: bool) -> None:
    state1 = DigitalState.from_char(char1)
    state2 = DigitalState.from_char(char2)

    assert DigitalState.test(state1, state2) == expected_result
    assert DigitalState.test(state2, state1) == expected_result


@pytest.mark.parametrize(
    "state1, state2",
    [
        (DigitalState.FORCE_DOWN, -1),
        (12, DigitalState.COMPARE_UNKNOWN),
    ],
)
def test___invalid_state___test___raises_value_error(
    state1: DigitalState, state2: DigitalState
) -> None:
    with pytest.raises(ValueError) as exc:
        _ = DigitalState.test(state1, state2)

    assert "is not a valid DigitalState" in exc.value.args[0]
