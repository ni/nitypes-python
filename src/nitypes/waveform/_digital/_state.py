from __future__ import annotations

from enum import IntEnum

_STATE_TEST_TABLE = [
    # 0  1  Z  L  H  X  T  V
    [1, 0, 0, 1, 0, 1, 0, 1],  # 0
    [0, 1, 0, 0, 1, 1, 0, 1],  # 1
    [0, 0, 1, 0, 0, 1, 1, 0],  # Z
    [1, 0, 0, 1, 0, 1, 0, 0],  # L
    [0, 1, 0, 0, 1, 1, 0, 0],  # H
    [1, 1, 1, 1, 1, 1, 1, 1],  # X
    [0, 0, 1, 0, 0, 1, 1, 0],  # T
    [1, 1, 0, 0, 0, 1, 0, 1],  # V
]


class DigitalState(IntEnum):
    """An IntEnum of the different digital states that a digital signal can represent.

    You can use :any:`DigitalState` in place of an :any:`int`:

    >>> DigitalState.FORCE_OFF
    <DigitalState.FORCE_OFF: 2>
    >>> DigitalState.FORCE_OFF == 2
    True

    Or you can use its :any:`value` and :any:`char` properties:

    >>> DigitalState.FORCE_OFF.value
    2
    >>> DigitalState.FORCE_OFF.char
    'Z'

    You can also use :any:`from_char` and :any:`to_char` to convert between states and characters:

    >>> DigitalState.from_char("Z")
    <DigitalState.FORCE_OFF: 2>
    >>> DigitalState.to_char(2)
    'Z'
    """

    _value_: int
    char: str

    def __new__(cls, value: int, pattern: str) -> DigitalState:
        """Construct a new digital state."""
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.char = pattern
        return obj

    FORCE_DOWN = (0, "0")
    """Force logic low. Drive to the low voltage level (VIL)."""

    FORCE_UP = (1, "1")
    """Force logic high. Drive to the high voltage level (VIH)."""

    FORCE_OFF = (2, "Z")
    """Force logic high impedance. Turn the driver off."""

    COMPARE_LOW = (3, "L")
    """Compare logic low (edge). Compare for a voltage level lower than the low voltage threshold
    (VOL)."""

    COMPARE_HIGH = (4, "H")
    """Compare logic high (edge). Compare for a voltage level higher than the high voltage threshold
    (VOH)."""

    COMPARE_UNKNOWN = (5, "X")
    """Compare logic unknown. Don't compare."""

    COMPARE_OFF = (6, "T")
    """Compare logic high impedance (edge). Compare for a voltage level between the low voltage
    threshold (VOL) and the high voltage threshold (VOH)."""

    COMPARE_VALID = (7, "V")
    """Compare logic valid level (edge). Compare for a voltage level either lower than the low
    voltage threshold (VOL) or higher than the high voltage threshold (VOH)."""

    @classmethod
    def from_char(cls, char: str) -> DigitalState:
        """Look up the digital state for the corresponding character."""
        obj = next((obj for obj in cls if obj.char == char), None)
        if obj is None:
            raise KeyError(char)
        return obj

    @classmethod
    def to_char(cls, state: DigitalState) -> str:
        """Get a character representing the digital state."""
        try:
            return DigitalState(state).char
        except ValueError:
            return "?"

    @staticmethod
    def test(state1: DigitalState, state2: DigitalState) -> bool:
        """Test two digital states and return True if the test failed."""
        return not _STATE_TEST_TABLE[state1][state2]
