from __future__ import annotations

from enum import IntEnum

from nitypes._exceptions import invalid_arg_value

_CHAR_TABLE = ["0", "1", "Z", "L", "H", "X", "T", "V", "0L", "1H", "N01", "N01LH"]

_STATE_TEST_TABLE = [
    # 0  1  Z  L  H  X  T  V 0L 1H N01 N01LH
    [1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0],  # 0
    [0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0],  # 1
    [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1],  # Z
    [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],  # L
    [0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0],  # H
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # X
    [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # T
    [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # V
    [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],  # 0L
    [0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0],  # 1H
    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0],  # N01
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],  # N01LH
]


class DigitalState(IntEnum):
    """An IntEnum of the different digital states that a digital signal can represent.

    You can use :any:`DigitalState` in place of an :any:`int`:

    >>> DigitalState.FORCE_OFF
    <DigitalState.FORCE_OFF: 2>
    >>> DigitalState.FORCE_OFF == 2
    True

    Use :any:`from_char` and :any:`to_char` to convert between states and characters:

    >>> DigitalState.from_char("Z")
    <DigitalState.FORCE_OFF: 2>
    >>> DigitalState.to_char(2)
    'Z'

    Use :any:`test` to compare actual vs. expected states, returning True on failure.

    >>> DigitalState.test(DigitalState.FORCE_DOWN, DigitalState.COMPARE_LOW)
    False
    >>> DigitalState.test(DigitalState.FORCE_UP, DigitalState.COMPARE_LOW)
    True
    """

    _value_: int

    FORCE_DOWN = 0
    """Force logic low (``0``). Drive to the low voltage level (VIL)."""

    FORCE_UP = 1
    """Force logic high (``1``). Drive to the high voltage level (VIH)."""

    FORCE_OFF = 2
    """Force logic high impedance (``Z``). Turn the driver off."""

    COMPARE_LOW = 3
    """Compare logic low (edge) (``L``). Compare for a voltage level lower than the low voltage
    threshold (VOL)."""

    COMPARE_HIGH = 4
    """Compare logic high (edge) (``H``). Compare for a voltage level higher than the high voltage
    threshold (VOH)."""

    COMPARE_UNKNOWN = 5
    """Compare logic unknown (``X``). Don't compare."""

    COMPARE_OFF = 6
    """Compare logic high impedance (edge) (``T``). Compare for a voltage level between the low
    voltage threshold (VOL) and the high voltage threshold (VOH)."""

    COMPARE_VALID = 7
    """Compare logic valid level (edge) (``V``). Compare for a voltage level either lower than the
    low voltage threshold (VOL) or higher than the high voltage threshold (VOH)."""

    EQUAL_0_L = 8
    """Compare logic equal to 0 or Low (edge) (``0L``). Compare for a voltage level equal to 0 or
    Low."""

    EQUAL_1_H = 9
    """Compare logic equal to 1 or High (edge) (``1H``). Compare for a voltage level equal to 1 or
    High."""

    NOT_EQUAL_0_1 = 10
    """Compare logic not equal to 0 or 1 (edge) (``N01``). Compare for a voltage level not equal
    to 0 or 1."""

    NOT_EQUAL_0_1_L_H = 11
    """Compare logic not equal to 0, 1, Low, or High (edge) (``N01LH``). Compare for a voltage
    level not equal to 0, 1, Low, or High."""

    @property
    def char(self) -> str:
        """The character representing the digital state."""
        return _CHAR_TABLE[self]

    @classmethod
    def from_char(cls, char: str) -> DigitalState:
        """Look up the digital state for the corresponding character."""
        try:
            return DigitalState(_CHAR_TABLE.index(char))
        except ValueError:
            raise KeyError(char)

    @classmethod
    def to_char(cls, state: DigitalState, errors: str = "strict") -> str:
        """Get a character representing the digital state.

        Args:
            state: The digital state.
            errors: Specifies how to handle errors.

                * "strict": raise ``KeyError``
                * "replace": return "?"

        Returns:
            A character representing the digital state.
        """
        if errors not in ("strict", "replace"):
            raise invalid_arg_value("errors argument", "supported value", errors)
        try:
            return DigitalState(state).char
        except ValueError:
            if errors == "strict":
                raise KeyError(state)
            elif errors == "replace":
                return "?"
            raise

    @staticmethod
    def test(state1: DigitalState, state2: DigitalState) -> bool:
        """Test two digital states and return True if the test failed."""
        return not _STATE_TEST_TABLE[DigitalState(state1)][DigitalState(state2)]
