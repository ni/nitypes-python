from __future__ import annotations

from enum import IntEnum


class DigitalState(IntEnum):
    """An IntEnum of the different digital states that a digital signal can represent.

    You can use :any:`DigitalState` in place of an :any:`int`:

    >>> DigitalState.FORCE_OFF
    <DigitalState.FORCE_OFF: 2>
    >>> DigitalState.FORCE_OFF == 2
    True

    Or you can use its :any:`value` and :any:`pattern` properties:

    >>> DigitalState.FORCE_OFF.value
    2
    >>> DigitalState.FORCE_OFF.pattern
    'Z'

    You can also use :any:`from_pattern` to look up the digital state for a given pattern:

    >>> DigitalState.from_pattern("Z")
    <DigitalState.FORCE_OFF: 2>
    """

    _value_: int
    pattern: str

    def __new__(cls, value: int, pattern: str) -> DigitalState:
        """Construct a new digital state."""
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.pattern = pattern
        return obj

    @classmethod
    def from_pattern(cls, pattern: str) -> DigitalState:
        """Look up the digital state for a digital pattern."""
        obj = next((obj for obj in cls if obj.pattern == pattern), None)
        if obj is None:
            raise KeyError(pattern)
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
