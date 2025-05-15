from __future__ import annotations

from typing import Any, ClassVar, SupportsIndex

from nitypes._arguments import arg_to_int, arg_to_uint

_INT64_MAX = (1 << 63) - 1
_INT64_MIN = -(1 << 63)

_UINT64_RANGE = 1 << 64
_UINT64_MAX = (1 << 64) - 1
_UINT64_MIN = 0


class TimeValue:
    """A time value in NI Binary Time Format (NI-BTF)."""

    min: ClassVar[TimeValue]
    max: ClassVar[TimeValue]

    __slots__ = ["_msb", "_lsb"]

    def __init__(
        self, whole_seconds: SupportsIndex = 0, fractional_second_ticks: SupportsIndex = 0
    ) -> None:
        """Initialize a TimeValue.

        Args:
            whole_seconds: The number of whole seconds.
            fractional_second_ticks: The number of fractional second ticks (2^(-64) seconds). This
                value cannot be less than zero.
        """
        self._msb = arg_to_int("whole seconds", whole_seconds)
        if not (_INT64_MIN <= self._msb <= _INT64_MAX):
            raise ValueError("The whole seconds must be a 64-bit signed integer.")

        self._lsb = arg_to_uint("fraction second ticks", fractional_second_ticks)
        if not (_UINT64_MIN <= self._lsb <= _UINT64_MAX):
            raise ValueError("The fractional second ticks must be a 64-bit unsigned integer.")

    ############
    # Comparison
    ############
    def __lt__(self, value: TimeValue, /) -> bool:
        """Return self<value."""
        if not isinstance(value, self.__class__):
            return NotImplemented
        return (self._msb, self._lsb) < (value._msb, value._lsb)

    def __le__(self, value: TimeValue, /) -> bool:
        """Return self<=value."""
        if not isinstance(value, self.__class__):
            return NotImplemented
        return (self._msb, self._lsb) <= (value._msb, value._lsb)

    def __eq__(self, value: object, /) -> bool:
        """Return self==value."""
        if not isinstance(value, self.__class__):
            return NotImplemented
        return (self._msb, self._lsb) == (value._msb, value._lsb)

    def __ne__(self, value: object, /) -> bool:
        """Return self!=value."""
        if not isinstance(value, self.__class__):
            return NotImplemented
        return (self._msb, self._lsb) != (value._msb, value._lsb)

    def __gt__(self, value: TimeValue, /) -> bool:
        """Return self<value."""
        if not isinstance(value, self.__class__):
            return NotImplemented
        return (self._msb, self._lsb) > (value._msb, value._lsb)

    def __ge__(self, value: TimeValue, /) -> bool:
        """Return self>=value."""
        if not isinstance(value, self.__class__):
            return NotImplemented
        return (self._msb, self._lsb) >= (value._msb, value._lsb)

    ###################
    # Binary arithmetic
    ###################
    def __add__(self, value: TimeValue, /) -> TimeValue:
        """Return self+value."""
        carry, lsb = divmod(self._lsb + value._lsb, _UINT64_MAX)
        msb = self._msb + value._msb + carry
        if not (_INT64_MIN <= msb <= _INT64_MAX):
            raise OverflowError
        return TimeValue(msb, lsb)

    def __sub__(self, value: TimeValue, /) -> TimeValue:
        """Return self-value."""
        return self + -value

    ##################
    # Unary arithmetic
    ##################
    def __neg__(self) -> TimeValue:
        """Return -self."""
        if self._lsb == 0:
            return TimeValue(-self._msb, 0)
        else:
            return TimeValue(-self._msb - 1, _UINT64_RANGE - self._lsb)

    def __pos__(self) -> TimeValue:
        """Return +self."""
        return self

    def __abs__(self) -> TimeValue:
        """Return abs(self)."""
        return -self if self._msb < 0 else self

    def __invert__(self) -> TimeValue:
        """Return ~self."""
        return TimeValue(~self._msb, ~self._lsb)

    ############
    # Conversion
    ############
    def __bool__(self) -> bool:
        """Return bool(self)."""
        return self._msb != 0 or self._lsb != 0

    def __int__(self) -> int:
        """Return int(self)."""
        return self._msb

    def __float__(self) -> float:
        """Return float(self)."""
        return self._msb + self._lsb / _UINT64_RANGE

    ###############
    # Miscellaneous
    ###############
    def __hash__(self) -> int:
        """Return hash(self)."""
        return hash((self._msb, self._lsb))

    def __reduce__(self) -> tuple[Any, ...]:
        """Return object state for pickling."""
        return (self.__class__, (self._msb, self._lsb))

    def __str__(self) -> str:
        """Return repr(self)."""
        return f"{self.__class__.__name__}({self._msb}, {self._lsb})"

    def __repr__(self) -> str:
        """Return repr(self)."""
        return f"{self.__class__.__module__}.{self.__class__.__name__}({self._msb}, {self._lsb})"


TimeValue.max = TimeValue(_INT64_MAX, _UINT64_MAX)
TimeValue.min = TimeValue(_INT64_MIN, _UINT64_MIN)
