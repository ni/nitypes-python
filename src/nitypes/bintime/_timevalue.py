from __future__ import annotations

import decimal
import math
from decimal import Decimal
from typing import Any, ClassVar, SupportsFloat, SupportsIndex, overload

from nitypes._arguments import arg_to_int, validate_unsupported_arg
from nitypes._exceptions import add_note, invalid_arg_type

_INT128_MAX = (1 << 127) - 1
_INT128_MIN = -(1 << 127)

_BITS_PER_SECOND = 64
_TICKS_PER_SECOND = 1 << _BITS_PER_SECOND
_FRACTIONAL_SECONDS_MASK = _TICKS_PER_SECOND - 1

_SECONDS_PER_DAY = 86400

_MICROSECONDS_PER_SECOND = 1_000_000
_NANOSECONDS_PER_SECOND = 1_000_000_000
_PICOSECONDS_PER_SECOND = 1_000_000_000_000
_FEMTOSECONDS_PER_SECOND = 1_000_000_000_000_000
_ATTOSECONDS_PER_SECOND = 1_000_000_000_000_000_000


class TimeValue:
    """A time value in NI Binary Time Format (NI-BTF).

    This class has a similar interface to :any:`datetime.timedelta` and :any:`hightime.timedelta`,
    implemented using a 128-bit fixed point number with a 64-bit whole seconds and 64-bit
    fractional seconds.
    """

    min: ClassVar[TimeValue]
    max: ClassVar[TimeValue]

    __slots__ = ["_ticks"]

    _ticks: int

    @overload
    def __init__(  # noqa: D107 - missing docstring in __init__
        self,
        seconds: SupportsIndex | None = ...,
        ticks: SupportsIndex | None = ...,
    ) -> None: ...

    @overload
    def __init__(  # noqa: D107 - missing docstring in __init__
        self,
        seconds: SupportsFloat | Decimal,
    ) -> None: ...

    def __init__(
        self,
        seconds: SupportsIndex | SupportsFloat | None = None,
        ticks: SupportsIndex | None = None,
    ) -> None:
        """Initialize a TimeValue."""
        self._ticks = 0
        try:
            if seconds is not None:
                if isinstance(seconds, SupportsIndex):
                    self._ticks += arg_to_int("seconds", seconds) * _TICKS_PER_SECOND
                elif isinstance(seconds, Decimal):
                    validate_unsupported_arg("ticks", ticks)
                    whole_seconds, fractional_seconds = divmod(seconds, 1)
                    self._ticks += int(whole_seconds) * _TICKS_PER_SECOND
                    self._ticks += int(fractional_seconds * _TICKS_PER_SECOND)
                elif isinstance(seconds, SupportsFloat):
                    validate_unsupported_arg("ticks", ticks)
                    fractional_seconds, whole_seconds = math.modf(seconds)
                    self._ticks += int(whole_seconds) * _TICKS_PER_SECOND
                    self._ticks += int(fractional_seconds * _TICKS_PER_SECOND)
                else:
                    raise invalid_arg_type("seconds", "integer or float", seconds)
            if ticks is not None:
                self._ticks += arg_to_int("ticks", ticks)
            if not (_INT128_MIN <= self._ticks <= _INT128_MAX):
                raise OverflowError("The time value is out of range.")
        except Exception as e:
            if seconds is not None:
                add_note(e, f"Requested seconds: {seconds}")
            if ticks is not None:
                add_note(e, f"Requested ticks: {ticks}")
            add_note(e, f"Calculated ticks: {self._ticks}")
            raise

    @property
    def days(self) -> int:
        """The number of days in the time value."""
        return (self._ticks >> _BITS_PER_SECOND) // _SECONDS_PER_DAY

    @property
    def seconds(self) -> int:
        """The number of seconds in the time value, up to the nearest day."""
        return (self._ticks >> _BITS_PER_SECOND) % _SECONDS_PER_DAY

    @property
    def microseconds(self) -> int:
        """The number of microseconds in the time value, up to the nearest second."""
        return (1000 * self._ticks) >> _BITS_PER_SECOND

    def total_seconds(self) -> float:
        """The total seconds in the time value."""
        return (self._ticks >> _BITS_PER_SECOND) + (
            (self._ticks & _FRACTIONAL_SECONDS_MASK) / _TICKS_PER_SECOND
        )

    def precision_total_seconds(self) -> Decimal:
        """The precise total seconds in the time value.

        Note: up to 64 significant digits are used in computation.
        """
        with decimal.localcontext() as ctx:
            ctx.prec = 64
            return Decimal(self._ticks >> _BITS_PER_SECOND) + Decimal(
                self._ticks & _FRACTIONAL_SECONDS_MASK
            ) / Decimal(_TICKS_PER_SECOND)

    def __add__(self, value: TimeValue, /) -> TimeValue:
        """Return self+value."""
        if not isinstance(value, TimeValue):
            return NotImplemented
        return TimeValue(ticks=self._ticks + value._ticks)

    __radd__ = __add__

    def __sub__(self, value: TimeValue, /) -> TimeValue:
        """Return self-value."""
        if not isinstance(value, TimeValue):
            return NotImplemented
        return TimeValue(ticks=self._ticks - value._ticks)

    def __rsub__(self, value: TimeValue, /) -> TimeValue:
        """Return value-self."""
        if not isinstance(value, TimeValue):
            return NotImplemented
        return TimeValue(ticks=value._ticks - self._ticks)

    def __neg__(self) -> TimeValue:
        """Return -self."""
        return TimeValue(ticks=-self._ticks)

    def __pos__(self) -> TimeValue:
        """Return +self."""
        return self

    def __abs__(self) -> TimeValue:
        """Return abs(self)."""
        return -self if self._ticks < 0 else self

    def __mul__(self, value: float, /) -> TimeValue:
        """Return self*value."""
        if isinstance(value, int):
            return TimeValue(ticks=self._ticks * value)
        elif isinstance(value, float):
            # Using floating point math on 128-bit numbers loses precision, so use Decimal math.
            return TimeValue(self.precision_total_seconds() * Decimal(value))
        else:
            return NotImplemented

    __rmul__ = __mul__

    @overload
    def __floordiv__(  # noqa: D105 - missing docstring in magic method
        self, value: TimeValue, /
    ) -> int: ...
    @overload
    def __floordiv__(  # noqa: D105 - missing docstring in magic method
        self, value: int, /
    ) -> TimeValue: ...

    def __floordiv__(self, value: TimeValue | int, /) -> int | TimeValue:
        """Return self//value."""
        if isinstance(value, TimeValue):
            return TimeValue(ticks=self._ticks // value._ticks)
        elif isinstance(value, int):
            return TimeValue(ticks=self._ticks // value)
        else:
            return NotImplemented

    @overload
    def __truediv__(  # noqa: D105 - missing docstring in magic method
        self, value: TimeValue, /
    ) -> float: ...
    @overload
    def __truediv__(  # noqa: D105 - missing docstring in magic method
        self, value: float, /
    ) -> TimeValue: ...

    def __truediv__(self, value: TimeValue | float, /) -> float | TimeValue:
        """Return self/value."""
        if isinstance(value, TimeValue):
            return self.total_seconds() / value.total_seconds()
        elif isinstance(value, float):
            return self.total_seconds() / value
        else:
            return NotImplemented

    def __mod__(self, value: TimeValue, /) -> TimeValue:
        """Return self%value."""
        if not isinstance(value, TimeValue):
            return NotImplemented
        return TimeValue(ticks=self._ticks % value._ticks)

    def __divmod__(self, value: TimeValue, /) -> tuple[int, TimeValue]:
        """Return (self//value, self%value)."""
        if not isinstance(value, TimeValue):
            return NotImplemented
        return (self // value, self % value)

    def __lt__(self, value: TimeValue, /) -> bool:
        """Return self<value."""
        if not isinstance(value, self.__class__):
            return NotImplemented
        return self._ticks < value._ticks

    def __le__(self, value: TimeValue, /) -> bool:
        """Return self<=value."""
        if not isinstance(value, self.__class__):
            return NotImplemented
        return self._ticks <= value._ticks

    def __eq__(self, value: object, /) -> bool:
        """Return self==value."""
        if not isinstance(value, self.__class__):
            return NotImplemented
        return self._ticks == value._ticks

    def __ne__(self, value: object, /) -> bool:
        """Return self==value."""
        if not isinstance(value, self.__class__):
            return NotImplemented
        return self._ticks != value._ticks

    def __gt__(self, value: TimeValue, /) -> bool:
        """Return self<value."""
        if not isinstance(value, self.__class__):
            return NotImplemented
        return self._ticks > value._ticks

    def __ge__(self, value: TimeValue, /) -> bool:
        """Return self>=value."""
        if not isinstance(value, self.__class__):
            return NotImplemented
        return self._ticks >= value._ticks

    def __bool__(self) -> bool:
        """Return bool(self)."""
        return self._ticks != 0

    def __hash__(self) -> int:
        """Return hash(self)."""
        return hash(self._ticks)

    def __reduce__(self) -> tuple[Any, ...]:
        """Return object state for pickling."""
        return (self.__class__, (None, self._ticks))

    def __str__(self) -> str:
        """Return repr(self)."""
        days = self.days
        seconds = self.seconds
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        attoseconds = (self._ticks * _ATTOSECONDS_PER_SECOND) >> _BITS_PER_SECOND
        s = f"{days} days, " if days else ""
        s += f"{hours}:{minutes:02}:{seconds:02}.{attoseconds:018}"
        return s

    def __repr__(self) -> str:
        """Return repr(self)."""
        return f"{self.__class__.__module__}.{self.__class__.__name__}(ticks={self._ticks})"


TimeValue.max = TimeValue(ticks=_INT128_MAX)
TimeValue.min = TimeValue(ticks=_INT128_MIN)
