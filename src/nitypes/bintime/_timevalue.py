from __future__ import annotations

import decimal
import math
import operator
from decimal import Decimal
from functools import singledispatchmethod
from typing import (
    Any,
    ClassVar,
    Protocol,
    SupportsIndex,
    Union,
    overload,
    runtime_checkable,
)

from typing_extensions import Self, TypeAlias

from nitypes._arguments import arg_to_int
from nitypes._exceptions import invalid_arg_type

_INT128_MAX = (1 << 127) - 1
_INT128_MIN = -(1 << 127)

_BITS_PER_SECOND = 64
_TICKS_PER_SECOND = 1 << _BITS_PER_SECOND
_FRACTIONAL_SECONDS_MASK = _TICKS_PER_SECOND - 1

_SECONDS_PER_DAY = 86400

_DECIMAL_DIGITS = 64
_REPR_TICKS = False


@runtime_checkable
class _SupportsTotalSeconds(Protocol):
    def total_seconds(self) -> float: ...


@runtime_checkable
class _SupportsPrecisionTotalSeconds(Protocol):
    def precision_total_seconds(self) -> Decimal: ...


class TimeValue:
    """A time value in NI Binary Time Format (NI-BTF).

    This class has a similar interface to :any:`datetime.timedelta` and :any:`hightime.timedelta`,
    implemented using a 128-bit fixed point number with a 64-bit whole seconds and 64-bit
    fractional seconds.
    """

    min: ClassVar[TimeValue]
    max: ClassVar[TimeValue]

    _TimeValueLike: TypeAlias = Union[
        "TimeValue", _SupportsTotalSeconds, _SupportsPrecisionTotalSeconds
    ]

    __slots__ = ["_ticks"]

    _ticks: int

    def __init__(
        self,
        seconds: (
            SupportsIndex
            | Decimal
            | float
            | _SupportsTotalSeconds
            | _SupportsPrecisionTotalSeconds
            | None
        ) = None,
    ) -> None:
        """Initialize a TimeValue."""
        if seconds is None:
            self._ticks = 0
        else:
            ticks = self.__class__._to_ticks(seconds)
            if not (_INT128_MIN <= ticks <= _INT128_MAX):
                raise OverflowError(
                    "The seconds value is out of range.\n\n"
                    f"Requested value: {seconds}\n"
                    f"Minimum value: {self.__class__.min.precision_total_seconds()}\n"
                    f"Maximum value: {self.__class__.max.precision_total_seconds()}"
                )
            self._ticks = ticks

    @singledispatchmethod
    @classmethod
    def _to_ticks(cls, seconds: object) -> int:
        raise invalid_arg_type("seconds", "number or timedelta", seconds)

    @_to_ticks.register
    @classmethod
    def _(cls, seconds: SupportsIndex) -> int:
        return operator.index(seconds) << _BITS_PER_SECOND

    @_to_ticks.register
    @classmethod
    def _(cls, seconds: Decimal) -> int:
        with decimal.localcontext() as ctx:
            ctx.prec = _DECIMAL_DIGITS
            whole_seconds, fractional_seconds = divmod(seconds, 1)
            ticks = int(whole_seconds) * _TICKS_PER_SECOND
            ticks += round(fractional_seconds * _TICKS_PER_SECOND)
            return ticks

    @_to_ticks.register
    @classmethod
    def _(cls, seconds: float) -> int:
        fractional_seconds, whole_seconds = math.modf(seconds)
        ticks = int(whole_seconds) * _TICKS_PER_SECOND
        ticks += round(fractional_seconds * _TICKS_PER_SECOND)
        return ticks

    @_to_ticks.register
    @classmethod
    def _(cls, seconds: _SupportsPrecisionTotalSeconds) -> int:
        return cls._to_ticks(seconds.precision_total_seconds())

    @_to_ticks.register
    @classmethod
    def _(cls, seconds: _SupportsTotalSeconds) -> int:
        return cls._to_ticks(seconds.total_seconds())

    @classmethod
    def from_ticks(cls, ticks: SupportsIndex) -> Self:
        """Create a TimeValue from a 128-bit fixed point number expressed as an integer."""
        ticks = arg_to_int("ticks", ticks)
        if not (_INT128_MIN <= ticks <= _INT128_MAX):
            raise OverflowError(
                "The ticks value is out of range.\n\n"
                f"Requested value: {ticks}\n"
                f"Minimum value: {_INT128_MIN}\n",
                f"Maximum value: {_INT128_MAX}",
            )
        self = cls()
        self._ticks = ticks
        return self

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
        """The total seconds in the time value.

        .. warning::
            Converting a time value to a floating point number loses precision.
        """
        seconds = float(self._ticks >> _BITS_PER_SECOND)
        seconds += float((self._ticks & _FRACTIONAL_SECONDS_MASK) / _TICKS_PER_SECOND)
        return seconds

    def precision_total_seconds(self) -> Decimal:
        """The precise total seconds in the time value.

        Note: up to 64 significant digits are used in computation.
        """
        with decimal.localcontext() as ctx:
            ctx.prec = _DECIMAL_DIGITS
            seconds = Decimal(self._ticks >> _BITS_PER_SECOND)
            seconds += Decimal(self._ticks & _FRACTIONAL_SECONDS_MASK) / Decimal(_TICKS_PER_SECOND)
            return seconds

    def __neg__(self) -> TimeValue:
        """Return -self."""
        return self.__class__.from_ticks(-self._ticks)

    def __pos__(self) -> TimeValue:
        """Return +self."""
        return self

    def __abs__(self) -> TimeValue:
        """Return abs(self)."""
        return -self if self._ticks < 0 else self

    def __add__(self, value: _TimeValueLike, /) -> TimeValue:
        """Return self+value."""
        if isinstance(value, TimeValue):
            return self.__class__.from_ticks(self._ticks + value._ticks)
        elif isinstance(value, (_SupportsTotalSeconds, _SupportsPrecisionTotalSeconds)):
            return self + TimeValue(value)
        else:
            return NotImplemented

    __radd__ = __add__

    def __sub__(self, value: _TimeValueLike, /) -> TimeValue:
        """Return self-value."""
        if isinstance(value, TimeValue):
            return self.__class__.from_ticks(self._ticks - value._ticks)
        elif isinstance(value, (_SupportsTotalSeconds, _SupportsPrecisionTotalSeconds)):
            return self - TimeValue(value)
        else:
            return NotImplemented

    def __rsub__(self, value: _TimeValueLike, /) -> TimeValue:
        """Return value-self."""
        if isinstance(value, TimeValue):
            return self.__class__.from_ticks(value._ticks - self._ticks)
        elif isinstance(value, (_SupportsTotalSeconds, _SupportsPrecisionTotalSeconds)):
            return TimeValue(value) - self
        else:
            return NotImplemented

    def __mul__(self, value: int | float | Decimal, /) -> TimeValue:
        """Return self*value."""
        if isinstance(value, int):
            return self.__class__.from_ticks(self._ticks * value)
        elif isinstance(value, float):
            # Using floating point math on 128-bit numbers loses precision, so use Decimal math.
            return self * Decimal(value)
        elif isinstance(value, Decimal):
            with decimal.localcontext() as ctx:
                ctx.prec = _DECIMAL_DIGITS
                return TimeValue(self.precision_total_seconds() * value)
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
            return self._ticks // value._ticks
        elif isinstance(value, int):
            return self.__class__.from_ticks(self._ticks // value)
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
            return TimeValue(self.total_seconds() / value)
        else:
            return NotImplemented

    def __mod__(self, value: _TimeValueLike, /) -> TimeValue:
        """Return self%value."""
        if isinstance(value, TimeValue):
            return self.__class__.from_ticks(self._ticks % value._ticks)
        elif isinstance(value, (_SupportsTotalSeconds, _SupportsPrecisionTotalSeconds)):
            return self % TimeValue(value)
        else:
            return NotImplemented

    def __divmod__(self, value: _TimeValueLike, /) -> tuple[int, TimeValue]:
        """Return (self//value, self%value)."""
        if isinstance(value, TimeValue):
            return (self // value, self % value)
        elif isinstance(value, (_SupportsTotalSeconds, _SupportsPrecisionTotalSeconds)):
            return divmod(self, TimeValue(value))
        else:
            return NotImplemented

    def __lt__(self, value: _TimeValueLike, /) -> bool:
        """Return self<value."""
        if isinstance(value, self.__class__):
            return self._ticks < value._ticks
        elif isinstance(value, (_SupportsTotalSeconds, _SupportsPrecisionTotalSeconds)):
            return self < TimeValue(value)
        else:
            return NotImplemented

    def __le__(self, value: _TimeValueLike, /) -> bool:
        """Return self<=value."""
        if isinstance(value, self.__class__):
            return self._ticks <= value._ticks
        elif isinstance(value, (_SupportsTotalSeconds, _SupportsPrecisionTotalSeconds)):
            return self <= TimeValue(value)
        else:
            return NotImplemented

    def __eq__(self, value: object, /) -> bool:
        """Return self==value."""
        if isinstance(value, self.__class__):
            return self._ticks == value._ticks
        elif isinstance(value, (_SupportsTotalSeconds, _SupportsPrecisionTotalSeconds)):
            return self == TimeValue(value)
        else:
            return NotImplemented

    def __gt__(self, value: _TimeValueLike, /) -> bool:
        """Return self<value."""
        if isinstance(value, self.__class__):
            return self._ticks > value._ticks
        elif isinstance(value, (_SupportsTotalSeconds, _SupportsPrecisionTotalSeconds)):
            return self > TimeValue(value)
        else:
            return NotImplemented

    def __ge__(self, value: _TimeValueLike, /) -> bool:
        """Return self>=value."""
        if isinstance(value, self.__class__):
            return self._ticks >= value._ticks
        elif isinstance(value, (_SupportsTotalSeconds, _SupportsPrecisionTotalSeconds)):
            return self >= TimeValue(value)
        else:
            return NotImplemented

    def __bool__(self) -> bool:
        """Return bool(self)."""
        return self._ticks != 0

    def __hash__(self) -> int:
        """Return hash(self)."""
        return hash(self._ticks)

    def __reduce__(self) -> tuple[Any, ...]:
        """Return object state for pickling."""
        return (self.__class__.from_ticks, (self._ticks,))

    def __str__(self) -> str:
        """Return repr(self)."""
        days = self.days
        seconds = self.seconds
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        _, fractional_seconds = divmod(self.precision_total_seconds(), 1)
        decimal_seconds = round(
            (Decimal(1) + fractional_seconds if fractional_seconds < 0 else fractional_seconds)
            * Decimal("1e18")
        )
        s = f"{days} day, " if abs(days) == 1 else f"{days} days, " if days else ""
        s += f"{hours}:{minutes:02}:{seconds:02}.{decimal_seconds:018}"
        return s

    def __repr__(self) -> str:
        """Return repr(self)."""
        if _REPR_TICKS:
            return (
                f"{self.__class__.__module__}.{self.__class__.__name__}.from_ticks({self._ticks})"
            )
        return f"{self.__class__.__module__}.{self.__class__.__name__}({self.precision_total_seconds()!r})"


TimeValue.max = TimeValue.from_ticks(_INT128_MAX)
TimeValue.min = TimeValue.from_ticks(_INT128_MIN)
