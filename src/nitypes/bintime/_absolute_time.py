from __future__ import annotations

import datetime as dt
from functools import singledispatchmethod
from typing import Any, ClassVar, SupportsIndex, Union, cast, final, overload

import hightime as ht
from typing_extensions import Self, TypeAlias

from nitypes._exceptions import invalid_arg_type, invalid_arg_value
from nitypes.bintime._time_value import (
    _OTHER_TIME_VALUE_TUPLE,
    TimeValue,
    _OtherTimeValue,
)

_DT_EPOCH_1904 = dt.datetime(1904, 1, 1, tzinfo=dt.timezone.utc)
_HT_EPOCH_1904 = ht.datetime(1904, 1, 1, tzinfo=dt.timezone.utc)

_OtherAbsoluteTime: TypeAlias = Union[dt.datetime, ht.datetime]
_OTHER_ABSOLUTE_TIME_TUPLE = (dt.datetime, ht.datetime)


@final
class AbsoluteTime:
    """An absolute time in NI Binary Time Format (NI-BTF).

    AbsoluteTime represents time as a 128-bit fixed point number with 64-bit whole seconds and
    64-bit fractional seconds.

    .. warning::
        The fractional seconds are represented as a binary fraction, which is a sum of inverse
        powers of 2. Values that are not exactly representable as binary fractions will display
        rounding error or "bruising" similar to a floating point number.

    AbsoluteTime instances are duck typing compatible with a subset of the method and properties
    supported by :any:`datetime.datetime` and :any:`hightime.datetime`.

    This class only supports the UTC time zone and does not support timezone-naive times.

    This class does not support the ``fold`` property for disambiguating repeated times for daylight
    saving time and time zone changes.
    """

    min: ClassVar[AbsoluteTime]
    max: ClassVar[AbsoluteTime]

    __slots__ = ["_offset"]

    _offset: TimeValue

    @overload
    def __init__(self) -> None: ...  # noqa: D107 - missing docstring in __init__

    @overload
    def __init__(  # noqa: D107 - missing docstring in __init__
        self, value: _OtherAbsoluteTime, /
    ) -> None: ...

    def __init__(
        self,
        year: SupportsIndex | _OtherAbsoluteTime | None = None,
    ) -> None:
        """Initialize an AbsoluteTime."""
        self._offset = self.__class__._to_offset(year)

    @singledispatchmethod
    @classmethod
    def _to_offset(cls, value: object) -> TimeValue:
        raise invalid_arg_type("value", "datetime", value)

    @_to_offset.register
    @classmethod
    def _(cls, value: ht.datetime) -> TimeValue:
        if value.tzinfo != dt.timezone.utc:
            raise invalid_arg_value("value.tzinfo", "datetime.timezone.utc", value.tzinfo)
        return TimeValue(value - _HT_EPOCH_1904)

    @_to_offset.register
    @classmethod
    def _(cls, value: dt.datetime) -> TimeValue:
        if value.tzinfo != dt.timezone.utc:
            raise invalid_arg_value("value.tzinfo", "datetime.timezone.utc", value.tzinfo)
        return TimeValue(value - _DT_EPOCH_1904)

    @_to_offset.register
    @classmethod
    def _(cls, value: None) -> TimeValue:
        return TimeValue()

    @classmethod
    def from_ticks(cls, ticks: SupportsIndex) -> Self:
        """Create an AbsoluteTime from a 128-bit fixed point number expressed as an integer."""
        self = cls.__new__(cls)
        self._offset = TimeValue.from_ticks(ticks)
        return self

    @classmethod
    def from_offset(cls, offset: TimeValue) -> Self:
        """Create an AbsoluteTime from a TimeValue offset from the epoch, Jan 1, 1904."""
        self = cls.__new__(cls)
        self._offset = offset
        return self

    def _to_datetime_datetime(self) -> dt.datetime:
        """Return self as a :any:`datetime.datetime`."""
        return _DT_EPOCH_1904 + self._offset._to_datetime_timedelta()

    def _to_hightime_datetime(self) -> ht.datetime:
        """Return self as a :any:`hightime.datetime`."""
        return _HT_EPOCH_1904 + self._offset._to_hightime_timedelta()

    # Calculating the year/month/day requires knowledge of leap years, days per month, etc., so
    # defer to hightime.datetime.
    @property
    def year(self) -> int:
        """The year."""
        return self._to_hightime_datetime().year

    @property
    def month(self) -> int:
        """The month, between 1 and 12 inclusive."""
        return self._to_hightime_datetime().month

    @property
    def day(self) -> int:
        """The day of the month, between 1 and 31 inclusive."""
        return self._to_hightime_datetime().day

    @property
    def hour(self) -> int:
        """The hour, between 0 and 23 inclusive."""
        return self._offset.seconds // 3600

    @property
    def minute(self) -> int:
        """The minute, between 0 and 59 inclusive."""
        return (self._offset.seconds // 60) % 60

    @property
    def second(self) -> int:
        """The second, between 0 and 59 inclusive."""
        return self._offset.seconds % 60

    @property
    def microsecond(self) -> int:
        """The microsecond, between 0 and 999_999 inclusive."""
        return self._offset.microseconds

    @property
    def femtosecond(self) -> int:
        """The femtosecond, between 0 and 999_999_999 inclusive."""
        return self._offset.femtoseconds

    @property
    def yoctosecond(self) -> int:
        """The yoctosecond, between 0 and 999_999_999 inclusive."""
        return self._offset.yoctoseconds

    @property
    def tzinfo(self) -> dt.tzinfo | None:
        """The time zone."""
        return dt.timezone.utc

    @classmethod
    def now(cls, tz: dt.tzinfo | None = None) -> Self:
        """Return the current absolute time."""
        if tz != dt.timezone.utc:
            raise invalid_arg_value("tz", "datetime.timezone.utc", tz)
        return cls(ht.datetime.now(tz))

    def __add__(self, value: TimeValue | _OtherTimeValue, /) -> AbsoluteTime:
        """Return self+value."""
        if isinstance(value, TimeValue):
            return self.__class__.from_offset(self._offset + value)
        elif isinstance(value, _OTHER_TIME_VALUE_TUPLE):
            return self + TimeValue(value)
        else:
            return NotImplemented

    __radd__ = __add__

    @overload
    def __sub__(  # noqa: D105 - missing docstring for magic method
        self, value: AbsoluteTime | _OtherAbsoluteTime, /
    ) -> TimeValue: ...
    @overload
    def __sub__(  # noqa: D105 - missing docstring for magic method
        self, value: TimeValue | _OtherTimeValue, /
    ) -> AbsoluteTime: ...

    def __sub__(
        self, value: AbsoluteTime | _OtherAbsoluteTime | TimeValue | _OtherTimeValue, /
    ) -> TimeValue | AbsoluteTime:
        """Return self-value."""
        if isinstance(value, AbsoluteTime):
            return self._offset - value._offset
        elif isinstance(value, _OTHER_ABSOLUTE_TIME_TUPLE):
            return self - self.__class__(value)
        elif isinstance(value, TimeValue):
            return self.__class__.from_offset(self._offset - value)
        elif isinstance(value, _OTHER_TIME_VALUE_TUPLE):
            return self - TimeValue(value)
        else:
            return NotImplemented

    def __lt__(self, value: AbsoluteTime | _OtherAbsoluteTime, /) -> bool:
        """Return self<value."""
        if isinstance(value, self.__class__):
            return self._offset < value._offset
        elif isinstance(value, _OTHER_ABSOLUTE_TIME_TUPLE):
            return self < self.__class__(value)
        else:
            return NotImplemented

    def __le__(self, value: AbsoluteTime | _OtherAbsoluteTime, /) -> bool:
        """Return self<=value."""
        if isinstance(value, self.__class__):
            return self._offset <= value._offset
        elif isinstance(value, _OTHER_ABSOLUTE_TIME_TUPLE):
            return self <= self.__class__(value)
        else:
            return NotImplemented

    def __eq__(self, value: object, /) -> bool:
        """Return self==value."""
        if isinstance(value, self.__class__):
            return self._offset == value._offset
        elif isinstance(value, _OTHER_ABSOLUTE_TIME_TUPLE):
            return self == self.__class__(value)
        else:
            return NotImplemented

    def __gt__(self, value: AbsoluteTime | _OtherAbsoluteTime, /) -> bool:
        """Return self<value."""
        if isinstance(value, self.__class__):
            return self._offset > value._offset
        elif isinstance(value, _OTHER_ABSOLUTE_TIME_TUPLE):
            return self > self.__class__(value)
        else:
            return NotImplemented

    def __ge__(self, value: AbsoluteTime | _OtherAbsoluteTime, /) -> bool:
        """Return self>=value."""
        if isinstance(value, self.__class__):
            return self._offset >= value._offset
        elif isinstance(value, _OTHER_ABSOLUTE_TIME_TUPLE):
            return self >= self.__class__(value)
        else:
            return NotImplemented

    def __hash__(self) -> int:
        """Return hash(self)."""
        return hash(self._offset)

    def __reduce__(self) -> tuple[Any, ...]:
        """Return object state for pickling."""
        return (self.__class__.from_ticks, (self._offset._ticks,))

    def __str__(self) -> str:
        """Return repr(self)."""
        return str(self._to_hightime_datetime())

    def __repr__(self) -> str:
        """Return repr(self)."""
        return f"{self.__class__.__module__}.{self.__class__.__name__}({self._to_hightime_datetime()!r})"


# These have to be within dt.datetime.max/min or else delegating to dt.datetime or ht.datetime for
# year/month/day, str(), repr(), etc. will fail. Use ticks to specify the maximum fractional second
# without rounding up to MAXYEAR+1.
AbsoluteTime.max = AbsoluteTime(
    dt.datetime(
        dt.MAXYEAR,
        12,
        31,
        23,
        59,
        59,
        tzinfo=dt.timezone.utc,
    )
) + TimeValue.from_ticks(0xFFFF_FFFF_FFFF_FFFF)
AbsoluteTime.min = AbsoluteTime(dt.datetime(dt.MINYEAR, 1, 1, tzinfo=dt.timezone.utc))
