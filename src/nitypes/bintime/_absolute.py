from __future__ import annotations

from typing import ClassVar, SupportsIndex

from nitypes._arguments import arg_to_int, arg_to_uint
from nitypes.bintime._timevalue import TimeValue


class AbsoluteTime:
    """An absolute time in NI Binary Time Format."""

    min: ClassVar[AbsoluteTime]
    max: ClassVar[AbsoluteTime]

    __slots__ = ["_value"]

    def __init__(
        self, whole_seconds: SupportsIndex = 0, fractional_second_ticks: SupportsIndex = 0
    ) -> None:
        """Initialize an AbsoluteTime.

        Args:
            whole_seconds: The number of whole seconds that have elapsed since midnight,
                January 1, 1904, UTC.
            fractional_second_ticks: The number of fractional second ticks (2^(-64) seconds) after
                the whole seconds that have elapsed since midnight, January 1, 1904, UTC. This
                value cannot be less than zero.

        .. warning::
            This constructor uses a different epoch (year 1904) than the constructors of .NET
            ``NationalInstruments.PrecisionDateTime`` class (year 0001). It is equivalent to the
            .NET ``PrecisionDateTime.FromLabViewTime`` method.
        """
        self._value = TimeValue(whole_seconds, fractional_second_ticks)
