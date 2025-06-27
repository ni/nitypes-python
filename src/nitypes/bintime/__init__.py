"""Binary time data types for NI Python APIs."""

from __future__ import annotations

from nitypes.bintime._datetime import DateTime
from nitypes.bintime._dtypes import (
    CVIAbsoluteTimeBase,
    CVIAbsoluteTimeDType,
    CVITimeIntervalBase,
    CVITimeIntervalDType,
)
from nitypes.bintime._time_value_tuple import TimeValueTuple
from nitypes.bintime._timedelta import TimeDelta

__all__ = [
    "DateTime",
    "CVIAbsoluteTimeBase",
    "CVIAbsoluteTimeDType",
    "CVITimeIntervalBase",
    "CVITimeIntervalDType",
    "TimeDelta",
    "TimeValueTuple",
]

# Hide that it was defined in a helper file
DateTime.__module__ = __name__
TimeDelta.__module__ = __name__
TimeValueTuple.__module__ = __name__
