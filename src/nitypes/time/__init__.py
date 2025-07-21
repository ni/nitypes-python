"""Time data types for NI Python APIs."""

from nitypes.time._conversion import convert_datetime, convert_timedelta
from nitypes.time._types import AnyDateTime, AnyTimeDelta, TDateTime, TTimeDelta

__all__ = [
    "AnyDateTime",
    "AnyTimeDelta",
    "convert_datetime",
    "convert_timedelta",
    "TDateTime",
    "TTimeDelta",
]
