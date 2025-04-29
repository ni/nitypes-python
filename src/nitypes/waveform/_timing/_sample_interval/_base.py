from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Generic, TypeVar, TYPE_CHECKING

from nitypes.waveform._timing._sample_interval._mode import SampleIntervalMode


if TYPE_CHECKING:
    from nitypes.waveform._timing._base import BaseTiming  # circular import

_TDateTime = TypeVar("_TDateTime", bound=dt.datetime)
_TTimeDelta = TypeVar("_TTimeDelta", bound=dt.timedelta)


class SampleIntervalStrategy(ABC, Generic[_TDateTime, _TTimeDelta]):
    """Implements SampleIntervalMode specific behavior."""

    # Note that timing is always passed as a parameter. The timing object has a reference to the
    # strategy, so saving a reference to the timing object would introduce a reference cycle.
    __slots__ = ()

    @abstractmethod
    def validate_init_args(
        self,
        timing: BaseTiming[_TDateTime, _TTimeDelta],
        sample_interval_mode: SampleIntervalMode,
        timestamp: _TDateTime | None,
        time_offset: _TTimeDelta | None,
        sample_interval: _TTimeDelta | None,
        timestamps: Sequence[_TDateTime] | None,
    ) -> None:
        """Validate the BaseTiming.__init__ arguments for this mode."""
        raise NotImplementedError

    @abstractmethod
    def get_timestamps(
        self, timing: BaseTiming[_TDateTime, _TTimeDelta], start_index: int, count: int
    ) -> Iterable[_TDateTime]:
        """Get or generate timestamps for the specified samples."""
        raise NotImplementedError

    @abstractmethod
    def append_timestamps(
        self,
        timing: BaseTiming[_TDateTime, _TTimeDelta],
        timestamps: Sequence[_TDateTime] | None,
    ) -> BaseTiming[_TDateTime, _TTimeDelta]:
        """Append timestamps and return a new waveform timing if needed."""
        raise NotImplementedError

    @abstractmethod
    def append_timing(
        self,
        timing: BaseTiming[_TDateTime, _TTimeDelta],
        other: BaseTiming[_TDateTime, _TTimeDelta],
    ) -> BaseTiming[_TDateTime, _TTimeDelta]:
        """Append timing and return a new waveform timing if needed."""
        raise NotImplementedError
