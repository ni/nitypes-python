from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Generic, overload

from nitypes._exceptions import invalid_arg_type
from nitypes.waveform._digital._types import _TState

if TYPE_CHECKING:
    # Import from the public package so the docs don't reference private submodules.
    from nitypes.waveform import DigitalWaveform, DigitalWaveformSignal
else:
    # DigitalWaveform is a circular import.
    from nitypes.waveform._digital._signal import DigitalWaveformSignal


class DigitalWaveformSignalCollection(Generic[_TState], Sequence[DigitalWaveformSignal[_TState]]):
    """A collection of digital waveform signals.

    To construct this object, use the :any:`DigitalWaveform.signals` property.
    """

    __slots__ = ["_owner", "_signals", "__weakref__"]

    _owner: DigitalWaveform[_TState]
    _signals: list[DigitalWaveformSignal[_TState] | None]

    def __init__(self, owner: DigitalWaveform[_TState]) -> None:
        """Initialize a new DigitalWaveformSignalCollection."""
        self._owner = owner
        self._signals = [None] * owner.signal_count

    def __len__(self) -> int:
        """Return len(self)."""
        return len(self._signals)

    @overload
    def __getitem__(  # noqa: D105 - missing docstring in magic method
        self, index: int | str
    ) -> DigitalWaveformSignal[_TState]: ...
    @overload
    def __getitem__(  # noqa: D105 - missing docstring in magic method
        self, index: slice
    ) -> Sequence[DigitalWaveformSignal[_TState]]: ...

    def __getitem__(
        self, index: int | str | slice
    ) -> DigitalWaveformSignal[_TState] | Sequence[DigitalWaveformSignal[_TState]]:
        """Get self[index]."""
        if isinstance(index, int):
            if index < 0:
                index += len(self._signals)
            value = self._signals[index]
            if value is None:
                value = self._signals[index] = DigitalWaveformSignal(self._owner, index)
            return value
        elif isinstance(index, str):
            signal_names = self._owner._get_signal_names()
            try:
                signal_index = signal_names.index(index)
            except ValueError:
                raise IndexError(index)
            return self[signal_index]
        elif isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        else:
            raise invalid_arg_type("index", "int or str", index)
