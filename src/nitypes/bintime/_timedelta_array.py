from __future__ import annotations

from collections.abc import Sequence
from typing import (
    Union,
    final,
    overload,
)

import numpy as np
import numpy.typing as npt

from nitypes.bintime import CVITimeIntervalDType, TimeDelta, TimeValueTuple


@final
class TimeDeltaArray(Sequence[TimeDelta]):
    """An array of TimeDelta values in NI Binary Time Format (NI-BTF)."""

    __slots__ = ["_array"]

    _array: npt.NDArray[np.void]

    def __init__(
        self,
        value: Union[Sequence[TimeDelta], None] = None,
    ) -> None:
        """Initialize a new TimeDeltaArray."""
        if value is None:
            value = []
        self._array = np.zeros(len(value), dtype=CVITimeIntervalDType)
        for index, entry in enumerate(value):
            as_tuple = entry.to_tuple()
            self._array[index] = (as_tuple.fractional_seconds, as_tuple.whole_seconds)

    @overload
    def __getitem__(  # noqa: D105 - missing docstring in magic method
        self, index: int
    ) -> TimeDelta: ...

    @overload
    def __getitem__(  # noqa: D105 - missing docstring in magic method
        self, index: slice
    ) -> Sequence[TimeDelta]: ...

    def __getitem__(self, index: Union[int, slice]) -> Union[TimeDelta, Sequence[TimeDelta]]:
        """Return the TimeDelta at the specified location."""
        if isinstance(index, int):
            entry = self._array[index]
            as_tuple = TimeValueTuple(entry["msb"], entry["lsb"])
            return TimeDelta.from_tuple(as_tuple)
        elif isinstance(index, slice):
            raise NotImplementedError("TODO AB#3137071")
        else:
            raise TypeError("Index must be an int or slice")

    def __len__(self) -> int:
        """Return the length of the array."""
        return np.size(self._array)
