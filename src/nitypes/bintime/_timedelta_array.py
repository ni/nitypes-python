from __future__ import annotations

from collections.abc import Collection, MutableSequence
from typing import (
    final,
    overload,
)

import numpy as np
import numpy.typing as npt

from nitypes.bintime import CVITimeIntervalDType, TimeDelta, TimeValueTuple


@final
class TimeDeltaArray(MutableSequence[TimeDelta]):
    """An array of TimeDelta values in NI Binary Time Format (NI-BTF)."""

    __slots__ = ["_array"]

    _array: npt.NDArray[np.void]

    def __init__(
        self,
        value: Collection[TimeDelta] | None = None,
    ) -> None:
        """Initialize a new TimeDeltaArray."""
        if value is None:
            value = []
        self._array = np.fromiter(
            (entry.to_tuple().to_cvi() for entry in value),
            dtype=CVITimeIntervalDType,
            count=len(value),
        )

    @overload
    def __getitem__(  # noqa: D105 - missing docstring in magic method
        self, index: int
    ) -> TimeDelta: ...

    @overload
    def __getitem__(  # noqa: D105 - missing docstring in magic method
        self, index: slice
    ) -> MutableSequence[TimeDelta]: ...

    def __getitem__(self, index: int | slice) -> TimeDelta | MutableSequence[TimeDelta]:
        """Return the TimeDelta at the specified location."""
        if isinstance(index, int):
            entry = self._array[index].item()
            as_tuple = TimeValueTuple.from_cvi(*entry)
            return TimeDelta.from_tuple(as_tuple)
        elif isinstance(index, slice):
            raise NotImplementedError("TODO AB#3137071")
        else:
            raise TypeError("Index must be an int or slice")

    def __len__(self) -> int:
        """Return the length of the array."""
        return len(self._array)
