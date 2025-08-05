from __future__ import annotations

from collections.abc import Collection, Iterable, MutableSequence
from typing import (
    TYPE_CHECKING,
    Any,
    final,
    overload,
)

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    # Import from the public package so the docs don't reference private submodules.
    from nitypes.bintime import CVITimeIntervalDType, TimeDelta, TimeValueTuple
else:
    from nitypes.bintime._dtypes import CVITimeIntervalDType
    from nitypes.bintime._timedelta import TimeDelta
    from nitypes.bintime._time_value_tuple import TimeValueTuple


@final
class TimeDeltaArray(MutableSequence[TimeDelta]):
    """A mutable array of :class:`TimeDelta` values in NI Binary Time Format (NI-BTF).

    Raises:
        TypeError: If any item in value is not a TimeDelta instance.
    """

    __slots__ = ["_array"]

    _array: npt.NDArray[np.void]

    def __init__(
        self,
        value: Collection[TimeDelta] | None = None,
    ) -> None:
        """Initialize a new TimeDeltaArray."""
        if value is None:
            value = []
        if not all(isinstance(item, TimeDelta) for item in value):
            raise TypeError("Cannot assign values that are not of type TimeDelta")
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
    ) -> TimeDeltaArray: ...

    def __getitem__(self, index: int | slice) -> TimeDelta | TimeDeltaArray:
        """Return self[index].

        Raises:
            TypeError: If index is an invalid type.
            IndexError: If index is out of range.
        """
        if isinstance(index, int):
            entry = self._array[index].item()
            as_tuple = TimeValueTuple.from_cvi(*entry)
            return TimeDelta.from_tuple(as_tuple)
        elif isinstance(index, slice):
            sliced_entries = self._array[index]
            new_array = TimeDeltaArray()
            new_array._array = sliced_entries
            return new_array
        else:
            raise TypeError("Index must be an int or slice")

    def __len__(self) -> int:
        """Return len(self)."""
        return len(self._array)

    @overload
    def __setitem__(  # noqa: D105 - missing docstring in magic method
        self, index: int, value: TimeDelta
    ) -> None: ...

    @overload
    def __setitem__(  # noqa: D105 - missing docstring in magic method
        self, index: slice, value: Iterable[TimeDelta]
    ) -> None: ...

    def __setitem__(self, index: int | slice, value: TimeDelta | Iterable[TimeDelta]) -> None:
        """Set a new value for TimeDelta at the specified location or slice.

        Raises:
            TypeError: If index is an invalid type, or slice value is not iterable.
            ValueError: If slice assignment length doesn't match the selected range.
            IndexError: If index is out of range.
        """
        if isinstance(index, int):
            if not isinstance(value, TimeDelta):
                raise TypeError("Cannot assign value that is not of type TimeDelta")
            self._array[index] = value.to_tuple().to_cvi()
        elif isinstance(index, slice):
            if not isinstance(value, Iterable):
                raise TypeError("Cannot assign a slice with a non-iterable")
            if not all(isinstance(item, TimeDelta) for item in value):
                raise TypeError("Cannot assign values that are not of type TimeDelta")
            selected_count = len(range(*index.indices(len(self))))
            values = list(value)
            new_entry_count = len(values)
            if new_entry_count != selected_count:
                message = f"Cannot assign slice with unmatched length. Expected {selected_count} but received {new_entry_count}"
                raise ValueError(message)
            self._array[index] = [item.to_tuple().to_cvi() for item in values]
        else:
            raise TypeError("Index must be an int or slice")

    @overload
    def __delitem__(self, index: int) -> None: ...  # noqa: D105 - missing docstring in magic method

    @overload
    def __delitem__(  # noqa: D105 - missing docstring in magic method
        self, index: slice
    ) -> None: ...

    def __delitem__(self, index: int | slice) -> None:
        """Delete the value at the specified location or slice.

        Raises:
            TypeError: If index is an invalid type.
            IndexError: If index is out of range.
        """
        if isinstance(index, (int, slice)):
            self._array = np.delete(self._array, index)
        else:
            raise TypeError("Index must be an int or slice")

    def insert(self, index: int, value: TimeDelta) -> None:
        """Insert the TimeDelta value before the specified index.

        Raises:
            TypeError: If index is not int or value is not TimeDelta.
        """
        if not isinstance(index, int):
            raise TypeError("Index must be an int")
        if not isinstance(value, TimeDelta):
            raise TypeError("Cannot assign value that is not of type TimeDelta")
        lower = -len(self._array)
        upper = len(self._array)
        index = min(max(index, lower), upper)
        as_cvi = value.to_tuple().to_cvi()
        self._array = np.insert(self._array, index, as_cvi)

    def __eq__(self, other: object) -> bool:
        """Return self == other."""
        if not isinstance(other, TimeDeltaArray):
            return NotImplemented
        return np.array_equal(self._array, other._array)

    def __reduce__(self) -> tuple[Any, ...]:
        """Return object state for pickling."""
        return (self.__class__, (list(iter(self)),))

    def __repr__(self) -> str:
        """Return repr(self)."""
        ctor_args = list(iter(self))
        return f"{self.__class__.__module__}.{self.__class__.__name__}({ctor_args})"

    def __str__(self) -> str:
        """Return str(self)."""
        values = list(iter(self))
        return f"[{'; '.join(str(v) for v in values)}]"
