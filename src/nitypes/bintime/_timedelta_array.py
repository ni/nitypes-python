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
    """An array of :class:`TimeDelta` values in NI Binary Time Format (NI-BTF).

    The TimeDeltaArray class provides a mutable sequence container for storing
    multiple :class:`TimeDelta` objects efficiently using the NI Binary Time Format.
    NI-BTF is a high-resolution time format consisting of a 128-bit fixed point
    number with 64-bit whole seconds and 64-bit fractional seconds parts.

    This class implements the MutableSequence interface, providing familiar
    list-like operations for time delta manipulation. The underlying storage
    uses NumPy arrays with :data:`CVITimeIntervalDType` for optimal performance.

    :param value: Initial collection of TimeDelta objects, defaults to empty array
    :type value: Collection[TimeDelta] | None, optional

    :raises TypeError: If any item in value is not a TimeDelta instance

    Examples:
        Create an empty array:

        >>> arr = TimeDeltaArray()
        >>> len(arr)
        0

        Create with initial values:

        >>> from nitypes.bintime import TimeDelta
        >>> td1 = TimeDelta(seconds=1.5)
        >>> td2 = TimeDelta(seconds=2.5)
        >>> arr = TimeDeltaArray([td1, td2])
        >>> len(arr)
        2

        Access and modify elements:

        >>> arr[0]
        TimeDelta(seconds=1.5)
        >>> arr[0] = TimeDelta(seconds=3.0)
        >>> arr[0]
        TimeDelta(seconds=3.0)

    See Also:
        :class:`TimeDelta`: Individual time delta values
        :data:`CVITimeIntervalDType`: Underlying NumPy data type
        :class:`TimeValueTuple`: Time value representation
    """

    __slots__ = ["_array"]

    _array: npt.NDArray[np.void]

    def __init__(
        self,
        value: Collection[TimeDelta] | None = None,
    ) -> None:
        """Initialize a new TimeDeltaArray.

        :param value: Initial collection of TimeDelta objects
        :type value: Collection[TimeDelta] | None, optional

        :raises TypeError: If any item in value is not a TimeDelta instance
        """
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
    def __getitem__(self, index: int) -> TimeDelta:
        """Get a single TimeDelta by index.

        :param index: Array index
        :type index: int
        :rtype: TimeDelta
        """
        ...

    @overload
    def __getitem__(self, index: slice) -> TimeDeltaArray:
        """Get a slice of TimeDelta objects as a new TimeDeltaArray.

        :param index: Slice object
        :type index: slice
        :rtype: TimeDeltaArray
        """
        ...

    def __getitem__(self, index: int | slice) -> TimeDelta | TimeDeltaArray:
        """Return the TimeDelta at the specified location or a slice of values.

        Supports both single index access returning a :class:`TimeDelta` and
        slice access returning a new :class:`TimeDeltaArray`.

        :param index: Array index (int) or slice object
        :type index: int | slice

        :returns: Single TimeDelta for int index, TimeDeltaArray for slice
        :rtype: TimeDelta | TimeDeltaArray

        :raises TypeError: If index is bool or unsupported type
        :raises IndexError: If index is out of range

        Examples:
            Single element access:

            >>> arr = TimeDeltaArray([TimeDelta(seconds=1), TimeDelta(seconds=2)])
            >>> arr[0]
            TimeDelta(seconds=1.0)

            Slice access:

            >>> arr[0:1]
            TimeDeltaArray([TimeDelta(seconds=1.0)])
        """
        if isinstance(index, bool):
            raise TypeError("Cannot index with bool")
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
        """Return the number of TimeDelta objects in the array.

        :returns: Number of elements in the array
        :rtype: int

        Examples:
            >>> arr = TimeDeltaArray([TimeDelta(seconds=1), TimeDelta(seconds=2)])
            >>> len(arr)
            2
        """
        return len(self._array)

    @overload
    def __setitem__(self, index: int, value: TimeDelta) -> None:
        """Set a single TimeDelta by index.

        :param index: Array index
        :type index: int
        :param value: TimeDelta value to assign
        :type value: TimeDelta
        :rtype: None
        """
        ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[TimeDelta]) -> None:
        """Set multiple TimeDelta objects by slice.

        :param index: Slice object
        :type index: slice
        :param value: Iterable of TimeDelta objects
        :type value: Iterable[TimeDelta]
        :rtype: None
        """
        ...

    def __setitem__(self, index: int | slice, value: TimeDelta | Iterable[TimeDelta]) -> None:
        """Set a new value for TimeDelta at the specified location or slice.

        Supports both single element assignment and slice assignment with
        length validation for slices.

        :param index: Array index (int) or slice object
        :type index: int | slice
        :param value: TimeDelta for single assignment or iterable for slice
        :type value: TimeDelta | Iterable[TimeDelta]

        :raises TypeError: If index is bool, value type is invalid, or slice value is not iterable
        :raises ValueError: If slice assignment length doesn't match selected range
        :raises IndexError: If index is out of range

        Examples:
            Single element assignment:

            >>> arr = TimeDeltaArray([TimeDelta(seconds=1)])
            >>> arr[0] = TimeDelta(seconds=2)
            >>> arr[0]
            TimeDelta(seconds=2.0)

            Slice assignment:

            >>> arr[0:1] = [TimeDelta(seconds=3)]
            >>> arr[0]
            TimeDelta(seconds=3.0)
        """
        if isinstance(index, bool):
            raise TypeError("Cannot index with bool")
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
    def __delitem__(self, index: int) -> None:
        """Delete a single TimeDelta by index.

        :param index: Array index
        :type index: int
        :rtype: None
        """
        ...

    @overload
    def __delitem__(self, index: slice) -> None:
        """Delete multiple TimeDelta objects by slice.

        :param index: Slice object
        :type index: slice
        :rtype: None
        """
        ...

    def __delitem__(self, index: int | slice) -> None:
        """Delete the value at the specified location or slice.

        Removes elements from the array and adjusts the array size accordingly.
        Supports both single element deletion and slice deletion.

        :param index: Array index (int) or slice object
        :type index: int | slice

        :raises TypeError: If index is bool or unsupported type
        :raises IndexError: If index is out of range

        Examples:
            Delete single element:

            >>> arr = TimeDeltaArray([TimeDelta(seconds=1), TimeDelta(seconds=2)])
            >>> del arr[0]
            >>> len(arr)
            1

            Delete slice:

            >>> del arr[0:1]
            >>> len(arr)
            0
        """
        if isinstance(index, bool):
            raise TypeError("Cannot index with bool")
        if isinstance(index, (int, slice)):
            self._array = np.delete(self._array, index)
        else:
            raise TypeError("Index must be an int or slice")

    def insert(self, index: int, value: TimeDelta) -> None:
        """Insert the TimeDelta value before the specified index.

        Inserts the value at the given position, shifting existing elements
        to the right. Index is clamped to valid range [-len, len].

        :param index: Position to insert at (0-based)
        :type index: int
        :param value: TimeDelta object to insert
        :type value: TimeDelta

        :raises TypeError: If index is not int or value is not TimeDelta

        Examples:
            Insert at beginning:

            >>> arr = TimeDeltaArray([TimeDelta(seconds=2)])
            >>> arr.insert(0, TimeDelta(seconds=1))
            >>> len(arr)
            2
            >>> arr[0]
            TimeDelta(seconds=1.0)

            Insert at end:

            >>> arr.insert(len(arr), TimeDelta(seconds=3))
            >>> arr[-1]
            TimeDelta(seconds=3.0)
        """
        if isinstance(index, bool):
            raise TypeError("Cannot insert with bool")
        if not isinstance(index, int):
            raise TypeError("Index must be an int")
        if not isinstance(value, TimeDelta):
            raise TypeError("Cannot assign value that is not of type TimeDelta")
        lower = -len(self._array)
        upper = len(self._array)
        index = min(max(index, lower), upper)
        as_cvi = value.to_tuple().to_cvi()
        self._array = np.insert(self._array, index, as_cvi)

    def __imul__(self, multiplier: int) -> TimeDeltaArray:
        """Repeat the array contents n times in place.

        Modifies the array by repeating its current contents the specified
        number of times. Zero or negative multipliers result in an empty array.

        :param multiplier: Number of times to repeat the array contents
        :type multiplier: int

        :returns: Reference to self for method chaining
        :rtype: TimeDeltaArray

        :raises TypeError: If multiplier is not an integer

        Examples:
            Repeat array contents:

            >>> arr = TimeDeltaArray([TimeDelta(seconds=1)])
            >>> arr *= 3
            >>> len(arr)
            3

            Clear with zero multiplier:

            >>> arr *= 0
            >>> len(arr)
            0
        """
        if isinstance(multiplier, bool):
            raise TypeError("Cannot multiply with bool")
        if not isinstance(multiplier, int):
            raise TypeError("Multiplier must be an int")
        if multiplier <= 0:
            self._array = np.array([], dtype=CVITimeIntervalDType)
        else:
            self._array = np.tile(self._array, multiplier)
        return self

    def __eq__(self, other: Any) -> bool:
        """Return True if other is equal to this TimeDeltaArray.

        Compares arrays element-wise for equality. Only returns True if
        other is also a TimeDeltaArray with identical contents.

        :param other: Object to compare with
        :type other: Any

        :returns: True if arrays are equal, NotImplemented for other types
        :rtype: bool | NotImplemented

        Examples:
            >>> arr1 = TimeDeltaArray([TimeDelta(seconds=1)])
            >>> arr2 = TimeDeltaArray([TimeDelta(seconds=1)])
            >>> arr1 == arr2
            True
            >>> arr1 == "not an array"
            False
        """
        if not isinstance(other, TimeDeltaArray):
            return NotImplemented
        return np.array_equal(self._array, other._array)

    def __reduce__(self) -> tuple[Any, ...]:
        """Return object state for pickling support.

        Enables serialization of TimeDeltaArray objects using Python's
        pickle module by returning constructor arguments.

        :returns: Tuple containing class and constructor arguments
        :rtype: tuple[Any, ...]

        Examples:
            >>> import pickle
            >>> arr = TimeDeltaArray([TimeDelta(seconds=1)])
            >>> data = pickle.dumps(arr)
            >>> restored = pickle.loads(data)
            >>> arr == restored
            True
        """
        return (self.__class__, (list(iter(self)),))

    def __repr__(self) -> str:
        """Return repr(self) - detailed string representation for debugging.

        Returns a string that could be used to recreate the object,
        including the full module path and constructor arguments.

        :returns: Detailed string representation
        :rtype: str

        Examples:
            >>> arr = TimeDeltaArray([TimeDelta(seconds=1)])
            >>> repr(arr)
            'nitypes.bintime._timedelta_array.TimeDeltaArray([TimeDelta(seconds=1.0)])'
        """
        ctor_args = list(iter(self))
        return f"{self.__class__.__module__}.{self.__class__.__name__}({ctor_args})"

    def __str__(self) -> str:
        """Return str(self) - human-readable string representation.

        Returns a concise, readable representation of the array contents
        in bracket notation with semicolon separators.

        :returns: Human-readable string representation
        :rtype: str

        Examples:
            >>> arr = TimeDeltaArray([TimeDelta(seconds=1), TimeDelta(seconds=2)])
            >>> str(arr)
            '[TimeDelta(seconds=1.0); TimeDelta(seconds=2.0)]'
        """
        values = list(iter(self))
        return f"[{'; '.join(str(v) for v in values)}]"
