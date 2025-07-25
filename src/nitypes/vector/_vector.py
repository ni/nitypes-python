from __future__ import annotations

from typing import Any, Generic, Union

from typing_extensions import TypeVar, final

from nitypes._exceptions import invalid_arg_type, invalid_arg_value
from nitypes.waveform._extended_properties import (
    UNIT_DESCRIPTION,
    ExtendedPropertyDictionary,
)

_VectorType = TypeVar("_VectorType", bound=Union[bool, int, float, str])


@final
class Vector(Generic[_VectorType]):
    """A Vector of scalar values with units information.

    Constructing
    ^^^^^^^^^^^^

    To construct a vector data object, use the :class:`Vector` class:

    >>> Vector([False, True])
    nitypes.vector.Vector(values=[False, True], units='')
    >>> Vector([0, 1, 2])
    nitypes.vector.Vector(values=[0, 1, 2], units='')
    >>> Vector([5.0, 6.0], 'volts')
    nitypes.vector.Vector(values=[5.0, 6.0], units='volts')
    >>> Vector(["one", "two"], "volts")
    nitypes.vector.Vector(values=['one', 'two'], units='volts')
    """

    __slots__ = [
        "_values",
        "_extended_properties",
    ]

    _values: list[_VectorType]
    _extended_properties: ExtendedPropertyDictionary

    def __init__(
        self,
        values: list[_VectorType],
        units: str = "",
    ) -> None:
        """Initialize a new vector.

        Args:
            values: The scalar values to store in this object.
            units: The units string associated with this data.

        Returns:
            A vector data object.
        """
        if not values:
            values = []
        else:
            first_element = values[0]
            if not isinstance(first_element, (bool, int, float, str)):
                raise invalid_arg_type("vector input data", "bool, int, float, or str", values)

            # The Vector proto type only supports 32 bit integers. Make sure we're in range.
            if isinstance(first_element, int):
                if first_element <= -0x80000000 or first_element >= 0x7FFFFFFF:
                    raise invalid_arg_value(
                        "integer vector value", "within the range of Int32", values
                    )

        if not isinstance(units, str):
            raise invalid_arg_type("units", "str", units)

        self._values = values
        self._extended_properties = ExtendedPropertyDictionary()
        self._extended_properties[UNIT_DESCRIPTION] = units

    @property
    def values(self) -> list[_VectorType]:
        """The vector values."""
        return self._values

    @property
    def units(self) -> str:
        """The unit of measurement, such as volts, of the vector."""
        value = self._extended_properties.get(UNIT_DESCRIPTION, "")
        assert isinstance(value, str)
        return value

    @property
    def extended_properties(self) -> ExtendedPropertyDictionary:
        """The extended properties for the vector.

        .. note::
            Data stored in the extended properties dictionary may not be encrypted when you send it
            over the network or write it to a TDMS file.
        """
        return self._extended_properties

    def append(self, value: _VectorType) -> None:
        """Append a value to this vector."""
        if not self.values:
            self.values.append(value)
        elif isinstance(value, type(self._values[0])):
            self.values.append(value)
        else:
            raise TypeError(
                "The datatype of the appended value must match the type of the existing vector"
                f" values. Appended type: {type(value)}. Existing type: {type(self._values[0])}"
            )

    def extend(self, values: list[_VectorType]) -> None:
        """Extend this vector with the given values."""
        if not values:
            return
        elif not self.values:
            self.values.extend(values)
        elif isinstance(values[0], type(self.values[0])):
            self.values.extend(values)
        else:
            raise TypeError(
                "The datatype of the extended values must match the type of the existing vector"
                f" values. extended type: {type(values[0])}. Existing type: {type(self._values[0])}"
            )

    def __eq__(self, value: object, /) -> bool:
        """Return self==value."""
        if not isinstance(value, self.__class__):
            return NotImplemented
        return self.values == value.values and self.units == value.units

    def __reduce__(self) -> tuple[Any, ...]:
        """Return object state for pickling."""
        return (self.__class__, (self.values, self.units))

    def __repr__(self) -> str:
        """Return repr(self)."""
        args = [f"values={self.values!r}", f"units={self.units!r}"]
        return f"{self.__class__.__module__}.{self.__class__.__name__}({', '.join(args)})"

    def __str__(self) -> str:
        """Return str(self)."""
        if self.units:
            values_with_units = [f"{value} {self.units}" for value in self.values]
            return ", ".join(values_with_units)
        else:
            values = [f"{value}" for value in self.values]
            return ", ".join(values)
