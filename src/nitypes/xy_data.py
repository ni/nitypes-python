"""XYData type for NI Python APIs.

XYData Data Type
=================
:class:`XYData`: An XYData object represents a two dimensional sequence of numeric values with
units information. Valid types for the numeric values are :any:`int` and :any:`float`.
"""

from __future__ import annotations

from collections.abc import Iterable, MutableSequence
from typing import TYPE_CHECKING, Any, Generic, Union

from typing_extensions import TypeVar, final

from nitypes._exceptions import invalid_arg_type
from nitypes.waveform._extended_properties import UNIT_DESCRIPTION

if TYPE_CHECKING:
    # Import from the public package so the docs don't reference private submodules.
    from nitypes.waveform import ExtendedPropertyDictionary
else:
    from nitypes.waveform._extended_properties import ExtendedPropertyDictionary

TNumeric = TypeVar("TNumeric", bound=Union[int, float])


@final
class XYData(Generic[TNumeric]):
    """A 2D sequence of numeric data with units and extended properties.

    Constructing
    ^^^^^^^^^^^^

    To construct an XYData object, use the :class:`XYData` class:

    >>> XYData([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    nitypes.xy_data.XYData(x_data=[1.0, 2.0, 3.0], y_data=[4.0, 5.0, 6.0], x_units='', y_units='')
    >>> XYData([1, 2, 3], [4, 5, 6], "A", "V")
    nitypes.xy_data.XYData(x_data=[1, 2, 3], y_data=[4, 5, 6], x_units='A', y_units='V')
    """

    __slots__ = [
        "_values",
        "_value_type",
        "_extended_properties",
    ]

    _values: list[list[TNumeric]]
    _value_type: type[TNumeric]
    _extended_properties: ExtendedPropertyDictionary

    def __init__(
        self,
        x_values: Iterable[TNumeric],
        y_values: Iterable[TNumeric],
        x_units: str = "",
        y_units: str = "",
        *,
        value_type: type[TNumeric] | None = None,
    ) -> None:
        """Initialize a new XYData.

        Args:
            x_values: The numeric values to store in first dimension of this object.
            y_values: The numeric values to store in second dimension of this object.
            x_units: The units string associated with x_values.
            y_units: The units string associated with y_values.
            value_type: The type of values that will be added to this XYData.
                This parameter should only be used when creating an XYData with
                empty Iterables.

        Returns:
            An XYData object.
        """
        if not x_values and not y_values:
            if not value_type:
                raise TypeError(
                    "You must specify x_values and y_values as non-empty or specify value_type."
                )
            self._value_type = value_type
        else:
            # Use the first x_value to determine _value_type.
            self._value_type = type(next(iter(x_values)))

            # Validate the values inputs
            self._validate_axis_data(x_values)
            self._validate_axis_data(y_values)

        if not isinstance(x_units, str):
            raise invalid_arg_type("x_units", "str", x_units)

        if not isinstance(y_units, str):
            raise invalid_arg_type("y_units", "str", y_units)

        self._values = [list(x_values), list(y_values)]
        if len(self._values[0]) != len(self._values[1]):
            raise ValueError("x_values and y_values must be the same length.")

        # X units and Y units are combined since there is only one units key.
        self._extended_properties = ExtendedPropertyDictionary()
        self._extended_properties[UNIT_DESCRIPTION] = ",".join([x_units, y_units])

    @property
    def x_data(self) -> MutableSequence[TNumeric]:
        """The x-axis data of this XYData."""
        return self._values[0]

    @property
    def y_data(self) -> MutableSequence[TNumeric]:
        """The y-axis data of this XYData."""
        return self._values[1]

    @property
    def x_units(self) -> str:
        """The unit of measurement, such as volts, of x_data."""
        value = self._extended_properties.get(UNIT_DESCRIPTION, "")
        assert isinstance(value, str)
        return value.split(",")[0] if value else ""

    @x_units.setter
    def x_units(self, value: str) -> None:
        if not isinstance(value, str):
            raise invalid_arg_type("x_units", "str", value)
        self._extended_properties[UNIT_DESCRIPTION] = ",".join([value, self.y_units])

    @property
    def y_units(self) -> str:
        """The unit of measurement, such as volts, of y_data."""
        value = self._extended_properties.get(UNIT_DESCRIPTION, "")
        assert isinstance(value, str)
        return value.split(",")[1] if value else ""

    @y_units.setter
    def y_units(self, value: str) -> None:
        if not isinstance(value, str):
            raise invalid_arg_type("y_units", "str", value)
        self._extended_properties[UNIT_DESCRIPTION] = ",".join([self.x_units, value])

    @property
    def extended_properties(self) -> ExtendedPropertyDictionary:
        """The extended properties for the XYData.

        .. note::
            Data stored in the extended properties dictionary may not be encrypted when you send it
            over the network or write it to a TDMS file.
        """
        return self._extended_properties

    def append(self, x_value: TNumeric, y_value: TNumeric) -> None:
        """Append an x and y value pair to this XYData."""
        self._validate_axis_data([x_value])
        self._validate_axis_data([y_value])
        self.x_data.append(x_value)
        self.y_data.append(y_value)

    def extend(
        self, x_values: MutableSequence[TNumeric], y_values: MutableSequence[TNumeric]
    ) -> None:
        """Extend x_data and y_data with the input sequences."""
        if len(x_values) != len(y_values):
            raise ValueError("X and Y sequences to extend must be the same length.")

        self._validate_axis_data(x_values)
        self._validate_axis_data(y_values)
        self.x_data.extend(x_values)
        self.y_data.extend(y_values)

    def __len__(self) -> int:
        """Return the length of x_data and y_data."""
        return len(self.x_data)

    def __eq__(self, value: object, /) -> bool:
        """Return self==value."""
        if not isinstance(value, self.__class__):
            return NotImplemented
        return (
            self._values == value._values
            and self.x_units == value.x_units
            and self.y_units == value.y_units
        )

    def __reduce__(self) -> tuple[Any, ...]:
        """Return object state for pickling."""
        return (self.__class__, (self.x_data, self.y_data, self.x_units, self.y_units))

    def __repr__(self) -> str:
        """Return repr(self)."""
        args = [
            f"x_data={self.x_data!r}",
            f"y_data={self.y_data!r}",
            f"x_units={self.x_units!r}",
            f"y_units={self.y_units!r}",
        ]
        return f"{self.__class__.__module__}.{self.__class__.__name__}({', '.join(args)})"

    def __str__(self) -> str:
        """Return str(self)."""
        if self.x_units:
            x_values_with_units = [f"{value} {self.x_units}" for value in self.x_data]
            x_str = ", ".join(x_values_with_units)
        else:
            x_values = [f"{value}" for value in self.x_data]
            x_str = ", ".join(x_values)
        x_str = f"[{x_str}]"

        if self.y_units:
            y_values_with_units = [f"{value} {self.y_units}" for value in self.y_data]
            y_str = ", ".join(y_values_with_units)
        else:
            y_values = [f"{value}" for value in self.y_data]
            y_str = ", ".join(y_values)
        y_str = f"[{y_str}]"

        return f"[{x_str}, {y_str}]"

    def _validate_axis_data(self, values: Iterable[TNumeric]) -> None:
        for value in values:
            if not isinstance(value, (int, float)):
                raise invalid_arg_type("XYData input data", "int or float", value)

            if not isinstance(value, self._value_type):
                raise self._create_value_mismatch_exception(value)

    def _create_value_mismatch_exception(self, value: object) -> TypeError:
        return TypeError(
            f"Input data does not match expected type. Input Type: {type(value)} "
            f"Expected Type: {self._value_type}"
        )
