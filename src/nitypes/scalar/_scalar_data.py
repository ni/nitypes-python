from __future__ import annotations

from typing import Generic

from typing_extensions import final, TypeVar

from nitypes._exceptions import invalid_arg_type, invalid_arg_value
from nitypes.waveform._extended_properties import (
    UNIT_DESCRIPTION,
    ExtendedPropertyDictionary,
)

_ScalarType = TypeVar("_ScalarType", bool, int, float, str)


@final
class ScalarData(Generic[_ScalarType]):
    """A scalar data class, which encapsulates scalar data and units information."""

    __slots__ = [
        "_value",
        "_extended_properties",
    ]

    _value: _ScalarType
    _extended_properties: ExtendedPropertyDictionary

    def __init__(
        self,
        value: _ScalarType,
        units: str = "",
    ) -> None:
        """Construct a scalar data object.

        Args:
            value: The scalar data to store in this object.
            units: The units string associated with this data.

        Returns:
            A scalar data object.
        """
        if not isinstance(value, (bool, int, float, str)):
            raise invalid_arg_type("scalar input data", "bool, int, float, or str", value)

        # The ScalarData proto type only supports 32 bit integers. Make sure we're in range.
        if isinstance(value, int):
            if value <= -0x80000000 or value >= 0x7FFFFFFF:
                raise invalid_arg_value("integer scalar value", "within the range of Int32", value)

        self._value = value
        self._extended_properties = ExtendedPropertyDictionary()
        self._extended_properties[UNIT_DESCRIPTION] = units

    @property
    def value(self) -> _ScalarType:
        """The scalar value."""
        return self._value

    @property
    def units(self) -> str:
        """The unit of measurement, such as volts, of the scalar."""
        value = self._extended_properties.get(UNIT_DESCRIPTION, "")
        assert isinstance(value, str)
        return value

    def __eq__(self, value: object, /) -> bool:
        """Return self==value."""
        if not isinstance(value, self.__class__):
            return NotImplemented
        return self.value == value.value and self.units == value.units

    def __gt__(self, value: object) -> bool:
        """Return self > value."""
        if not isinstance(value, self.__class__):
            return NotImplemented

        if self.units != value.units:
            raise ValueError("Comparing ScalarData objects with different units is not permitted.")

        return self.value > value.value

    def __ge__(self, value: object) -> bool:
        """Return self >= value."""
        if not isinstance(value, self.__class__):
            return NotImplemented

        if self.units != value.units:
            raise ValueError("Comparing ScalarData objects with different units is not permitted.")

        return self.value >= value.value

    def __lt__(self, value: object) -> bool:
        """Return self < value."""
        if not isinstance(value, self.__class__):
            return NotImplemented

        if self.units != value.units:
            raise ValueError("Comparing ScalarData objects with different units is not permitted.")

        return self.value < value.value

    def __le__(self, value: object) -> bool:
        """Return self <= value."""
        if not isinstance(value, self.__class__):
            return NotImplemented

        if self.units != value.units:
            raise ValueError("Comparing ScalarData objects with different units is not permitted.")

        return self.value <= value.value

    def __repr__(self) -> str:
        """Return repr(self)."""
        args = [f"value={self.value}", f"units={self.units}"]
        return f"{self.__class__.__module__}.{self.__class__.__name__}({', '.join(args)})"

    def __str__(self) -> str:
        """Return str(self)."""
        value_str = str(self.value)
        if self.units:
            value_str += f" {self.units}"

        return value_str
