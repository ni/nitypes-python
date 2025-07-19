from __future__ import annotations

from typing import Any, Generic, Union

from typing_extensions import TypeVar, final

from nitypes._exceptions import invalid_arg_type, invalid_arg_value
from nitypes.waveform._extended_properties import (
    UNIT_DESCRIPTION,
    ExtendedPropertyDictionary,
)

_ScalarType_co = TypeVar("_ScalarType_co", bound=Union[bool, int, float, str], covariant=True)
_NUMERIC = (bool, int, float)


@final
class Scalar(Generic[_ScalarType_co]):
    """A scalar data class, which encapsulates scalar data and units information.

    Constructing
    ^^^^^^^^^^^^

    To construct a scalar data object, use the :class:`Scalar` class:

    >>> Scalar(False)
    nitypes.scalar.Scalar(value=False, units='')
    >>> Scalar(0)
    nitypes.scalar.Scalar(value=0, units='')
    >>> Scalar(5.0, 'volts')
    nitypes.scalar.Scalar(value=5.0, units='volts')
    >>> Scalar("value", "volts")
    nitypes.scalar.Scalar(value='value', units='volts')

    Class members
    ^^^^^^^^^^^^^
    """

    __slots__ = [
        "_value",
        "_extended_properties",
    ]

    _value: _ScalarType_co
    _extended_properties: ExtendedPropertyDictionary

    def __init__(
        self,
        value: _ScalarType_co,
        units: str = "",
    ) -> None:
        """Initialize a new scalar.

        Args:
            value: The scalar data to store in this object.
            units: The units string associated with this data.

        Returns:
            A scalar data object.
        """
        if not isinstance(value, (bool, int, float, str)):
            raise invalid_arg_type("scalar input data", "bool, int, float, or str", value)

        if not isinstance(units, str):
            raise invalid_arg_type("units", "str", units)

        # The ScalarData proto type only supports 32 bit integers. Make sure we're in range.
        if isinstance(value, int):
            if value <= -0x80000000 or value >= 0x7FFFFFFF:
                raise invalid_arg_value("integer scalar value", "within the range of Int32", value)

        self._value = value
        self._extended_properties = ExtendedPropertyDictionary()
        self._extended_properties[UNIT_DESCRIPTION] = units

    @property
    def value(self) -> _ScalarType_co:
        """The scalar value."""
        return self._value

    @property
    def units(self) -> str:
        """The unit of measurement, such as volts, of the scalar."""
        value = self._extended_properties.get(UNIT_DESCRIPTION, "")
        assert isinstance(value, str)
        return value

    @property
    def extended_properties(self) -> ExtendedPropertyDictionary:
        """The extended properties for the scalar.

        .. note::
            Data stored in the extended properties dictionary may not be encrypted when you send it
            over the network or write it to a TDMS file.
        """
        return self._extended_properties

    def __eq__(self, value: object, /) -> bool:
        """Return self==value."""
        if not isinstance(value, self.__class__):
            return NotImplemented
        return self.value == value.value and self.units == value.units

    def __gt__(self, value: Scalar[_ScalarType_co]) -> bool:
        """Return self > value."""
        if not isinstance(value, self.__class__):
            return NotImplemented
        self._check_units_equal_for_comparison(value.units)
        if isinstance(self.value, _NUMERIC) and isinstance(value.value, _NUMERIC):
            return self.value > value.value  # type: ignore[no-any-return,operator]  # https://github.com/python/mypy/issues/19454
        elif isinstance(self.value, str) and isinstance(value.value, str):
            return self.value > value.value
        else:
            raise TypeError("Comparing Scalar objects of numeric and string types is not permitted")

    def __ge__(self, value: Scalar[_ScalarType_co]) -> bool:
        """Return self >= value."""
        if not isinstance(value, self.__class__):
            return NotImplemented
        self._check_units_equal_for_comparison(value.units)
        if isinstance(self.value, _NUMERIC) and isinstance(value.value, _NUMERIC):
            return self.value >= value.value  # type: ignore[no-any-return,operator]  # https://github.com/python/mypy/issues/19454
        elif isinstance(self.value, str) and isinstance(value.value, str):
            return self.value >= value.value
        else:
            raise TypeError("Comparing Scalar objects of numeric and string types is not permitted")

    def __lt__(self, value: Scalar[_ScalarType_co]) -> bool:
        """Return self < value."""
        if not isinstance(value, self.__class__):
            return NotImplemented
        self._check_units_equal_for_comparison(value.units)
        if isinstance(self.value, _NUMERIC) and isinstance(value.value, _NUMERIC):
            return self.value < value.value  # type: ignore[no-any-return,operator]  # https://github.com/python/mypy/issues/19454
        elif isinstance(self.value, str) and isinstance(value.value, str):
            return self.value < value.value
        else:
            raise TypeError("Comparing Scalar objects of numeric and string types is not permitted")

    def __le__(self, value: Scalar[_ScalarType_co]) -> bool:
        """Return self <= value."""
        if not isinstance(value, self.__class__):
            return NotImplemented
        self._check_units_equal_for_comparison(value.units)
        if isinstance(self.value, _NUMERIC) and isinstance(value.value, _NUMERIC):
            return self.value <= value.value  # type: ignore[no-any-return,operator]  # https://github.com/python/mypy/issues/19454
        elif isinstance(self.value, str) and isinstance(value.value, str):
            return self.value <= value.value
        else:
            raise TypeError("Comparing Scalar objects of numeric and string types is not permitted")

    def __reduce__(self) -> tuple[Any, ...]:
        """Return object state for pickling."""
        return (self.__class__, (self.value, self.units))

    def __repr__(self) -> str:
        """Return repr(self)."""
        args = [f"value={self.value!r}", f"units={self.units!r}"]
        return f"{self.__class__.__module__}.{self.__class__.__name__}({', '.join(args)})"

    def __str__(self) -> str:
        """Return str(self)."""
        value_str = str(self.value)
        if self.units:
            value_str += f" {self.units}"

        return value_str

    def _check_units_equal_for_comparison(self, other_units: str) -> None:
        """Raise a ValueError if other_units != self.units."""
        if self.units != other_units:
            raise ValueError("Comparing Scalar objects with different units is not permitted.")
