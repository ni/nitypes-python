"""Scalar data types for NI Python APIs.

Scalar Data Type
=================

:class:`Scalar`: A scalar data object represents a single scalar value with units information.
Valid types for the scalar value are :any:`bool`, :any:`int`, :any:`float`, and :any:`str`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Generic, Union

from typing_extensions import Self, TypeVar, final

from nitypes._exceptions import invalid_arg_type
from nitypes.waveform._extended_properties import UNIT_DESCRIPTION
from nitypes.waveform.typing import ExtendedPropertyValue

if TYPE_CHECKING:
    # Import from the public package so the docs don't reference private submodules.
    from nitypes.waveform import ExtendedPropertyDictionary
else:
    from nitypes.waveform._extended_properties import ExtendedPropertyDictionary

TScalar_co = TypeVar("TScalar_co", bound=Union[bool, int, float, str], covariant=True)
_NUMERIC = (bool, int, float)


@final
class Scalar(Generic[TScalar_co]):
    """A scalar data class, which encapsulates scalar data and units information.

    Constructing
    ^^^^^^^^^^^^

    To construct a scalar data object, use the :class:`Scalar` class:

    >>> Scalar(False)
    nitypes.scalar.Scalar(value=False)
    >>> Scalar(0)
    nitypes.scalar.Scalar(value=0)
    >>> Scalar(5.0, 'volts')
    nitypes.scalar.Scalar(value=5.0, units='volts')
    >>> Scalar("value", "volts")
    nitypes.scalar.Scalar(value='value', units='volts')

    Comparing Scalar Objects
    ^^^^^^^^^^^^^^^^^^^^^^^^

    You can compare scalar objects using standard comparison
    operators: ``<``, ``<=``, ``>``, ``>=``, ``==``, and ``!=``.
    Detailed descriptions of operator behaviors are provided below.

    Equality Comparison Operators
    -----------------------------

    Equality comparison operators (``==`` and ``!=``) are always supported and behave as follows:

    - Comparison of scalar objects with compatible types and identical units results
      in ``True`` or ``False`` based on the comparison of scalar object values.
    - Comparison of scalar objects with incompatible types (such as numeric and string)
      results in inequality.
    - Comparison of scalar objects with different units results in inequality.

    Examples:

    >>> Scalar(5.0, 'V') == Scalar(5.0, 'V') # Numeric scalars with identical values and units
    True
    >>> Scalar(5.0, 'V') == Scalar(12.3, 'V') # Numeric scalars with identical units
    False
    >>> Scalar(5.0, 'V') != Scalar(12.3, 'V') # Numeric scalars with identical units
    True
    >>> Scalar("apple") == Scalar("banana") # String scalars
    False
    >>> Scalar("apple") == Scalar("Apple") # String scalars - note case sensitivity
    False
    >>> Scalar(0.5, 'V') == Scalar(500, 'mV') # Numeric scalars with different units
    False
    >>> Scalar(5.0, 'V') == Scalar("5.0", 'V') # Comparison of a numeric and a string scalar
    False

    Order Comparison Operators
    --------------------------

    Order comparison operators (``<``, ``<=``, ``>``, and ``>=``) behave as follows:

    - Comparison of scalar objects with compatible types and identical units results
      in ``True`` or ``False`` based on the comparison of scalar object values.
    - Comparison of scalar objects with incompatible types (such as numeric and string)
      is not permitted and will raise a ``TypeError`` exception.
    - Comparison of scalar objects with compatible types and different units
      is not permitted and will raise a ``ValueError`` exception.

    Examples:

    >>> Scalar(5.0, 'V') < Scalar(10.0, 'V') # Numeric scalars with identical units
    True
    >>> Scalar(5.0, 'V') >= Scalar(10.0, 'V') # Numeric scalars with identical units
    False
    >>> Scalar("apple") < Scalar("banana") # String scalars
    True
    >>> Scalar("apple") < Scalar("Banana") # String scalars - note case sensitivity
    False
    >>> Scalar(5.0, 'V') < Scalar("5.0", 'V') # Comparison of a numeric and a string scalar
    Traceback (most recent call last):
        ...
    TypeError: Comparing Scalar objects of numeric and string types is not permitted.
    >>> Scalar(0.5, 'V') < Scalar(500, 'mV') # Numeric scalars with different units
    Traceback (most recent call last):
        ...
    ValueError: Comparing Scalar objects with different units is not permitted.

    Class Members
    ^^^^^^^^^^^^^
    """

    __slots__ = [
        "_value",
        "_extended_properties",
    ]

    _value: TScalar_co
    _extended_properties: ExtendedPropertyDictionary

    def __init__(
        self,
        value: TScalar_co,
        units: str = "",
        *,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = None,
        copy_extended_properties: bool = True,
    ) -> None:
        """Initialize a new scalar.

        Args:
            value: The scalar data to store in this object.
            units: The units string associated with this data.
            extended_properties: The extended properties of the Scalar.
            copy_extended_properties: Specifies whether to copy the extended properties or take
                ownership.

        Returns:
            A scalar data object.
        """
        if not isinstance(value, (bool, int, float, str)):
            raise invalid_arg_type("scalar input data", "bool, int, float, or str", value)

        if not isinstance(units, str):
            raise invalid_arg_type("units", "str", units)

        self._value = value
        if copy_extended_properties or not isinstance(
            extended_properties, ExtendedPropertyDictionary
        ):
            extended_properties = ExtendedPropertyDictionary(extended_properties)
        self._extended_properties = extended_properties

        # If units are not already in extended properties, set them.
        if UNIT_DESCRIPTION not in self._extended_properties:
            self._extended_properties[UNIT_DESCRIPTION] = units
        elif units and units != self._extended_properties.get(UNIT_DESCRIPTION):
            raise ValueError(
                "The specified units input does not match the units specified in "
                "extended_properties."
            )

    @property
    def value(self) -> TScalar_co:
        """The scalar value."""
        return self._value

    @property
    def units(self) -> str:
        """The unit of measurement, such as volts, of the scalar."""
        value = self._extended_properties.get(UNIT_DESCRIPTION, "")
        assert isinstance(value, str)
        return value

    @units.setter
    def units(self, value: str) -> None:
        if not isinstance(value, str):
            raise invalid_arg_type("units", "str", value)
        self._extended_properties[UNIT_DESCRIPTION] = value

    @property
    def extended_properties(self) -> ExtendedPropertyDictionary:
        """The extended properties for the scalar.

        .. note::
            Data stored in the extended properties dictionary may not be encrypted when you send it
            over the network or write it to a TDMS file.
        """
        return self._extended_properties

    def __eq__(self, other: object, /) -> bool:
        """Return self == other."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.value == other.value and self.units == other.units

    def __gt__(self, other: Scalar[TScalar_co], /) -> bool:
        """Return self > other."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        self._check_units_equal_for_comparison(other.units)
        if isinstance(self.value, _NUMERIC) and isinstance(other.value, _NUMERIC):
            return self.value > other.value  # type: ignore[no-any-return,operator]  # https://github.com/python/mypy/issues/19454
        elif isinstance(self.value, str) and isinstance(other.value, str):
            return self.value > other.value
        else:
            raise _comparing_numeric_and_string_not_permitted()

    def __ge__(self, other: Scalar[TScalar_co], /) -> bool:
        """Return self >= other."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        self._check_units_equal_for_comparison(other.units)
        if isinstance(self.value, _NUMERIC) and isinstance(other.value, _NUMERIC):
            return self.value >= other.value  # type: ignore[no-any-return,operator]  # https://github.com/python/mypy/issues/19454
        elif isinstance(self.value, str) and isinstance(other.value, str):
            return self.value >= other.value
        else:
            raise _comparing_numeric_and_string_not_permitted()

    def __lt__(self, other: Scalar[TScalar_co], /) -> bool:
        """Return self < other."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        self._check_units_equal_for_comparison(other.units)
        if isinstance(self.value, _NUMERIC) and isinstance(other.value, _NUMERIC):
            return self.value < other.value  # type: ignore[no-any-return,operator]  # https://github.com/python/mypy/issues/19454
        elif isinstance(self.value, str) and isinstance(other.value, str):
            return self.value < other.value
        else:
            raise _comparing_numeric_and_string_not_permitted()

    def __le__(self, other: Scalar[TScalar_co], /) -> bool:
        """Return self <= other."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        self._check_units_equal_for_comparison(other.units)
        if isinstance(self.value, _NUMERIC) and isinstance(other.value, _NUMERIC):
            return self.value <= other.value  # type: ignore[no-any-return,operator]  # https://github.com/python/mypy/issues/19454
        elif isinstance(self.value, str) and isinstance(other.value, str):
            return self.value <= other.value
        else:
            raise _comparing_numeric_and_string_not_permitted()

    def __reduce__(self) -> tuple[Any, ...]:
        """Return object state for pickling."""
        ctor_args = (self.value,)
        ctor_kwargs: dict[str, Any] = {
            "extended_properties": self._extended_properties,
            "copy_extended_properties": False,
        }
        return (self.__class__._unpickle, (ctor_args, ctor_kwargs))

    @classmethod
    def _unpickle(cls, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Self:
        return cls(*args, **kwargs)

    def __repr__(self) -> str:
        """Return repr(self)."""
        args = [f"value={self.value!r}"]

        if self.units:
            args.append(f"units={self.units!r}")

        # Only display the extended properties if non-units entries are specified.
        if any(key for key in self.extended_properties.keys() if key != UNIT_DESCRIPTION):
            args.append(f"extended_properties={self.extended_properties!r}")

        return f"{self.__class__.__module__}.{self.__class__.__name__}({', '.join(args)})"

    def __str__(self) -> str:
        """Return str(self)."""
        value_str = str(self.value)
        if self.units:
            value_str += f" {self.units}"

        return value_str

    def _check_units_equal_for_comparison(self, other_units: str) -> None:
        if self.units != other_units:
            raise ValueError("Comparing Scalar objects with different units is not permitted.")


def _comparing_numeric_and_string_not_permitted() -> TypeError:
    return TypeError("Comparing Scalar objects of numeric and string types is not permitted.")
