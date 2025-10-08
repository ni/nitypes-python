"""XYData type for NI Python APIs.

XYData Data Type
=================
:class:`XYData`: An XYData object represents two axes (sequences) of numeric values with units
information. Valid types for the numeric values are :any:`int` and :any:`float`.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, Union, overload

import numpy as np
import numpy.typing as npt
from typing_extensions import Self, TypeVar, final

from nitypes._arguments import validate_dtype
from nitypes._exceptions import invalid_arg_type, invalid_array_ndim
from nitypes._numpy import asarray as _np_asarray, long as _np_long, ulong as _np_ulong
from nitypes.waveform._exceptions import create_datatype_mismatch_error
from nitypes.waveform.typing import ExtendedPropertyValue

if sys.version_info < (3, 10):
    import array as std_array

if TYPE_CHECKING:
    # Import from the public package so the docs don't reference private submodules.
    from nitypes.waveform import ExtendedPropertyDictionary
else:
    from nitypes.waveform._extended_properties import ExtendedPropertyDictionary

# Extended property keys for X and Y units.
_UNIT_DESCRIPTION_X = "NI_UnitDescription_X"
_UNIT_DESCRIPTION_Y = "NI_UnitDescription_Y"

TData = TypeVar("TData", bound=Union[np.floating, np.integer])
TOtherData = TypeVar("TOtherData", bound=Union[np.floating, np.integer])

# Use the C types here because np.isdtype() considers some of them to be distinct types, even when
# they have the same size (e.g. np.intc vs. np.int_).
_DATA_DTYPES = (
    # Floating point
    np.single,
    np.double,
    # Signed integers
    np.byte,
    np.short,
    np.intc,
    np.int_,
    _np_long,
    np.longlong,
    # Unsigned integers
    np.ubyte,
    np.ushort,
    np.uintc,
    np.uint,
    _np_ulong,
    np.ulonglong,
)


@final
class XYData(Generic[TData]):
    """Two axes (sequences) of numeric values with units information.

    Constructing
    ^^^^^^^^^^^^

    To construct an XYData object, use the :class:`XYData` class:

    >>> XYData(np.array([1.1], np.float64), np.array([4.1], np.float64))
    nitypes.xy_data.XYData(x_data=array([1.1]), y_data=array([4.1]),
    extended_properties={'NI_UnitDescription_X': '', 'NI_UnitDescription_Y': ''})
    >>> XYData(np.array([1, 2]), np.array([4, 5]), x_units="A", y_units="V")
    nitypes.xy_data.XYData(x_data=array([1, 2]), y_data=array([4, 5]),
    extended_properties={'NI_UnitDescription_X': 'A', 'NI_UnitDescription_Y': 'V'})

    To construct an XYData object using built-in lists, use from_arrays_1d():

    >>> XYData.from_arrays_1d([1, 2], [5, 6], np.int32)
    nitypes.xy_data.XYData(x_data=array([1, 2], dtype=int32), y_data=array([5, 6], dtype=int32),
    extended_properties={'NI_UnitDescription_X': '', 'NI_UnitDescription_Y': ''})
    >>> XYData.from_arrays_1d([1.0, 1.1], [1.2, 1.3], np.float64)
    nitypes.xy_data.XYData(x_data=array([1. , 1.1]), y_data=array([1.2, 1.3]),
    extended_properties={'NI_UnitDescription_X': '', 'NI_UnitDescription_Y': ''})
    """

    __slots__ = [
        "_x_data",
        "_y_data",
        "_extended_properties",
    ]

    _x_data: npt.NDArray[TData]
    _y_data: npt.NDArray[TData]
    _extended_properties: ExtendedPropertyDictionary

    @overload
    @classmethod
    def from_arrays_1d(
        cls,
        x_array: npt.NDArray[TOtherData],
        y_array: npt.NDArray[TOtherData],
        dtype: None = ...,
        *,
        x_units: str = ...,
        y_units: str = ...,
        copy: bool = ...,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = ...,
    ) -> XYData[TOtherData]: ...

    @overload
    @classmethod
    def from_arrays_1d(
        cls,
        x_array: npt.NDArray[Any] | Sequence[Any],
        y_array: npt.NDArray[Any] | Sequence[Any],
        dtype: type[TOtherData] | np.dtype[TOtherData],
        *,
        x_units: str = ...,
        y_units: str = ...,
        copy: bool = ...,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = ...,
    ) -> XYData[TOtherData]: ...

    @overload
    @classmethod
    def from_arrays_1d(
        cls,
        x_array: npt.NDArray[Any] | Sequence[Any],
        y_array: npt.NDArray[Any] | Sequence[Any],
        dtype: npt.DTypeLike = ...,
        *,
        x_units: str = ...,
        y_units: str = ...,
        copy: bool = ...,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = ...,
    ) -> XYData[Any]: ...

    @classmethod
    def from_arrays_1d(
        cls,
        x_array: npt.NDArray[Any] | Sequence[Any],
        y_array: npt.NDArray[Any] | Sequence[Any],
        dtype: npt.DTypeLike = None,
        *,
        x_units: str = "",
        y_units: str = "",
        copy: bool = True,
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = None,
    ) -> XYData[Any]:
        """Construct an XYData from two one-dimensional arrays or sequences.

        Args:
            x_array: The x-axis data as a one-dimensional array or a sequence.
            y_array: The y-axis data as a one-dimensional array or a sequence.
            dtype: The NumPy data type for the XYdata axes. This argument is required
                when x_array and y_array are sequences.
            x_units: The units string associated with x_array.
            y_units: The units string associated with y_array
            copy: Specifies whether to copy the arrays or save references to them.
            extended_properties: The extended properties of the XYData.

        Returns:
            An XYData object containing the specified data.
        """
        if isinstance(x_array, np.ndarray):
            if x_array.ndim != 1:
                raise invalid_array_ndim(
                    "input array", "one-dimensional array or sequence", x_array.ndim
                )
        elif isinstance(x_array, Sequence) or (
            sys.version_info < (3, 10) and isinstance(x_array, std_array.array)
        ):
            if dtype is None:
                raise ValueError("You must specify a dtype when the input array is a sequence.")
        else:
            raise invalid_arg_type("input array", "one-dimensional array or sequence", x_array)

        if isinstance(y_array, np.ndarray):
            if y_array.ndim != 1:
                raise invalid_array_ndim(
                    "input array", "one-dimensional array or sequence", y_array.ndim
                )
        elif isinstance(y_array, Sequence) or (
            sys.version_info < (3, 10) and isinstance(y_array, std_array.array)
        ):
            if dtype is None:
                raise ValueError("You must specify a dtype when the input array is a sequence.")
        else:
            raise invalid_arg_type("input array", "one-dimensional array or sequence", y_array)

        return cls(
            x_data=_np_asarray(x_array, dtype, copy=copy),
            y_data=_np_asarray(y_array, dtype, copy=copy),
            x_units=x_units,
            y_units=y_units,
            extended_properties=extended_properties,
        )

    def __init__(
        self: XYData[TOtherData],
        x_data: npt.NDArray[TOtherData],
        y_data: npt.NDArray[TOtherData],
        *,
        x_units: str = "",
        y_units: str = "",
        extended_properties: Mapping[str, ExtendedPropertyValue] | None = None,
        copy_extended_properties: bool = True,
    ) -> None:
        """Initialize a new XYData.

        Args:
            x_data: A NumPy ndarray to use for x-axis data storage. The XYData takes ownership
                of this array. If not specified, an ndarray is created based on the specified
                dtype and capacity.
            y_data: A NumPy ndarray to use for y-axis data storage. The XYData takes ownership
                of this array. If not specified, an ndarray is created based on the specified
                dtype and capacity.
            x_units: The units string associated with x_data.
            y_units: The units string associated with y_data.
            extended_properties: The extended properties of the XYData.
            copy_extended_properties: Specifies whether to copy the extended properties or take
                ownership.

        Returns:
            An XYData object.
        """
        if x_data.dtype != y_data.dtype:
            raise TypeError("x_data and y_data must have the same type.")

        if isinstance(x_data, np.ndarray) and isinstance(y_data, np.ndarray):
            self._init_with_provided_arrays(
                x_data,
                y_data,
                x_data.dtype,
            )
        else:
            raise invalid_arg_type("raw data", "NumPy ndarray", x_data)

        if copy_extended_properties or not isinstance(
            extended_properties, ExtendedPropertyDictionary
        ):
            extended_properties = ExtendedPropertyDictionary(extended_properties)
        self._extended_properties = extended_properties

        # If x and y units are not already in extended properties, set them.
        # If the caller specifies a non-blank x or y units, overwrite the existing entry.
        if _UNIT_DESCRIPTION_X not in self._extended_properties or x_units:
            self._extended_properties[_UNIT_DESCRIPTION_X] = x_units
        if _UNIT_DESCRIPTION_Y not in self._extended_properties or y_units:
            self._extended_properties[_UNIT_DESCRIPTION_Y] = y_units

    def _init_with_provided_arrays(
        self,
        x_data: npt.NDArray[TData],
        y_data: npt.NDArray[TData],
        dtype: npt.DTypeLike = None,
    ) -> None:
        if not isinstance(x_data, np.ndarray):
            raise invalid_arg_type("x-axis input array", "one-dimensional array", x_data)
        if not isinstance(y_data, np.ndarray):
            raise invalid_arg_type("y-axis input array", "one-dimensional array", y_data)
        if x_data.ndim != 1:
            raise invalid_array_ndim("x-axis input array", "one-dimensional array", x_data.ndim)
        if y_data.ndim != 1:
            raise invalid_array_ndim("y-axis input array", "one-dimensional array", y_data.ndim)
        if len(x_data) != len(y_data):
            raise ValueError("x_data and y_data must be the same length.")

        if dtype is None:
            if x_data.dtype != y_data.dtype:
                raise TypeError("x_data and y_data must be the same type.")
            dtype = x_data.dtype

        validate_dtype(dtype, _DATA_DTYPES)
        if dtype != x_data.dtype:
            raise create_datatype_mismatch_error(
                "input array", x_data.dtype, "requested", np.dtype(dtype)
            )
        if dtype != y_data.dtype:
            raise create_datatype_mismatch_error(
                "input array", y_data.dtype, "requested", np.dtype(dtype)
            )

        self._x_data = x_data
        self._y_data = y_data

    @property
    def x_data(self) -> npt.NDArray[TData]:
        """The x-axis data of this XYData."""
        return self._x_data

    @property
    def y_data(self) -> npt.NDArray[TData]:
        """The y-axis data of this XYData."""
        return self._y_data

    @property
    def x_units(self) -> str:
        """The unit of measurement, such as volts, of x_data."""
        value = self._extended_properties.get(_UNIT_DESCRIPTION_X, "")
        assert isinstance(value, str)
        return value

    @x_units.setter
    def x_units(self, value: str) -> None:
        if not isinstance(value, str):
            raise invalid_arg_type("x_units", "str", value)
        self._extended_properties[_UNIT_DESCRIPTION_X] = value

    @property
    def y_units(self) -> str:
        """The unit of measurement, such as volts, of y_data."""
        value = self._extended_properties.get(_UNIT_DESCRIPTION_Y, "")
        assert isinstance(value, str)
        return value

    @y_units.setter
    def y_units(self, value: str) -> None:
        if not isinstance(value, str):
            raise invalid_arg_type("y_units", "str", value)
        self._extended_properties[_UNIT_DESCRIPTION_Y] = value

    @property
    def dtype(self) -> np.dtype[TData]:
        """The NumPy dtype for the XYData."""
        return self._x_data.dtype

    @property
    def extended_properties(self) -> ExtendedPropertyDictionary:
        """The extended properties for the XYData.

        .. note::
            Data stored in the extended properties dictionary may not be encrypted when you send it
            over the network or write it to a TDMS file.
        """
        return self._extended_properties

    def __eq__(self, value: object, /) -> bool:
        """Return self==value."""
        if not isinstance(value, self.__class__):
            return NotImplemented
        return (
            np.array_equal(self.x_data, value.x_data)
            and np.array_equal(self.y_data, value.y_data)
            and self.x_units == value.x_units
            and self.y_units == value.y_units
        )

    def __reduce__(self) -> tuple[Any, ...]:
        """Return object state for pickling."""
        ctor_args = (self._x_data, self._y_data)
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
        args = [
            f"x_data={self.x_data!r}",
            f"y_data={self.y_data!r}",
            f"extended_properties={self._extended_properties._properties!r}",
        ]
        return f"{self.__class__.__module__}.{self.__class__.__name__}({', '.join(args)})"

    def __str__(self) -> str:
        """Return str(self)."""
        x_str = XYData._format_values_with_units(self.x_data, self.x_units)
        y_str = XYData._format_values_with_units(self.y_data, self.y_units)
        return f"[{x_str}, {y_str}]"

    @staticmethod
    def _format_values_with_units(values: npt.NDArray[TData], units: str) -> str:
        if units:
            values_with_units = [f"{value} {units}" for value in values]
            values_str = ", ".join(values_with_units)
        else:
            values_without_units = [f"{value}" for value in values]
            values_str = ", ".join(values_without_units)

        return f"[{values_str}]"
