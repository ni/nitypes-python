from __future__ import annotations

from abc import ABC
from typing import Any


class ScalarBase(ABC):
    """A class representing scalar data with units.

    This is an abstract base class. To create a scalar data object, use :any:`ScalarData`.
    """

    @staticmethod
    def _get_supported_dtypes() -> list[type]:
        return [bool, int, float, str]

    __slots__ = [
        "_data",
        "_units",
    ]

    _data: Any
    _units: str

    def __init__(
        self,
        data: Any,
        units: str,
    ) -> None:
        """Construct a base scalar object.

        Args:
            data: The scalar data (singular or array) to store.
            units: The units string associated with this data.

        Returns:
            A numeric waveform.
        """
        self._data = data
        self._units = units

    @property
    def data(self) -> Any:
        """The scalar data."""
        return self._data

    @property
    def units(self) -> str:
        """The data units."""
        return self._units

    def __eq__(self, value: object, /) -> bool:
        """Return self==value."""
        if not isinstance(value, self.__class__):
            return NotImplemented
        return self._data == value._data and self._units == value._units
