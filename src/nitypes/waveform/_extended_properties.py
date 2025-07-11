from __future__ import annotations

import operator
from collections.abc import Iterator, Mapping, MutableMapping
from typing import Union

from typing_extensions import TypeAlias

# Extended property keys
CHANNEL_NAME = "NI_ChannelName"
LINE_NAMES = "NI_LineNames"
UNIT_DESCRIPTION = "NI_UnitDescription"


ExtendedPropertyValue: TypeAlias = Union[bool, float, int, str]
"""An ExtendedPropertyDictionary value."""


class ExtendedPropertyDictionary(MutableMapping[str, ExtendedPropertyValue]):
    """A dictionary of extended properties.

    .. note::
        Data stored in the extended properties dictionary may not be encrypted when you send it
        over the network or write it to a TDMS file.
    """

    __slots__ = ["_properties"]

    def __init__(self, properties: Mapping[str, ExtendedPropertyValue] | None = None, /) -> None:
        """Initialize a new ExtendedPropertyDictionary."""
        self._properties: dict[str, ExtendedPropertyValue] = {}
        if properties is not None:
            self._properties.update(properties)

    def __len__(self) -> int:
        """Return len(self)."""
        return len(self._properties)

    def __iter__(self) -> Iterator[str]:
        """Implement iter(self)."""
        return iter(self._properties)

    def __contains__(self, value: object, /) -> bool:
        """Implement value in self."""
        return operator.contains(self._properties, value)

    def __getitem__(self, key: str, /) -> ExtendedPropertyValue:
        """Get self[key]."""
        return operator.getitem(self._properties, key)

    def __setitem__(self, key: str, value: ExtendedPropertyValue, /) -> None:
        """Set self[key] to value."""
        operator.setitem(self._properties, key, value)

    def __delitem__(self, key: str, /) -> None:
        """Delete self[key]."""
        operator.delitem(self._properties, key)

    def _merge(self, other: ExtendedPropertyDictionary) -> None:
        for key, value in other.items():
            self._properties.setdefault(key, value)

    def __repr__(self) -> str:
        """Return repr(self)."""
        return f"{self.__class__.__module__}.{self.__class__.__name__}({self._properties!r})"
