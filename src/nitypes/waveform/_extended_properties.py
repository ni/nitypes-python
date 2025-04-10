from __future__ import annotations

import operator
import sys
from typing import MutableMapping, Iterator

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

ExtendedPropertyValue: TypeAlias = bool | float | int | str
"""An ExtendedPropertyDictionary value."""


class ExtendedPropertyDictionary(MutableMapping[str, ExtendedPropertyValue]):
    """A dictionary of extended properties."""

    def __init__(self) -> None:
        """Construct an ExtendedPropertyDictionary."""
        self._properties: dict[str, ExtendedPropertyValue] = {}

    def __len__(  # noqa: D105 - Missing docstring in magic method (auto-generated noqa)
        self,
    ) -> int:
        return len(self._properties)

    def __iter__(  # noqa: D105 - Missing docstring in magic method (auto-generated noqa)
        self,
    ) -> Iterator[str]:
        return iter(self._properties)

    def __contains__(  # noqa: D105 - Missing docstring in magic method (auto-generated noqa)
        self, x: object, /
    ) -> bool:
        return operator.contains(self._properties, x)

    def __getitem__(  # noqa: D105 - Missing docstring in magic method (auto-generated noqa)
        self, key: str, /
    ) -> ExtendedPropertyValue:
        return operator.getitem(self._properties, key)

    def __setitem__(  # noqa: D105 - Missing docstring in magic method (auto-generated noqa)
        self, key: str, value: ExtendedPropertyValue, /
    ) -> None:
        operator.setitem(self._properties, key, value)

    def __delitem__(  # noqa: D105 - Missing docstring in magic method (auto-generated noqa)
        self, key: str, /
    ) -> None:
        operator.delitem(self._properties, key)
