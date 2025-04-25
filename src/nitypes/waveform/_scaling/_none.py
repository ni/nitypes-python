from __future__ import annotations

import numpy.typing as npt

from nitypes.waveform._scaling._base import ScaleMode, _ScalarType


class NoneScaleMode(ScaleMode):
    """A scale mode that does not scale data."""

    __slots__ = ()

    def _transform_data(self, data: npt.NDArray[_ScalarType]) -> npt.NDArray[_ScalarType]:
        return data

    def __eq__(self, value: object, /) -> bool:
        """Return self==value."""
        if not isinstance(value, self.__class__):
            return NotImplemented
        return True

    def __repr__(
        self,
    ) -> str:
        """Return repr(self)."""
        return f"{self.__class__.__module__}.{self.__class__.__name__}()"
