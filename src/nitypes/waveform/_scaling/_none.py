from __future__ import annotations

import numpy.typing as npt

from nitypes.waveform._scaling._base import ScaleMode, _ScalarType


class NoneScaleMode(ScaleMode):
    """A scale mode that does not scale data."""

    __slots__ = ()

    def _transform_data(self, data: npt.NDArray[_ScalarType]) -> npt.NDArray[_ScalarType]:
        return data

    def __repr__(  # noqa: D105 - Missing docstring in magic method (auto-generated noqa)
        self,
    ) -> str:
        return f"{self.__class__.__module__}.{self.__class__.__name__}()"
