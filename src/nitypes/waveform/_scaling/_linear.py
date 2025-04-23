from __future__ import annotations

from typing import SupportsFloat

import numpy.typing as npt

from nitypes.waveform._scaling._base import ScaleMode, _TScaled
from nitypes.waveform._utils import arg_to_float


class LinearScaleMode(ScaleMode):
    """A scale mode that scales data linearly."""

    __slots__ = ["_gain", "_offset", "__weakref__"]

    _gain: float
    _offset: float

    def __init__(self, gain: SupportsFloat, offset: SupportsFloat) -> None:
        """Construct a scale mode object that scales data linearly.

        Args:
            gain: The gain of the linear scale.
            offset: The offset of the linear scale.

        Returns:
            A scale mode that scales data linearly.
        """
        self._gain = arg_to_float("gain", gain)
        self._offset = arg_to_float("offset", offset)

    @property
    def gain(self) -> float:
        """The gain of the linear scale."""
        return self._gain

    @property
    def offset(self) -> float:
        """The offset of the linear scale."""
        return self._offset

    def _transform_data(self, data: npt.NDArray[_TScaled]) -> npt.NDArray[_TScaled]:
        # https://github.com/numpy/numpy/issues/28805 - TYP: mypy infers that adding/multiplying a
        # npt.NDArray[np.float32] with a float promotes dtype to Any or np.float64
        return data * self._gain + self._offset  # type: ignore[operator,no-any-return]

    def __repr__(  # noqa: D105 - Missing docstring in magic method (auto-generated noqa)
        self,
    ) -> str:
        return f"{self.__class__.__module__}.{self.__class__.__name__}({self.gain}, {self.offset})"
