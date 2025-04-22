from __future__ import annotations

import numpy as np
import numpy.typing as npt

from nitypes.waveform._scaling._base import ScaleMode, _TScaled


class LinearScaleMode(ScaleMode):
    """A scale mode that scales data linearly."""

    __slots__ = ["_gain", "_offset", "__weakref__"]

    _gain: float
    _offset: float

    def __init__(self, gain: float, offset: float) -> None:
        """Construct a scale mode object that scales data linearly.

        Args:
            gain: The gain of the linear scale.
            offset: The offset of the linear scale.

        Returns:
            A scale mode that scales data linearly.
        """
        if not isinstance(gain, (float, int)):
            raise TypeError(
                "The gain must be a floating point number.\n\n" f"Provided value: {gain}"
            )
        if not isinstance(offset, (float, int)):
            raise TypeError(
                "The offset must be a floating point number.\n\n" f"Provided value: {offset}"
            )
        self._gain = gain
        self._offset = offset

    @property
    def gain(self) -> float:
        """The gain of the linear scale."""
        return self._gain

    @property
    def offset(self) -> float:
        """The offset of the linear scale."""
        return self._offset

    def _transform_data(self, data: npt.NDArray[_TScaled]) -> npt.NDArray[_TScaled]:
        # TODO: are numpy's __mul__ and __add__ operators missing overloads for np.float32?
        gain = np.array(self.gain, data.dtype)
        offset = np.array(self.offset, data.dtype)
        return data * gain + offset

    def __repr__(  # noqa: D105 - Missing docstring in magic method (auto-generated noqa)
        self,
    ) -> str:
        return f"{self.__class__.__module__}.{self.__class__.__name__}({self.gain}, {self.offset})"
