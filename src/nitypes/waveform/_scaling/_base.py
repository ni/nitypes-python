from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, SupportsIndex, TypeVar, overload

import numpy as np
import numpy.typing as npt

from nitypes.waveform._analog_waveform import AnalogWaveform
from nitypes.waveform._utils import validate_dtype

if TYPE_CHECKING:
    from nitypes.waveform._scaling._none import NoneScaleMode

_TRaw = TypeVar("_TRaw", bound=np.generic)
_TScaled = TypeVar("_TScaled", bound=np.generic)


_SCALED_DTYPES = (
    # Floating point
    np.single,
    np.double,
)


class ScaleMode(ABC):
    """An object that specifies how the waveform is scaled."""

    __slots__ = ()

    none: ClassVar[NoneScaleMode]

    # If dtype is not specified, _ScaledDataType defaults to np.float64.
    @overload
    def get_scaled_data(  # noqa: D107 - Missing docstring in __init__ (auto-generated noqa)
        self,
        waveform: AnalogWaveform[_TRaw],
        dtype: None = ...,
        *,
        start_index: SupportsIndex | None = ...,
        sample_count: SupportsIndex | None = ...,
    ) -> npt.NDArray[np.float64]: ...

    @overload
    def get_scaled_data(  # noqa: D107 - Missing docstring in __init__ (auto-generated noqa)
        self,
        waveform: AnalogWaveform[_TRaw],
        dtype: type[_TScaled] | np.dtype[_TScaled] = ...,
        *,
        start_index: SupportsIndex | None = ...,
        sample_count: SupportsIndex | None = ...,
    ) -> npt.NDArray[_TScaled]: ...

    @overload
    def get_scaled_data(  # noqa: D107 - Missing docstring in __init__ (auto-generated noqa)
        self,
        waveform: AnalogWaveform[Any],
        dtype: npt.DTypeLike = ...,
        *,
        start_index: SupportsIndex | None = ...,
        sample_count: SupportsIndex | None = ...,
    ) -> npt.NDArray[Any]: ...

    def get_scaled_data(
        self,
        waveform: AnalogWaveform[Any],
        dtype: npt.DTypeLike = None,
        *,
        start_index: SupportsIndex | None = 0,
        sample_count: SupportsIndex | None = None,
    ) -> npt.NDArray[_TScaled]:
        """Get scaled analog waveform data using the specified sample index and count.

        Args:
            waveform: The waveform to scale.
            dtype: The NumPy data type for the analog waveform data. If not specified, the data
                type defaults to np.float64.
            start_index: The start index.
            sample_count: The number of samples to scale.

        Returns:
            The scaled analog waveform data.
        """
        if not isinstance(waveform, AnalogWaveform):
            raise TypeError(
                "The waveform must be an AnalogWaveform object.\n\n" f"Type: {type(waveform)}"
            )

        if dtype is None:
            dtype = np.float64
        validate_dtype(dtype, _SCALED_DTYPES)

        raw_data = waveform.get_raw_data(start_index, sample_count)
        return self._transform_data(raw_data.astype(dtype))

    @abstractmethod
    def _transform_data(self, data: npt.NDArray[_TScaled]) -> npt.NDArray[_TScaled]:
        raise NotImplementedError
