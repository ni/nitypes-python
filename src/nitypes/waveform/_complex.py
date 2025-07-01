from __future__ import annotations

from collections.abc import Sequence
from typing import Any, SupportsIndex, Union, overload

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeVar, Unpack, final, override

from nitypes.complex import ComplexInt32Base, ComplexInt32DType, convert_complex
from nitypes.waveform._numeric import NumericWaveform, _TOtherScaled
from nitypes.waveform._options import WaveformOptions

# _TRaw specifies the type of the raw_data array. ComplexWaveform accepts a narrower set of types
# than NumericWaveform. Note that ComplexInt32Base is an alias for np.void, but other structured
# data types are rejected at run time.
_TRaw = TypeVar("_TRaw", bound=Union[np.complexfloating, ComplexInt32Base])
_TOtherRaw = TypeVar("_TOtherRaw", bound=Union[np.complexfloating, ComplexInt32Base])

_RAW_DTYPES = (
    # Complex floating point
    np.csingle,
    np.cdouble,
    # Complex integers
    ComplexInt32DType,
)

_SCALED_DTYPES = (
    # Complex floating point
    np.csingle,
    np.cdouble,
)


@final
class ComplexWaveform(NumericWaveform[_TRaw, np.complex128]):
    """A complex waveform, which encapsulates complex data and timing information."""

    @override
    @staticmethod
    def _get_default_raw_dtype() -> type[np.generic] | np.dtype[np.generic]:
        return np.complex128

    @override
    @staticmethod
    def _get_default_scaled_dtype() -> type[np.generic] | np.dtype[np.generic]:
        return np.complex128

    @override
    @staticmethod
    def _get_supported_raw_dtypes() -> tuple[npt.DTypeLike, ...]:
        return _RAW_DTYPES

    @override
    @staticmethod
    def _get_supported_scaled_dtypes() -> tuple[npt.DTypeLike, ...]:
        return _SCALED_DTYPES

    @overload
    @classmethod
    def from_array_1d(
        cls,
        array: npt.NDArray[_TOtherRaw],
        dtype: None = ...,
        *,
        copy: bool = ...,
        start_index: SupportsIndex | None = ...,
        sample_count: SupportsIndex | None = ...,
        **kwargs: Unpack[WaveformOptions],
    ) -> ComplexWaveform[_TOtherRaw]: ...

    @overload
    @classmethod
    def from_array_1d(
        cls,
        array: npt.NDArray[Any] | Sequence[Any],
        dtype: type[_TOtherRaw] | np.dtype[_TOtherRaw],
        *,
        copy: bool = ...,
        start_index: SupportsIndex | None = ...,
        sample_count: SupportsIndex | None = ...,
        **kwargs: Unpack[WaveformOptions],
    ) -> ComplexWaveform[_TOtherRaw]: ...

    @overload
    @classmethod
    def from_array_1d(
        cls,
        array: npt.NDArray[Any] | Sequence[Any],
        dtype: npt.DTypeLike = ...,
        *,
        copy: bool = ...,
        start_index: SupportsIndex | None = ...,
        sample_count: SupportsIndex | None = ...,
        **kwargs: Unpack[WaveformOptions],
    ) -> ComplexWaveform[Any]: ...

    @override
    @classmethod
    def from_array_1d(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        array: npt.NDArray[Any] | Sequence[Any],
        dtype: npt.DTypeLike = None,
        *,
        copy: bool = True,
        start_index: SupportsIndex | None = 0,
        sample_count: SupportsIndex | None = None,
        **kwargs: Unpack[WaveformOptions],
    ) -> ComplexWaveform[Any]:
        """Construct a complex waveform from a one-dimensional array or sequence.

        Args:
            array: The waveform data as a one-dimensional array or a sequence.
            dtype: The NumPy data type for the waveform data. This argument is required
                when array is a sequence.
            copy: Specifies whether to copy the array or save a reference to it.
            start_index: The sample index at which the waveform data begins.
            sample_count: The number of samples in the waveform.
            kwargs: Waveform options saved in a :any:`WaveformOptions` object.

        Returns:
            A complex waveform containing the specified data.
        """
        return super().from_array_1d(
            array,
            dtype,
            **kwargs,
            copy=copy,
            start_index=start_index,
            sample_count=sample_count,
        )

    @overload
    @classmethod
    def from_array_2d(
        cls,
        array: npt.NDArray[_TOtherRaw],
        dtype: None = ...,
        *,
        copy: bool = ...,
        start_index: SupportsIndex | None = ...,
        sample_count: SupportsIndex | None = ...,
        **kwargs: Unpack[WaveformOptions],
    ) -> Sequence[ComplexWaveform[_TOtherRaw]]: ...

    @overload
    @classmethod
    def from_array_2d(
        cls,
        array: npt.NDArray[Any] | Sequence[Sequence[Any]],
        dtype: type[_TOtherRaw] | np.dtype[_TOtherRaw],
        *,
        copy: bool = ...,
        start_index: SupportsIndex | None = ...,
        sample_count: SupportsIndex | None = ...,
        **kwargs: Unpack[WaveformOptions],
    ) -> Sequence[ComplexWaveform[_TOtherRaw]]: ...

    @overload
    @classmethod
    def from_array_2d(
        cls,
        array: npt.NDArray[Any] | Sequence[Sequence[Any]],
        dtype: npt.DTypeLike = ...,
        *,
        copy: bool = ...,
        start_index: SupportsIndex | None = ...,
        sample_count: SupportsIndex | None = ...,
        **kwargs: Unpack[WaveformOptions],
    ) -> Sequence[ComplexWaveform[Any]]: ...

    @override
    @classmethod
    def from_array_2d(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        array: npt.NDArray[Any] | Sequence[Sequence[Any]],
        dtype: npt.DTypeLike = None,
        *,
        copy: bool = True,
        start_index: SupportsIndex | None = 0,
        sample_count: SupportsIndex | None = None,
        **kwargs: Unpack[WaveformOptions],
    ) -> Sequence[ComplexWaveform[Any]]:
        """Construct multiple complex waveforms from a two-dimensional array or nested sequence.

        Args:
            array: The waveform data as a two-dimensional array or a nested sequence.
            dtype: The NumPy data type for the waveform data. This argument is required
                when array is a sequence.
            copy: Specifies whether to copy the array or save a reference to it.
            start_index: The sample index at which the waveform data begins.
            sample_count: The number of samples in the waveform.
            kwargs: Waveform options saved in a :any:`WaveformOptions` object.

        Returns:
            A sequence containing a complex waveform for each row of the specified data.

        When constructing multiple waveforms, the same extended properties, timing
        information, and scale mode are applied to all waveforms. Consider assigning
        these properties after construction.
        """
        return super().from_array_2d(
            array,
            dtype,
            **kwargs,
            copy=copy,
            start_index=start_index,
            sample_count=sample_count,
        )

    __slots__ = ()

    # If neither dtype nor raw_data is specified, _TRaw defaults to np.complex128.
    @overload
    def __init__(  # noqa: D107 - Missing docstring in __init__ (auto-generated noqa)
        self: ComplexWaveform[np.complex128],
        sample_count: SupportsIndex | None = ...,
        dtype: None = ...,
        *,
        raw_data: None = ...,
        start_index: SupportsIndex | None = ...,
        capacity: SupportsIndex | None = ...,
        **kwargs: Unpack[WaveformOptions],
    ) -> None: ...

    @overload
    def __init__(  # noqa: D107 - Missing docstring in __init__ (auto-generated noqa)
        self: ComplexWaveform[_TOtherRaw],
        sample_count: SupportsIndex | None = ...,
        dtype: type[_TOtherRaw] | np.dtype[_TOtherRaw] = ...,
        *,
        raw_data: None = ...,
        start_index: SupportsIndex | None = ...,
        capacity: SupportsIndex | None = ...,
        **kwargs: Unpack[WaveformOptions],
    ) -> None: ...

    @overload
    def __init__(  # noqa: D107 - Missing docstring in __init__ (auto-generated noqa)
        self: ComplexWaveform[_TOtherRaw],
        sample_count: SupportsIndex | None = ...,
        dtype: None = ...,
        *,
        raw_data: npt.NDArray[_TOtherRaw] = ...,
        start_index: SupportsIndex | None = ...,
        capacity: SupportsIndex | None = ...,
        **kwargs: Unpack[WaveformOptions],
    ) -> None: ...

    @overload
    def __init__(  # noqa: D107 - Missing docstring in __init__ (auto-generated noqa)
        self: ComplexWaveform[Any],
        sample_count: SupportsIndex | None = ...,
        dtype: npt.DTypeLike = ...,
        *,
        raw_data: npt.NDArray[Any] | None = ...,
        start_index: SupportsIndex | None = ...,
        capacity: SupportsIndex | None = ...,
        **kwargs: Unpack[WaveformOptions],
    ) -> None: ...

    def __init__(
        self,
        sample_count: SupportsIndex | None = None,
        dtype: npt.DTypeLike = None,
        *,
        raw_data: npt.NDArray[Any] | None = None,
        start_index: SupportsIndex | None = None,
        capacity: SupportsIndex | None = None,
        **kwargs: Unpack[WaveformOptions],
    ) -> None:
        """Initialize a new complex waveform.

        Args:
            start_index: The sample index at which the waveform data begins.
            sample_count: The number of samples in the waveform.
            capacity: The number of samples to allocate. Pre-allocating a larger buffer optimizes
                appending samples to the waveform.
            dtype: The NumPy data type for the waveform data. If not specified, the data
                type defaults to np.complex128.
            raw_data: A NumPy ndarray to use for sample storage. The waveform takes ownership
                of this array. If not specified, an ndarray is created based on the specified dtype,
                start index, sample count, and capacity.
            kwargs: Waveform options saved in a :any:`WaveformOptions` object.

        Returns:
            A complex waveform.
        """
        return super().__init__(
            sample_count,
            dtype,
            **kwargs,
            raw_data=raw_data,
            start_index=start_index,
            capacity=capacity,
        )

    @override
    def _convert_data(
        self,
        dtype: npt.DTypeLike | type[_TOtherScaled] | np.dtype[_TOtherScaled],
        raw_data: npt.NDArray[_TRaw],
    ) -> npt.NDArray[_TOtherScaled]:
        return convert_complex(dtype, raw_data)
