"""Custom error classes for waveforms."""

from __future__ import annotations


class TimingMismatchError(RuntimeError):
    """Exception used when appending waveforms with mismatched timing."""

    pass


class CapacityMismatchError(ValueError):
    """An error for an invalid capacity."""

    pass


class CapacityTooSmallError(ValueError):
    """An error for an invalid capacity argument."""

    pass


class DatatypeMismatchError(TypeError):
    """An error for a data type mismatch."""

    pass


class IrregularTimestampCountMismatchError(ValueError):
    """An error for an irregular timestamp count mismatch."""

    pass


class StartIndexTooLargeError(ValueError):
    """Create a ValueError for an invalid start index argument."""

    pass


class StartIndexOrSampleCountTooLargeError(ValueError):
    """An error for an invalid start index or sample count argument."""

    pass


class NoTimestampInformationError(RuntimeError):
    """An error for waveform timing with no timestamp information."""

    def __init__(self) -> None:
        """Create a NoTimestampInformationError."""
        message = (
            "The waveform timing does not have valid timestamp information. "
            "To obtain timestamps, the waveform must be irregular or must be initialized "
            "with a valid time stamp and sample interval."
        )
        super().__init__(message)


class SampleIntervalModeMismatchError(TimingMismatchError):
    """An error for mixing none/regular with irregular timing."""

    pass


class SignalCountMismatchError(ValueError):
    """An error for a mismatched signal count."""

    pass
