from __future__ import annotations

from typing_extensions import Literal


class TimingMismatchError(RuntimeError):
    """Exception used when appending waveforms with mismatched timing."""

    pass


class CapacityMismatchError(Exception):
    """An error for an invalid capacity."""

    def __init__(self, capacity: int, array_length: int) -> None:
        """Create a CapacityMismatchError."""
        self.message = (
            f"The capacity must match the input array length.\n\n"
            f"Capacity: {capacity}\n"
            f"Array length: {array_length}"
        )
        super().__init__(self.message)


class CapacityTooSmallError(Exception):
    """An error for an invalid capacity argument."""

    def __init__(self, capacity: int, min_capacity: int, object_description: str) -> None:
        """Create a CapacityTooSmallError."""
        self.message = (
            f"The capacity must be equal to or greater than the number of samples in the {object_description}.\n\n"
            f"Capacity: {capacity}\n"
            f"Number of samples: {min_capacity}"
        )
        super().__init__(self.message)

class DatatypeMismatchError(Exception):
    """An error for a data type mismatch."""

    def __init__(
        self,
        arg_description: Literal["input array", "input spectrum", "input waveform"],
        arg_dtype: object,
        other_description: Literal["requested", "spectrum", "waveform"],
        other_dtype: object,
    ) -> None:
        """Create a TypeError for a data type mismatch."""
        arg_key = {
            "input array": "Input array data type",
            "input spectrum": "Input spectrum data type",
            "input waveform": "Input waveform data type",
        }
        other_key = {
            "requested": "Requested data type",
            "spectrum": "Spectrum data type",
            "waveform": "Waveform data type",
        }
        self.message = (
            f"The data type of the {arg_description} must match the {other_description} data type.\n\n"
            f"{arg_key[arg_description]}: {arg_dtype}\n"
            f"{other_key[other_description]}: {other_dtype}"
        )
        super().__init__(self.message)


class IrregularTimestampCountMismatchError(Exception):
    """An error for an irregular timestamp count mismatch."""
    def __init__(
        self,
        irregular_timestamp_count: int,
        other_description: Literal["input array length", "number of samples in the waveform"],
        other: int,
        *,
        reversed: bool = False,
    ) -> None:
        """Create a IrregularTimestampCountMismatchError."""
        other_key = {
            "input array length": "Array length",
            "number of samples in the waveform": "Number of samples",
        }
        if reversed:
            self.message = (
                "The input array length must be equal to the number of irregular timestamps.\n\n"
                f"{other_key[other_description]}: {other}\n"
                f"Number of timestamps: {irregular_timestamp_count}"
            )
        else:
            self.message = (
                f"The number of irregular timestamps must be equal to the {other_description}.\n\n"
                f"Number of timestamps: {irregular_timestamp_count}\n"
                f"{other_key[other_description]}: {other}"
            )

        super().__init__(self.message)


class StartIndexTooLargeError(Exception):
    """Create a ValueError for an invalid start index argument."""

    def __init__(
        self,
        start_index: int,
        capacity_description: Literal[
            "capacity",
            "input array length",
            "number of samples in the spectrum",
            "number of samples in the waveform",
        ],
        capacity: int,
    ) -> None:
        """Create a StartIndexTooLargeError."""
        capacity_key = {
            "capacity": "Capacity",
            "input array length": "Array length",
            "number of samples in the spectrum": "Number of samples",
            "number of samples in the waveform": "Number of samples",
        }
        self.message = (
            f"The start index must be less than or equal to the {capacity_description}.\n\n"
            f"Start index: {start_index}\n"
            f"{capacity_key[capacity_description]}: {capacity}"
        )
        super().__init__(self.message)


class StartIndexOrSampleCountTooLargeError(Exception):
    """An error for an invalid start index or sample count argument."""
    def __init__(
        self,
        start_index: int,
        sample_count: int,
        capacity_description: Literal[
            "capacity",
            "input array length",
            "number of samples in the expected waveform",
            "number of samples in the spectrum",
            "number of samples in the waveform",
        ],
        capacity: int,
    ) -> None:
        """Create a StartIndexOrSampleCountTooLargeError."""
        capacity_key = {
            "capacity": "Capacity",
            "input array length": "Array length",
            "number of samples in the expected waveform": "Number of samples",
            "number of samples in the spectrum": "Number of samples",
            "number of samples in the waveform": "Number of samples",
        }
        self.message = (
            f"The sum of the start index and sample count must be less than or equal to the {capacity_description}.\n\n"
            f"Start index: {start_index}\n"
            f"Sample count: {sample_count}\n"
            f"{capacity_key[capacity_description]}: {capacity}"
        )
        super().__init__(self.message)


class NoTimestampInformationError(Exception):
    """An error for waveform timing with no timestamp information."""

    def __init__(self) -> None:
        """Create a NoTimestampInformationError."""
        self.message = (
            "The waveform timing does not have valid timestamp information. "
            "To obtain timestamps, the waveform must be irregular or must be initialized "
            "with a valid time stamp and sample interval."
        )
        super().__init__(self.message)

class SampleIntervalModeMismatchError(Exception):
    """An error for mixing none/regular with irregular timing."""

    def __init__(self) -> None:
        """Create a SampleIntervalModeMismatchError."""
        self.message = (
            "The timing of one or more waveforms does not match the timing of the current waveform."
        )
        super().__init__(self.message)


class SignalCountMismatchError(Exception):
    def SignalCountMismatchError(
        self,
        arg_description: Literal["expected waveform", "input array", "input waveform", "provided"],
        arg_signal_count: int,
        other_description: Literal["array", "port", "waveform"],
        other_signal_count: int,
    ) -> None:
        """Create a ValueError for an mismatched signal count."""
        arg_key = {
            "expected waveform": "Expected waveform signal count",
            "input array": "Input array signal count",
            "input waveform": "Input waveform signal count",
            "provided": "Signal count",
        }
        other_key = {
            "array": "Array signal count",
            "port": "Port signal count",
            "waveform": "Waveform signal count",
        }
        self.message = (
            f"The {arg_description} signal count must match the {other_description} signal count.\n\n"
            f"{arg_key[arg_description]}: {arg_signal_count}\n"
            f"{other_key[other_description]}: {other_signal_count}"
        )
        super().__init__(self.message)
