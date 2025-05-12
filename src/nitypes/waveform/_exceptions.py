from __future__ import annotations

from typing_extensions import Literal


class TimingMismatchError(RuntimeError):
    """Exception used when appending waveforms with mismatched timing."""

    pass


def capacity_mismatch(capacity: int, array_length: int) -> ValueError:
    """Create a ValueError for an invalid capacity."""
    return ValueError(
        f"The capacity must match the input array length.\n\n"
        f"Capacity: {capacity}\n"
        f"Array length: {array_length}"
    )


def capacity_too_small(capacity: int, min_capacity: int, object_description: str) -> ValueError:
    """Create a ValueError for an invalid capacity argument."""
    return ValueError(
        f"The capacity must be equal to or greater than the number of samples in the {object_description}.\n\n"
        f"Capacity: {capacity}\n"
        f"Number of samples: {min_capacity}"
    )


def data_type_mismatch(
    arg_description: Literal["input array", "input spectrum", "input waveform"],
    arg_dtype: object,
    other_description: Literal["requested", "spectrum", "waveform"],
    other_dtype: object,
) -> TypeError:
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
    return TypeError(
        f"The data type of the {arg_description} must match the {other_description} data type.\n\n"
        f"{arg_key[arg_description]}: {arg_dtype}\n"
        f"{other_key[other_description]}: {other_dtype}"
    )


def irregular_timestamp_count_mismatch(
    irregular_timestamp_count: int,
    other_description: Literal["input array length", "number of samples in the waveform"],
    other: int,
    *,
    reversed: bool = False,
) -> ValueError:
    """Create a ValueError for an irregular timestamp count mismatch."""
    other_key = {
        "input array length": "Array length",
        "number of samples in the waveform": "Number of samples",
    }
    if reversed:
        raise ValueError(
            "The input array length must be equal to the number of irregular timestamps.\n\n"
            f"{other_key[other_description]}: {other}\n"
            f"Number of timestamps: {irregular_timestamp_count}"
        )
    else:
        raise ValueError(
            f"The number of irregular timestamps must be equal to the {other_description}.\n\n"
            f"Number of timestamps: {irregular_timestamp_count}\n"
            f"{other_key[other_description]}: {other}"
        )


def start_index_too_large(
    start_index: int,
    capacity_description: Literal[
        "capacity",
        "input array length",
        "number of samples in the spectrum",
        "number of samples in the waveform",
    ],
    capacity: int,
) -> ValueError:
    """Create a ValueError for an invalid start index argument."""
    capacity_key = {
        "capacity": "Capacity",
        "input array length": "Array length",
        "number of samples in the spectrum": "Number of samples",
        "number of samples in the waveform": "Number of samples",
    }
    return ValueError(
        f"The start index must be less than or equal to the {capacity_description}.\n\n"
        f"Start index: {start_index}\n"
        f"{capacity_key[capacity_description]}: {capacity}"
    )


def start_index_or_sample_count_too_large(
    start_index: int,
    sample_count: int,
    capacity_description: Literal[
        "capacity",
        "input array length",
        "number of samples in the spectrum",
        "number of samples in the waveform",
    ],
    capacity: int,
) -> ValueError:
    """Create a ValueError for an invalid start index or sample count argument."""
    capacity_key = {
        "capacity": "Capacity",
        "input array length": "Array length",
        "number of samples in the spectrum": "Number of samples",
        "number of samples in the waveform": "Number of samples",
    }
    return ValueError(
        f"The sum of the start index and sample count must be less than or equal to the {capacity_description}.\n\n"
        f"Start index: {start_index}\n"
        f"Sample count: {sample_count}\n"
        f"{capacity_key[capacity_description]}: {capacity}"
    )


def no_timestamp_information() -> RuntimeError:
    """Create a RuntimeError for waveform timing with no timestamp information."""
    return RuntimeError(
        "The waveform timing does not have valid timestamp information. "
        "To obtain timestamps, the waveform must be irregular or must be initialized "
        "with a valid time stamp and sample interval."
    )


def sample_interval_mode_mismatch() -> TimingMismatchError:
    """Create a TimingMismatchError about mixing none/regular with irregular timing."""
    return TimingMismatchError(
        "The timing of one or more waveforms does not match the timing of the current waveform."
    )
