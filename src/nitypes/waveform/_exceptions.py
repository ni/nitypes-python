from __future__ import annotations


class TimingMismatchError(RuntimeError):
    """Exception used when appending waveforms with mismatched timing."""

    pass


def input_array_data_type_mismatch(input_dtype: object, waveform_dtype: object) -> TypeError:
    """Create a TypeError for an input array data type mismatch."""
    return TypeError(
        "The data type of the input array must match the waveform data type.\n\n"
        f"Input array data type: {input_dtype}\n"
        f"Waveform data type: {waveform_dtype}"
    )


def input_spectrum_data_type_mismatch(input_dtype: object, spectrum_dtype: object) -> TypeError:
    """Create a TypeError for an input spectrum data type mismatch."""
    return TypeError(
        "The data type of the input spectrum must match the spectrum data type.\n\n"
        f"Input spectrum data type: {input_dtype}\n"
        f"Spectrum data type: {spectrum_dtype}"
    )


def input_waveform_data_type_mismatch(input_dtype: object, waveform_dtype: object) -> TypeError:
    """Create a TypeError for an input waveform data type mismatch."""
    return TypeError(
        "The data type of the input waveform must match the waveform data type.\n\n"
        f"Input waveform data type: {input_dtype}\n"
        f"Waveform data type: {waveform_dtype}"
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
