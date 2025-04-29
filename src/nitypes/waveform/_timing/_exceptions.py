from __future__ import annotations


class TimingMismatchError(RuntimeError):
    """Exception used when appending waveforms with mismatched timing."""

    pass


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
