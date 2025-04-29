from __future__ import annotations


class TimingMismatchWarning(RuntimeWarning):
    """Warning used when appending waveforms with mismatched timing."""

    pass


def sample_interval_mismatch() -> TimingMismatchWarning:
    """Create a TimingMismatchWarning about appending waveforms with mismatched sample intervals."""
    return TimingMismatchWarning(
        "The sample interval of one or more waveforms does not match the sample interval of the current waveform."
    )
