from __future__ import annotations

import reprlib
import sys


def add_note(exception: Exception, note: str) -> None:
    """Add a note to an exception.

    >>> try:
    ...     raise ValueError("Oh no")
    ... except Exception as e:
    ...     add_note(e, "p.s. This is bad")
    ...     raise
    Traceback (most recent call last):
    ...
    ValueError: Oh no
    p.s. This is bad
    """
    if sys.version_info >= (3, 11):
        exception.add_note(note)
    else:
        message = exception.args[0] + "\n" + note
        exception.args = (message,) + exception.args[1:]


def invalid_arg_value(
    arg_description: str, valid_value_description: str, value: object
) -> ValueError:
    """Create a ValueError for an invalid argument value."""
    return ValueError(
        f"The {arg_description} must be {_a(valid_value_description)}.\n\n"
        f"Provided value: {reprlib.repr(value)}"
    )


def invalid_arg_type(arg_description: str, type_description: str, value: object) -> TypeError:
    """Create a TypeError for an invalid argument type."""
    return TypeError(
        f"The {arg_description} must be {_a(type_description)}.\n\n"
        f"Provided value: {reprlib.repr(value)}"
    )


def invalid_array_ndim(arg_description: str, valid_value_description: str, ndim: int) -> ValueError:
    """Create a ValueError for an array with an invalid number of dimensions."""
    return ValueError(
        f"The {arg_description} must be {_a(valid_value_description)}.\n\n"
        f"Number of dimensions: {ndim}"
    )


def invalid_requested_type(type_description: str, requested_type: type) -> TypeError:
    """Create a TypeError for an invalid requested type."""
    return TypeError(
        f"The requested type must be {_a(type_description)} type.\n\n"
        f"Requested type: {requested_type}"
    )


def no_timestamp_information() -> RuntimeError:
    """Create a RuntimeError for waveform timing with no timestamp information."""
    return RuntimeError(
        "The waveform timing does not have valid timestamp information. "
        "To obtain timestamps, the waveform must be irregular or must be initialized "
        "with a valid time stamp and sample interval."
    )


def sample_interval_mode_mismatch() -> RuntimeError:
    """Create a RuntimeError about mixing none/regular with irregular timing."""
    return RuntimeError(
        "The timing of one or more waveforms does not match the timing of the current waveform."
    )


def unsupported_arg(arg_description: str, value: object) -> ValueError:
    """Create a ValueError for an unsupported argument."""
    return ValueError(
        f"The {arg_description} argument is not supported.\n\n"
        f"Provided value: {reprlib.repr(value)}"
    )


# English-specific hack. This is why we prefer "Key: value" for localizable errors. TODO: consider
# moving the full strings into a string table instead of building them out of English noun phrases.
def _a(noun: str) -> str:
    indefinite_article = "an" if noun[0] in "AEIOUaeiou" else "a"
    if noun.startswith("one-"):
        indefinite_article = "a"
    return f"{indefinite_article} {noun}"
