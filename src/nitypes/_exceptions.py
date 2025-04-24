from __future__ import annotations

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
