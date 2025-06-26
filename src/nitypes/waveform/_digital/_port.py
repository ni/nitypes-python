from __future__ import annotations

import sys
from typing import TypeVar, Union

import numpy as np
import numpy.typing as npt

AnyPort = Union[np.uint8, np.uint16, np.uint32]
TPort = TypeVar("TPort", bound=AnyPort)

DIGITAL_PORT_DTYPES = (np.uint8, np.uint16, np.uint32)


def bit_count(value: int, /) -> int:
    """Return the number of 1 bits in value.

    >>> bit_count(0xFFFF)
    16
    >>> bit_count(0x1084)
    3
    """
    return _bit_count(value)


if sys.version_info >= (3, 10):

    def _bit_count(value: int) -> int:
        return value.bit_count()

else:

    def _bit_count(value: int) -> int:
        return bin(value).count("1")


def bit_mask(n: int, /) -> int:
    """Return the bit mask with the lower n bits set.

    >>> bit_mask(0)
    0
    >>> bit_mask(4)
    15
    >>> bit_mask(9)
    511
    >>> bit_mask(32)
    4294967295
    >>> bit_mask(-1)
    Traceback (most recent call last):
    ...
    ValueError: The number of bits must be a non-negative integer.
    <BLANKLINE>
    Number of bits: -1
    """
    if n < 0:
        raise ValueError(
            "The number of bits must be a non-negative integer.\n\n" f"Number of bits: {n}"
        )
    return (1 << n) - 1


def get_port_dtype(mask: int) -> np.dtype[AnyPort]:
    """Return the NumPy port dtype for the given mask.

    >>> get_port_dtype(0xF)
    dtype('uint8')
    >>> get_port_dtype(0x100)
    dtype('uint16')
    >>> get_port_dtype(0xDEADBEEF)
    dtype('uint32')
    >>> get_port_dtype(0x1_0000_0000)
    Traceback (most recent call last):
    ...
    ValueError: The mask must be an unsigned 8-, 16-, or 32-bit integer.
    <BLANKLINE>
    Mask: 4294967296
    """
    if (mask & 0xFF) == mask:
        return np.dtype(np.uint8)
    elif (mask & 0xFFFF) == mask:
        return np.dtype(np.uint16)
    elif (mask & 0xFFFFFFFF) == mask:
        return np.dtype(np.uint32)
    else:
        raise ValueError(
            "The mask must be an unsigned 8-, 16-, or 32-bit integer.\n\n" f"Mask: {mask}"
        )


def port_to_line_data(port_data: npt.NDArray[AnyPort], mask: int) -> npt.NDArray[np.uint8]:
    """Convert a 1D array of port data to a 2D array of line data, using the specified mask.

    >>> port_to_line_data(np.array([0,1,2,3], np.uint8), 0xFF)
    array([[0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0],
           [1, 1, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    >>> port_to_line_data(np.array([0,1,2,3], np.uint8), 0x3)
    array([[0, 0],
           [1, 0],
           [0, 1],
           [1, 1]], dtype=uint8)
    >>> port_to_line_data(np.array([0,1,2,3], np.uint8), 0x2)
    array([[0],
           [0],
           [1],
           [1]], dtype=uint8)
    >>> port_to_line_data(np.array([0,1,2,3], np.uint8), 0)
    array([], shape=(4, 0), dtype=uint8)
    >>> port_to_line_data(np.array([0x12000000,0xFE000000], np.uint32), 0xFF000000)
    array([[0, 1, 0, 0, 1, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 1, 1]], dtype=uint8)
    """
    port_size = port_data.dtype.itemsize * 8
    line_data = np.unpackbits(port_data.view(np.uint8), bitorder="little")
    line_data = line_data.reshape(len(port_data), port_size)

    if mask == bit_mask(port_size):
        return line_data
    else:
        return line_data[:, _mask_to_line_indices(mask)]


def _mask_to_line_indices(mask: int) -> list[int]:
    """Return the line indices for the given mask.

    >>> _mask_to_line_indices(0xF)
    [0, 1, 2, 3]
    >>> _mask_to_line_indices(0x100)
    [8]
    >>> _mask_to_line_indices(0xDEADBEEF)
    [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 18, 19, 21, 23, 25, 26, 27, 28, 30, 31]
    >>> _mask_to_line_indices(-1)
    Traceback (most recent call last):
    ...
    ValueError: The mask must be a non-negative integer.
    <BLANKLINE>
    Mask: -1
    """
    if mask < 0:
        raise ValueError("The mask must be a non-negative integer.\n\n" f"Mask: {mask}")
    line_indices = []
    line_index = 0
    while mask != 0:
        if mask & 1:
            line_indices.append(line_index)
        line_index += 1
        mask >>= 1
    return line_indices
