from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from nitypes.waveform import AnalogWaveform


@pytest.mark.parametrize("copy", [False, True])
def test___memmap_array_1d___create_waveform_from_array___waveform_contains_memmap_data(
    tmp_path: Path, copy: bool
) -> None:
    memmap_path = tmp_path / "memmap_array.bin"
    memmap_path.write_bytes(struct.pack("4d", 1.23, -4.56, 7e89, 1e-23))
    memmap_array = np.memmap(memmap_path, np.float64)

    waveform = AnalogWaveform.from_array_1d(memmap_array, copy=copy)

    assert list(waveform.raw_data) == [1.23, -4.56, 7e89, 1e-23]


@pytest.mark.parametrize("copy", [False, True])
def test___memmap_array_2d___create_waveforms_from_array___waveforms_contains_memmap_data(
    tmp_path: Path, copy: bool
) -> None:
    memmap_path = tmp_path / "memmap_array.bin"
    memmap_path.write_bytes(struct.pack("6d", 1.23, -4.56, 7e89, 1e-23, 456.0, 7.89))
    memmap_array = np.memmap(memmap_path, np.float64, shape=(2, 3))

    waveforms = AnalogWaveform.from_array_2d(memmap_array, copy=copy)
    assert memmap_array.shape == (2, 3)

    assert len(waveforms) == 2
    assert list(waveforms[0].raw_data) == [1.23, -4.56, 7e89]
    assert list(waveforms[1].raw_data) == [1e-23, 456.0, 7.89]


def test___memmap_waveform___append___waveform_writes_to_memmap(tmp_path: Path) -> None:
    memmap_path = tmp_path / "memmap_array.bin"
    memmap_array = np.memmap(memmap_path, np.float64, "w+", shape=10)
    waveform = AnalogWaveform.from_array_1d(memmap_array, copy=False, sample_count=0)

    waveform.append(np.array([1.23, -4.56, 7e89, 1e-23]))
    memmap_array.flush()

    memmap_bytes = memmap_path.read_bytes()
    memmap_data = struct.unpack_from("4d", memmap_bytes)
    assert memmap_data == (1.23, -4.56, 7e89, 1e-23)


def test___memmap_waveforms___append___waveforms_write_to_memmap(tmp_path: Path) -> None:
    memmap_path = tmp_path / "memmap_array.bin"
    memmap_array = np.memmap(memmap_path, np.float64, "w+", shape=(2, 10))
    waveforms = AnalogWaveform.from_array_2d(memmap_array, copy=False, sample_count=0)

    waveforms[0].append(np.array([1.23, -4.56, 7e89]))
    waveforms[1].append(np.array([1e-23, 456.0, 7.89]))
    memmap_array.flush()

    memmap_bytes = memmap_path.read_bytes()
    memmap_data0 = struct.unpack_from("3d", memmap_bytes, offset=0)
    memmap_data1 = struct.unpack_from("3d", memmap_bytes, offset=80)
    assert memmap_data0 == (1.23, -4.56, 7e89)
    assert memmap_data1 == (1e-23, 456.0, 7.89)
