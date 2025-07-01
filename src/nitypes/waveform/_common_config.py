from __future__ import annotations

from collections.abc import Mapping
from typing import SupportsIndex

from typing_extensions import TypedDict

from nitypes.waveform._extended_properties import ExtendedPropertyValue
from nitypes.waveform._scaling import ScaleMode
from nitypes.waveform._timing import Timing, _AnyDateTime, _AnyTimeDelta


class CommonWaveformConfig(TypedDict, total=False):
    """Common waveform configuration as a typed dictionary."""

    start_index: SupportsIndex | None
    capacity: SupportsIndex | None
    extended_properties: Mapping[str, ExtendedPropertyValue] | None
    copy_extended_properties: bool
    timing: Timing[_AnyDateTime, _AnyTimeDelta, _AnyTimeDelta] | None
    scale_mode: ScaleMode | None
