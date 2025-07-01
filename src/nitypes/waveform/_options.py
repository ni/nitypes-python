from __future__ import annotations

from collections.abc import Mapping

from typing_extensions import TypedDict

from nitypes.waveform._extended_properties import ExtendedPropertyValue
from nitypes.waveform._scaling import ScaleMode
from nitypes.waveform._timing import Timing, _AnyDateTime, _AnyTimeDelta


class WaveformOptions(TypedDict, total=False):
    """Waveform options as a typed dictionary."""

    extended_properties: Mapping[str, ExtendedPropertyValue] | None
    """The extended properties of the waveform."""

    copy_extended_properties: bool
    """Specifies whether to copy the extended properties or take ownership."""

    timing: Timing[_AnyDateTime, _AnyTimeDelta, _AnyTimeDelta] | None
    """The timing information of the waveform."""

    scale_mode: ScaleMode | None
    """The scale mode of the waveform."""
