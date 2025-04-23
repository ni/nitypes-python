"""Waveform scaling data types for NI Python APIs."""

from nitypes.waveform._scaling._base import ScaleMode
from nitypes.waveform._scaling._none import NoneScaleMode

# Defined here to avoid a circular dependency
ScaleMode.none = NoneScaleMode()
"""A scale mode that does not scale data."""
