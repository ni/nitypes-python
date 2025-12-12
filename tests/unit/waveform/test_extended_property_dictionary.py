from __future__ import annotations

import pickle

import pytest

from nitypes.waveform import ExtendedPropertyDictionary


@pytest.mark.parametrize(
    "pickled_value, expected",
    [
        # nitypes 1.0.0
        (
            b"\x80\x04\x95\x88\x00\x00\x00\x00\x00\x00\x00\x8c\x10nitypes.waveform\x94\x8c\x1aExtendedPropertyDictionary\x94\x93\x94)\x81\x94N}\x94\x8c\x0b_properties\x94}\x94(\x8c\x0eNI_ChannelName\x94\x8c\x08Dev1/ai0\x94\x8c\x12NI_UnitDescription\x94\x8c\x05Volts\x94us\x86\x94b.",
            ExtendedPropertyDictionary(
                {"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Volts"}
            ),
        ),
        # nitypes 1.0.1
        (
            b"\x80\x04\x95t\x00\x00\x00\x00\x00\x00\x00\x8c\x10nitypes.waveform\x94\x8c\x1aExtendedPropertyDictionary\x94\x93\x94}\x94(\x8c\x0eNI_ChannelName\x94\x8c\x08Dev1/ai0\x94\x8c\x12NI_UnitDescription\x94\x8c\x05Volts\x94u\x85\x94R\x94.",
            ExtendedPropertyDictionary(
                {"NI_ChannelName": "Dev1/ai0", "NI_UnitDescription": "Volts"}
            ),
        ),
    ],
)
def test___pickled_value___unpickle___is_compatible(
    pickled_value: bytes, expected: ExtendedPropertyDictionary
) -> None:
    new_value = pickle.loads(pickled_value)
    assert new_value == expected
