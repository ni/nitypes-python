from __future__ import annotations

import numpy as np
from typing_extensions import TypeAlias

from nitypes.waveform._analog_waveform import AnalogWaveform, _TComplexRaw

ComplexWaveform: TypeAlias = AnalogWaveform[_TComplexRaw, np.complex128]
"""An analog waveform containing complex-number data.

.. note::
    This is a type alias, so it can only be used in type hints, not at run time.
"""

# Also see https://github.com/python/mypy/issues/14315 - Generic TypeAlias does not infer types
