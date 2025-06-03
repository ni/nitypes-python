from __future__ import annotations

from typing import Any

from typing_extensions import final

from nitypes._exceptions import invalid_arg_value, invalid_arg_type
from nitypes.scalar._scalar_base import ScalarBase


@final
class ScalarData(ScalarBase):
    """A scalar data class, which encapsulates scalar data and units information."""

    __slots__ = ()

    def __init__(
        self,
        data: Any = None,
        units: str = "",
    ) -> None:
        """Construct a scalar data object.

        Args:
            data: The scalar data to store in this object.
            units: The units string associated with this data.

        Returns:
            A scalar data object.
        """
        if data is None:
            raise invalid_arg_value("scalar input data", "non-None scalar value", data)

        if type(data) not in ScalarBase._get_supported_dtypes():
            raise invalid_arg_type("scalar input data", "bool, int, float, or str", data)

        return super().__init__(data, units)
