"""Single source for typing backports to avoid depending on typing_extensions at run time."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if sys.version_info >= (3, 10):
    from typing import TypeAlias
elif TYPE_CHECKING:
    from typing_extensions import TypeAlias

if sys.version_info >= (3, 11):
    from typing import Self, assert_type
elif TYPE_CHECKING:
    from typing_extensions import Self, assert_type
else:

    def assert_type(val, typ, /):  # noqa: D103 - Missing docstring in public function
        pass


if sys.version_info >= (3, 12):
    from typing import override
elif TYPE_CHECKING:
    from typing_extensions import override
else:

    def override(arg, /):  # noqa: D103 - Missing docstring in public function
        return arg


__all__ = [
    "assert_type",
    "override",
    "Self",
    "TypeAlias",
]
