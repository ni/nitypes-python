"""Increment version number.

This script is a workaround for https://github.com/python-poetry/poetry/issues/8718 - "Add 'dev' as
version bump rule for developmental releases."
"""

import re
import sys


def main(args: list[str]) -> int | str | None:
    """Increment version number."""
    version = args[0]
    match = re.match(r"^(.*-dev)(\d+)$", version)
    if match:
        version = f"{match.group(1)}{int(match.group(2))+1}"
    print(version)
    return None


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
