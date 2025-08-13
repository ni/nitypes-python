"""Pin dependencies to the oldest compatible version for testing."""

from __future__ import annotations

import sys
from pathlib import Path

import tomlkit
from tomlkit.items import AbstractTable, Array


def main(args: list[str]) -> int | str | None:
    """Pin dependencies to the oldest compatible version for testing."""
    pyproject_path = Path(args.pop())
    if args:
        return f"Unsupported arguments: {args!r}"
    pyproject = tomlkit.loads(pyproject_path.read_text())

    poetry_deps = pyproject["tool"]["poetry"]["dependencies"]  # type: ignore[index]
    assert isinstance(poetry_deps, AbstractTable)
    _pin_oldest_for_deps_list(poetry_deps)

    dev_deps = pyproject["tool"]["poetry"]["group"]["dev"]["dependencies"]  # type: ignore[index]
    assert isinstance(dev_deps, AbstractTable)
    _pin_oldest_for_deps_list(dev_deps)

    pyproject_path.write_text(tomlkit.dumps(pyproject))
    print("Updated pyproject.toml with pinned dependencies.")
    return None


def _pin_oldest_for_deps_list(deps_list: AbstractTable) -> None:
    assert isinstance(deps_list, AbstractTable)

    for dep, value in deps_list.items():
        if dep == "python":
            continue
        if isinstance(value, str) and (
            value.startswith("^") or value.startswith("~") or value.startswith(">=")
        ):
            deps_list[dep] = "==" + value.lstrip("^~>=")
        elif isinstance(value, Array):
            for constraint in value:
                if "version" in constraint and (
                    constraint["version"].startswith("^")
                    or constraint["version"].startswith("~")
                    or constraint["version"].startswith(">=")
                ):
                    constraint["version"] = "==" + constraint["version"].lstrip("^~>=")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
