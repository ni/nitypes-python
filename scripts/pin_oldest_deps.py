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

    for dep, value in poetry_deps.items():
        if dep == "python":
            continue
        if isinstance(value, str) and (
            value.startswith("^") or value.startswith("~") or value.startswith(">=")
        ):
            poetry_deps[dep] = "==" + value.lstrip("^~>=")
        elif isinstance(value, Array):
            for constraint in value:
                if "version" in constraint and (
                    constraint["version"].startswith("^")
                    or constraint["version"].startswith("~")
                    or constraint["version"].startswith(">=")
                ):
                    constraint["version"] = "==" + constraint["version"].lstrip("^~>=")

    pyproject_path.write_text(tomlkit.dumps(pyproject))
    print("Updated pyproject.toml with pinned dependencies.")
    return None


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
