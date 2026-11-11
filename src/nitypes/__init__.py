"""Data types for NI Python APIs."""

import importlib.metadata

try:
    # Use a literal package name so PyInstaller automatically copies package metadata.
    __version__ = importlib.metadata.version("nitypes")
    """nitypes version string."""
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"
