import importlib.metadata

import nitypes


def test___version_attribute___matches_importlib_metadata() -> None:
    assert nitypes.__version__ == importlib.metadata.version("nitypes")
