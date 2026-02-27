from importlib.metadata import version as _version, PackageNotFoundError

try:
    __version__ = _version(__name__)
except PackageNotFoundError:  # pragma: no cover
    pass

# Import unred function to make it available at package level
from .unred import unred
