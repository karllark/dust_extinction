from importlib.metadata import version as _version, PackageNotFoundError

try:
    __version__ = _version(__name__)
except PackageNotFoundError:  # pragma: no cover
    pass

from .deredden import deredden_flux

__all__ = ["deredden_flux"]
