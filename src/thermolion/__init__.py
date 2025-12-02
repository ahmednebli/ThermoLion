"""Top-level package for ThermoLion.

This package exposes the :class:`ThermoLion` optimizer.
"""

from .version import __version__
from .optimizers import ThermoLion

__all__ = ["ThermoLion", "__version__"]
