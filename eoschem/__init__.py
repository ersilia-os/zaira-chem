# Version
from ._version import __version__

del _version

import warnings
warnings.filterwarnings("ignore")

# Internal imports
from .utils.logging import logger

__all__ = ["__version__"]
