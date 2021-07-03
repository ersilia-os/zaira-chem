# Version
from ._version import __version__

del _version

# Disable third-party warnings
import warnings
warnings.filterwarnings("ignore")

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Internal imports
from .utils.logging import logger

__all__ = ["__version__"]
