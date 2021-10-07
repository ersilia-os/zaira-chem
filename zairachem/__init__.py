# Version
__version__ = "0.0.1"

# Disable third-party warnings
import warnings

warnings.filterwarnings("ignore")

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# Internal imports
from .utils.logging import logger

# Base Zaira class
class ZairaBase(self):
    def __init__(self):
        self.logger = logger


__all__ = ["__version__"]
