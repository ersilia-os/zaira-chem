# Version
__version__ = "0.0.1"

# Disable third-party warnings
import warnings

warnings.filterwarnings("ignore")

try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except:
    pass

# Internal imports
from .utils.logging import logger

# Base Zaira class
import json
import os
from .vars import BASE_DIR
from .vars import SESSION_FILE


class ZairaBase(object):
    def __init__(self):
        self.logger = logger

    def get_output_dir(self):
        with open(os.path.join(BASE_DIR, SESSION_FILE), "r") as f:
            session = json.load(f)
        return session["output_dir"]


__all__ = ["__version__"]
