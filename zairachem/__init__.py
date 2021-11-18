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
from time import time
from .vars import BASE_DIR
from .vars import SESSION_FILE
from .vars import TRAINED_MODEL_SUBFOLDER


class ZairaBase(object):
    def __init__(self):
        self.logger = logger

    def get_output_dir(self):
        with open(os.path.join(BASE_DIR, SESSION_FILE), "r") as f:
            session = json.load(f)
        return session["output_dir"]

    def get_elapsed_time(self):
        with open(os.path.join(BASE_DIR, SESSION_FILE), "r") as f:
            session = json.load(f)
        return session["elapsed_time"]

    def reset_time(self):
        with open(os.path.join(BASE_DIR, SESSION_FILE), "r") as f:
            session = json.load(f)
        session["time_stamp"] = int(time())
        with open(os.path.join(BASE_DIR, SESSION_FILE), "w") as f:
            json.dump(session, f, indent=4)

    def update_elapsed_time(self):
        with open(os.path.join(BASE_DIR, SESSION_FILE), "r") as f:
            session = json.load(f)
        delta_time = int(time()) - session["time_stamp"]
        session["elapsed_time"] = session["elapsed_time"] + delta_time
        with open(os.path.join(BASE_DIR, SESSION_FILE), "w") as f:
            json.dump(session, f, indent=4)

    def get_trained_dir(self):
        output_dir = self.get_output_dir()
        return os.path.join(output_dir, TRAINED_MODEL_SUBFOLDER)

    def is_predict(self):
        trained_dir = self.get_trained_dir()
        if os.path.exists(trained_dir):
            return True
        else:
            return False

    def is_train(self):
        if self.is_predict():
            return False
        else:
            return True


__all__ = ["__version__"]
