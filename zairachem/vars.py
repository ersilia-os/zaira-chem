import os
from pathlib import Path

BASE_DIR = os.path.join(str(Path.home()), "zairachem")
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

LOGGING_FILE = "console.log"
SESSION_FILE = "session.json"

# Environmental variables

DATA_SUBFOLDER = "data"
DESCRIPTORS_SUBFOLDER = "descriptors"
MODELS_SUBFOLDER = "models"
POOL_SUBFOLDER = "pool"
LITE_SUBFOLDER = "lite"

_CONFIG_FILENAME = "config.json"
