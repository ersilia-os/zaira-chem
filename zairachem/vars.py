import os
from pathlib import Path

BASE_DIR = os.path.join(str(Path.home()), "zairachem")
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

LOGGING_FILE = "console.log"
SESSION_FILE = "session.json"

# Environmental variables

DATA_SUBFOLDER = "data"
DATA_FILENAME = "data.csv"
DESCRIPTORS_SUBFOLDER = "descriptors"
MODELS_SUBFOLDER = "models"
POOL_SUBFOLDER = "pool"
LITE_SUBFOLDER = "lite"

TRAINED_MODEL_SUBFOLDER = "trained"

CLF_SUBFOLDER = "clf"
REG_SUBFOLDER = "reg"

CLF_PERCENTILES = [1, 10, 25]

MIN_CLASS = 30
N_FOLDS = 5

_CONFIG_FILENAME = "config.json"

AUTOML_DEFAULT_TIME_BUDGET_MIN = 10
