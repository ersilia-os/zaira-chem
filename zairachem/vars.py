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
ESTIMATORS_SUBFOLDER = "estimators"
POOL_SUBFOLDER = "pool"
RESULTS_FILENAME = "results_unmapped.csv"
LITE_SUBFOLDER = "lite"
REPORT_SUBFOLDER = "report"

TRAINED_MODEL_SUBFOLDER = "trained"

CLF_PERCENTILES = [1, 10, 25]

MIN_CLASS = 30
N_FOLDS = 5

_CONFIG_FILENAME = "config.json"

# Ersilia Model Hub

ERSILIA_HUB_DEFAULT_MODELS = [
    "morgan-counts",
    "cc-signaturizer",
    "grover-embedding",
    "molbert",
    "mordred",
]
