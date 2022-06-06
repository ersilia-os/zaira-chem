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

CLF_PERCENTILES = [1, 10, 25, 50]

MIN_CLASS = 30
N_FOLDS = 5

_CONFIG_FILENAME = "config.json"

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
ZAIRACHEM_DATA_PATH = os.path.join(PACKAGE_ROOT, "data")

# Ersilia Model Hub

ERSILIA_HUB_DEFAULT_MODELS = [
    "morgan-counts",
    "cc-signaturizer",
    "grover-embedding",
    "molbert",
    "mordred",
]

ENSEMBLE_MODE = (
    "bagging"  # bagging, blending, stacking / at the moment stacking is not available
)
