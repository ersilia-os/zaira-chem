import os

ROOT = os.path.dirname(os.path.abspath(__file__))

MELLODDY_SUBFOLDER = "melloddy"

_SCRIPT_FILENAME = "run.sh"

T0_FILE = "T0.csv"
T1_FILE = "T1.csv"
T2_FILE = "T2.csv"

NUM_CPU = 1
KEY_FILE = os.path.join(ROOT, "config", "example_key.json")
DEFAULT_PARAMS_FILE = os.path.join(ROOT, "config", "example_parameters.json")
REF_HASH_PUBLIC = os.path.join(ROOT, "config", "ref_hash_public.json")

# Todo
PARAMS_FILE = DEFAULT_PARAMS_FILE

TAG = "results"
