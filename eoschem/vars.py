import os
from pathlib import Path

EOSCHEM_DIR = os.path.join(str(Path.home()), "eoschem")
if not os.path.exists(EOSCHEM_DIR):
    os.makedirs(EOSCHEM_DIR)

LOGGING_FILE = "console.log"
