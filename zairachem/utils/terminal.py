import os
import subprocess
from ..vars import BASE_DIR, LOGGING_FILE


def run_command(cmd):
    with open(os.path.join(BASE_DIR, LOGGING_FILE), "a+") as fp:
        subprocess.Popen(cmd, stdout=fp, stderr=fp, shell=True, env=os.environ).wait()
        subprocess.Popen(cmd, shell=True, env=os.environ).wait()
