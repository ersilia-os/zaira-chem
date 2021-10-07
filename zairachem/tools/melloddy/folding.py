import os
import tempfile
from ...utils.terminal import run_command

from . import TAG
from . import MELLODDY_SUBFOLDER
from . import PARAMS_FILE, KEY_FILE
from . import NUM_CPU

_SCRIPT_FILENAME = "run.sh"


class Folding(object):
    def __init__(self, outdir):
        self.outdir = self.get_output_path(outdir)
        self.input_file = self.get_input_file()

    def get_output_path(self, outdir):
        return os.path.join(outdir, MELLODDY_SUBFOLDER)

    def get_input_file(self):
        return os.path.join(
            self.outdir, TAG, "results_tmp", "descriptors", "T2_descriptors.csv"
        )

    def script_file(self, is_lsh):
        if is_lsh:
            s = "assign_lsh_fold"
        else:
            s = "assign_fold"
        text = "tunercli {6} --structure_file {0} --config_file {1} --key_file {2} --output_dir {3} --run_name {4} --number_cpu {5} --non_interactive".format(
            self.input_file,
            PARAMS_FILE,
            KEY_FILE,
            os.path.join(self.outdir),
            TAG,
            NUM_CPU,
            s,
        )
        tmp_dir = tempfile.mkdtemp()
        self.script_path = os.path.join(tmp_dir, _SCRIPT_FILENAME)
        with open(self.script_path, "w") as f:
            f.write(text)

    def _run(self, is_lsh):
        self.script_file(is_lsh=is_lsh)
        cmd = "bash {0}".format(self.script_path)
        run_command(cmd)

    def run(self):
        self._run(is_lsh=False)
        self._run(is_lsh=True)
