import os
import tempfile
from ...utils.terminal import run_command

from . import T0_FILE
from . import TAG
from . import MELLODDY_SUBFOLDER
from . import PARAMS_FILE, KEY_FILE
from . import NUM_CPU
from . import _SCRIPT_FILENAME


class Thresholding(object):
    def __init__(self, outdir):
        self.outdir = self.get_output_path(outdir)
        self.input_file_0 = self.get_input_file_0()
        self.input_file_4r = self.get_input_file_4r()

    def get_output_path(self, outdir):
        return os.path.join(outdir, MELLODDY_SUBFOLDER)

    def get_input_file_0(self):
        return os.path.join(self.outdir, T0_FILE)

    def get_input_file_4r(self):
        return os.path.join(self.outdir, TAG, "results_tmp", "aggregation", "T4r.csv")

    def script_file(self):
        text = "tunercli apply_thresholding --activity_file {0} --assay_file {1} --config_file {2} --key_file {3} --output_dir {4} --run_name {5} --number_cpu {6} --non_interactive".format(
            self.input_file_4r,
            self.input_file_0,
            PARAMS_FILE,
            KEY_FILE,
            os.path.join(self.outdir),
            TAG,
            NUM_CPU,
        )
        tmp_dir = tempfile.mkdtemp()
        self.script_path = os.path.join(tmp_dir, _SCRIPT_FILENAME)
        with open(self.script_path, "w") as f:
            f.write(text)

    def run(self):
        self.script_file()
        cmd = "bash {0}".format(self.script_path)
        run_command(cmd)
