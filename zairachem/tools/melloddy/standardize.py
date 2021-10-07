import os, sys
import tempfile
from ...utils.terminal import run_command

ROOT = os.path.dirname(os.path.abspath(__file__))

NUM_CPU=1
KEY_FILE="example_key.json"
PARAMS_FILE="example_parameters.json"
REF_HASH_PUBLIC="ref_hash_public.json"

_SCRIPT_FILENAME = "run.sh"


class Standardize(object):

    def __init__(self, tag):
        self.tag = tag
        self.outdir = outdir

    def script_file(self):
        text = """
        t2 = {0}
        param = {1}
        key = {2}
        outdir = {3}
        number_cpu = {4}
        run_name = {5}

        tunercli standardize_smiles --structure_file $t2 \
                                    --config_file $param \
                                    --key_file $key \
                                    --output_dir $outdir \
                                    --run_name $run_name \
                                    --number_cpu $num_cpu \
                                    --non_interactive

        """.format(
            infile,
            os.path.join(ROOT, PARAMS_FILE),
            os.path.join(ROOT, KEY_FILE),
            os.path.join(self.outdir)
            NUM_CPU,
            self.tag
        )
        tmp_dir = tempfile.mkdtemp()
        self.script_path = os.path.join(tmp_dir, _SCRIPT_FILENAME)
        with open(self.script_path, "w") as f:
            f.write(text)

    def run(self):
        self.script_file()
        cmd = "bash {0}".format(self.script_path)
        run_command(cmd)
