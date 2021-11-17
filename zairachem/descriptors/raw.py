import json
import os
import csv

from .. import ZairaBase
from ..utils.terminal import run_command
from ..utils.matrices import Hdf5

from ..setup import PARAMETERS_FILE
from ..vars import DATA_SUBFOLDER, DATA_FILENAME, DESCRIPTORS_SUBFOLDER


class RawLoader(ZairaBase):
    def __init__(self):
        ZairaBase.__init__(self)
        self.path = self.get_output_dir()

    def open(self, eos_id):
        path = os.path.join(self.path, DESCRIPTORS_SUBFOLDER, eos_id, "raw.h5")
        return Hdf5(path)


class RawDescriptors(ZairaBase):
    def __init__(self):
        ZairaBase.__init__(self)
        self.path = self.get_output_dir()
        self.params = self._load_params()
        self.input_csv = os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME)

    def _load_params(self):
        with open(os.path.join(self.path, DATA_SUBFOLDER, PARAMETERS_FILE), "r") as f:
            params = json.load(f)
        return params

    def eos_ids(self):
        for x in self.params["ersilia_hub"]:
            yield x

    def output_h5_filename(self, eos_id):
        path = os.path.join(self.path, DESCRIPTORS_SUBFOLDER, eos_id)
        os.makedirs(path, exist_ok=True)
        return os.path.join(path, "raw.h5")

    def run(self):
        done_eos = []
        for eos_id in self.eos_ids():
            output_h5 = self.output_h5_filename(eos_id)
            run_command(
                "ersilia -v serve {0}; ersilia -v api {0} -i {1} -o {2}".format(
                    eos_id, self.input_csv, output_h5
                )
            )
            done_eos += [eos_id]
        with open(
            os.path.join(self.path, DESCRIPTORS_SUBFOLDER, "done_eos.json"), "w"
        ) as f:
            json.dump(done_eos, f, indent=4)
