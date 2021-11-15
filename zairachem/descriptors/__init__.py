import os
import json
from .. import ZairaBase
from .raw import DESCRIPTORS_SUBFOLDER

GLOBAL_SUPERVISED_FILE_NAME = "global_supervised.h5"


class DescriptorBase(ZairaBase):
    def __init__(self):
        ZairaBase.__init__(self)
        self.path = self.get_output_dir()

    def done_eos_iter(self):
        with open(
            os.path.join(self.path, DESCRIPTORS_SUBFOLDER, "done_eos.json"), "r"
        ) as f:
            data = json.load(f)
        for eos_id in data:
            yield eos_id
