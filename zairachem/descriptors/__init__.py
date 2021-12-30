import os
import json
from .. import ZairaBase
from .raw import DESCRIPTORS_SUBFOLDER


class DescriptorBase(ZairaBase):
    def __init__(self):
        ZairaBase.__init__(self)
        self.path = self.get_output_dir()
        self.trained_path = self.get_trained_dir()
        self._is_predict = self.is_predict()
        self._is_train = self.is_train()

    def done_eos_iter(self):
        with open(
            os.path.join(self.path, DESCRIPTORS_SUBFOLDER, "done_eos.json"), "r"
        ) as f:
            data = json.load(f)
        for eos_id in data:
            yield eos_id
