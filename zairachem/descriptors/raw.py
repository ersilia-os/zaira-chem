import json
from ... import ZairaBase


class RawDescriptors(ZairaBase):

    def __init__(self):
        ZairaBase.__init__(self)
        self.path = self.get_output_dir()
        self.params = self._load_params()

    def _load_params(self):
        with open(os.path.join(self.path, PARAMETERS_FILE), "r") as f:
            params = json.load(f)
        return params

    def model_ids(self):
        pass

    def run(self):
        pass
