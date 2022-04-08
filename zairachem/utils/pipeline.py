import os
import json

from .. import ZairaBase
from ..vars import BASE_DIR, SESSION_FILE


class PipelineStep(ZairaBase):
    def __init__(self, name):
        ZairaBase.__init__(self)
        self.name = name

    def _read_session(self):
        with open(os.path.join(BASE_DIR, SESSION_FILE), "r") as f:
            data = json.load(f)
        if "steps" not in data:
            data["steps"] = []
        return data

    def _write_session(self, data):
        with open(os.path.join(BASE_DIR, SESSION_FILE), "w") as f:
            json.dump(data, f, indent=4)

    def update(self):
        data = self._read_session()
        data["steps"] += [self.name]
        self._write_session(data)

    def is_done(self):
        data = self._read_session()
        if self.name in data["steps"]:
            return True
        else:
            return False

    def reset(self):
        data = self._read_session()
        steps = []
        for s in data["steps"]:
            if s == self.name:
                break
            else:
                steps += [s]
        data["steps"] = steps
        self._write_session(data)
