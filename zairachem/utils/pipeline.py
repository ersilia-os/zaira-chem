import os
import json
from time import time

from .. import ZairaBase
from ..vars import SESSION_FILE


class SessionFile(ZairaBase):
    def __init__(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        self.session_file = os.path.join(os.path.abspath(output_dir), SESSION_FILE)
        print("Session file", self.session_file)

    def open_session(self, mode, output_dir, model_dir=None):
        self.mode = mode
        self.output_dir = os.path.abspath(output_dir)
        if model_dir is None:
            self.model_dir = self.output_dir
        else:
            self.model_dir = os.path.abspath(model_dir)
        data = {
            "output_dir": self.output_dir,
            "model_dir": self.model_dir,
            "time_stamp": int(time()),
            "elapsed_time": 0,
            "mode": self.mode,
        }
        with open(self.session_file, "w") as f:
            json.dump(data, f, indent=4)

    def delete_session_file(self):
        if os.path.exists(self.session_file):
            os.remove(self.session_file)


class PipelineStep(ZairaBase):
    def __init__(self, name, output_dir):
        ZairaBase.__init__(self)
        self.name = name
        sf = SessionFile(output_dir)
        self.session_file = sf.session_file

    def _read_session(self):
        if not os.path.exists(self.session_file):
            return None
        with open(self.session_file, "r") as f:
            data = json.load(f)
        if "steps" not in data:
            data["steps"] = []
        return data

    def _write_session(self, data):
        with open(self.session_file, "w") as f:
            json.dump(data, f, indent=4)

    def update(self):
        data = self._read_session()
        data["steps"] += [self.name]
        self._write_session(data)

    def is_done(self):
        data = self._read_session()
        if data is None:
            return False
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
