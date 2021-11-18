import shutil
import os


class SetupCleaner(object):
    def __init__(self, path):
        self.path = path

    def _melloddy(self):
        path = os.path.join(self.path, "melloddy")
        if os.path.exists(path):
            shutil.rmtree(path)

    def run(self):
        self._melloddy()
