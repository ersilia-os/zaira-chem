import shutil
import os

from . import ASSAYS_FILENAME, COMPOUNDS_FILENAME, FOLDS_FILENAME, TASKS_FILENAME, VALUES_FILENAME


class SetupCleaner(object):
    def __init__(self, path):
        self.path = path

    def _individual_files(self):
        for f in [ASSAYS_FILENAME, COMPOUNDS_FILENAME, FOLDS_FILENAME, TASKS_FILENAME, VALUES_FILENAME]:
            path = os.path.join(self.path, f)
            if os.path.exists(path):
                os.remove(path)

    def _melloddy(self):
        path = os.path.join(self.path, "melloddy")
        if os.path.exists(path):
            shutil.rmtree(path)

    def _augmenter(self):
        path = os.path.join(self.path, "augmenter")
        if os.path.exists(path):
            shutil.rmtree(path)

    def run(self):
        self._melloddy()
        self._individual_files()
        self._augmenter()
