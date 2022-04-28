import os
import shutil

from ..vars import DESCRIPTORS_SUBFOLDER, POOL_SUBFOLDER
from .. import ZairaBase
from ..estimators import RESULTS_MAPPED_FILENAME
from . import OUTPUT_FILENAME


class Cleaner(ZairaBase):
    def __init__(self, path):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        self.output_dir = os.path.abspath(self.path)
        assert os.path.exists(self.output_dir)

    def _clean_descriptors_by_subfolder(self, path, subfolder):
        path = os.path.join(path, subfolder)
        for d in os.listdir(path):
            if d.startswith("fp2sim"):
                continue
            if os.path.isdir(os.path.join(path, d)):
                self._clean_descriptors_by_subfolder(path, d)
            else:
                if d.endswith(".h5"):
                    os.remove(os.path.join(path, d))

    def _clean_descriptors(self):
        self._clean_descriptors_by_subfolder(self.path, DESCRIPTORS_SUBFOLDER)

    def run(self):
        self._clean_descriptors()


class Finisher(ZairaBase):
    def __init__(self, path):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path

    def _predictions_file(self):
        shutil.copy(
            os.path.join(self.path, POOL_SUBFOLDER, RESULTS_MAPPED_FILENAME),
            os.path.join(self.path, OUTPUT_FILENAME),
        )

    def run(self):
        self.logger.debug("Finishing")
        Cleaner(path=self.path).run()
        self._predictions_file()
