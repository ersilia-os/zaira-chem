import shutil

import os
import shutil

from ..vars import POOL_SUBFOLDER
from .. import ZairaBase
from ..estimators import RESULTS_MAPPED_FILENAME
from . import OUTPUT_FILENAME


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
        self._predictions_file()
