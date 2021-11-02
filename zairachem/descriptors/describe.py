import os
import pandas as pd

from .raw import RawDescriptors

from ..setup import COMPOUNDS_FILENAME
from ..vars import DATA_SUBFOLDER
from .. import ZairaBase


class Describer(ZairaBase):

    def __init__(self, path):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        self.logger.debug(self.path)

    def load_inputs(self):
        self.logger.debug("Reading inputs")
        df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, COMPOUNDS_FILENAME))

    def _raw_descriptions(self):
        RawDescriptors().run()

    def run(self):
        self._raw_descriptions()
        df = self.load_inputs()
        pass
