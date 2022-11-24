import os
import pandas as pd

from .baseline import Embedder, Fingerprinter
from ..utils.matrices import Hdf5
from .. import ZairaBase

from ..setup import SMILES_COLUMN
from ..vars import DATA_SUBFOLDER, DATA_FILENAME, DESCRIPTORS_SUBFOLDER

REFERENCE_FILE_NAME = "reference.h5"
SIMPLE_FILE_NAME = "simple.h5"


class ReferenceLoader(ZairaBase):
    def __init__(self):
        ZairaBase.__init__(self)
        self.path = self.get_output_dir()

    def open(self, eos_id):
        path = os.path.join(
            self.path, DESCRIPTORS_SUBFOLDER, eos_id, REFERENCE_FILE_NAME
        )
        return Hdf5(path)


class ReferenceDescriptors(ZairaBase):
    def __init__(self):
        ZairaBase.__init__(self)
        self.path = self.get_output_dir()
        self.input_csv = os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME)
        self.smiles_list = self._get_smiles_list()

    def _get_smiles_list(self):
        df = pd.read_csv(self.input_csv)
        return list(df[SMILES_COLUMN])

    def output_h5_filename(self):
        path = os.path.join(self.path, DESCRIPTORS_SUBFOLDER)
        os.makedirs(path, exist_ok=True)
        return os.path.join(path, REFERENCE_FILE_NAME)

    def run(self):
        output_h5 = self.output_h5_filename()
        ref = Embedder()
        ref.calculate(self.smiles_list, output_h5)


class SimpleDescriptors(ZairaBase):
    def __init__(self):
        ZairaBase.__init__(self)
        self.path = self.get_output_dir()
        self.input_csv = os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME)
        self.smiles_list = self._get_smiles_list()

    def _get_smiles_list(self):
        df = pd.read_csv(self.input_csv)
        return list(df[SMILES_COLUMN])

    def output_h5_filename(self):
        path = os.path.join(self.path, DESCRIPTORS_SUBFOLDER)
        os.makedirs(path, exist_ok=True)
        return os.path.join(path, SIMPLE_FILE_NAME)

    def run(self):
        output_h5 = self.output_h5_filename()
        smp = Fingerprinter()
        smp.calculate(self.smiles_list, output_h5)
