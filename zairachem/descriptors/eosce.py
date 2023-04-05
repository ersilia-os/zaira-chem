import os
import pandas as pd
import h5py

from eosce.models import ErsiliaCompoundEmbedding
from ..utils.matrices import Hdf5
from .. import ZairaBase

from ..setup import SMILES_COLUMN
from ..vars import DATA_SUBFOLDER, DATA_FILENAME, DESCRIPTORS_SUBFOLDER

EOSCE_FILE_NAME = "eosce.h5"


class EosceEmbedder(ZairaBase):
    def __init__(self):
        ZairaBase.__init__(self)
        self.model = ErsiliaCompoundEmbedding()

    def calculate(self, smiles_list, output_h5):
        X = self.model.transform(smiles_list)
        if output_h5 is None:
            return X
        keys = ["key-{0}".format(i) for i in range(len(smiles_list))]
        features = ["feat-{0}".format(i) for i in range(X.shape[1])]
        inputs = smiles_list
        with h5py.File(output_h5, "w") as f:
            f.create_dataset("Keys", data=keys)
            f.create_dataset("Features", data=features)
            f.create_dataset("Inputs", data=inputs)
            f.create_dataset("Values", data=X)


class EosceLoader(ZairaBase):
    def __init__(self):
        ZairaBase.__init__(self)
        self.path = self.get_output_dir()

    def open(self, eos_id):
        path = os.path.join(self.path, DESCRIPTORS_SUBFOLDER, eos_id, EOSCE_FILE_NAME)
        return Hdf5(path)


class EosceDescriptors(ZairaBase):
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
        return os.path.join(path, EOSCE_FILE_NAME)

    def run(self):
        output_h5 = self.output_h5_filename()
        ref = EosceEmbedder()
        ref.calculate(self.smiles_list, output_h5)
