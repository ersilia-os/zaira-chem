import os
import numpy as np
import csv

from rdkit import Chem
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

from .types.avalon import Avalon
from .types.cddd import Cddd
from .types.ecfp import Ecfp
from .types.grover import Grover
from .types.maccs import Maccs
from .types.mordred import Mordred
from .types.rdkit2d import Rdkit2d
from .types.rdkitfpbits import RdkitFpBits
from .types.signaturizer import Signaturizer
from .types.whales import Whales

from .. import logger
from ..setup.setup import DATA_SUBFOLDER, DESCRIPTORS_SUBFOLDER, _SMILES_FILENAME


_X_DESC_FILENAME = "X.npy"

DESCRIPTOR_FACTORY = [
    Avalon(),
    # Cddd(),
    Ecfp(),
    # Grover(),
    Maccs(),
<<<<<<< HEAD
    # Mordred(),
=======
<<<<<<< HEAD
    Mordred(),
=======
    # Mordred(),
>>>>>>> f7356c4... Major updates
>>>>>>> 8a05dcf7bcbfbee32aa23019194890e2b9ee485e
    Rdkit2d(),
    RdkitFpBits(),
    Signaturizer(),
    # Whales()
]


class DescriptorsCalculator(object):

    def __init__(self, data):
        self.data = data
        self._factory = dict((d.name, d) for d in DESCRIPTOR_FACTORY)
        self._funcs = dict((k, v.calc) for k, v in self._factory.items())

    @staticmethod
    def smiles_to_mols(query_smiles):
        mols = [Chem.MolFromSmiles(smi) for smi in query_smiles]
        valid = [0 if mol is None else 1 for mol in mols]
        valid_idxs = [idx for idx, boolean in enumerate(valid) if boolean == 1]
        valid_mols = [mols[idx] for idx in valid_idxs]
        return valid_mols, valid_idxs

    def _calculate(self, name):
        mols, idxs = self.smiles_to_mols(self.data)
        func = self._funcs[name]
        res = func(mols)
        X = np.zeros(
            (len(self.data), np.array(res).shape[1]), dtype=res.dtype
        )  # TODO: Do not assume 0 for invalid smiles.
        X[np.array(idxs)] = res
        return X

    def calculate_one(self, name):
        return self._calculate(name)

    def calculate(self):
        names = sorted([k for k, _ in self._funcs.items()])
        for name in names:
            X = self._calculate(name)
            yield X, name


class Descriptors(object):

    def __init__(self, dir):
        self.dir = os.path.abspath(dir)
        logger.debug("Calculating descriptors in {0}".format(self.dir))

    def _find_batches(self):
        logger.debug("Finding batches from setup output")
        data = os.path.join(self.dir, DATA_SUBFOLDER)
        for dir in os.listdir(data):
            if dir[:len(_BATCH_PREFIX)] == _BATCH_PREFIX:
                batch_dir = os.path.join(data, dir)
                yield dir, batch_dir

    def _read_batch_data(self, batch_dir):
        with open(os.path.join(batch_dir, _SMILES_FILENAME), "r") as f:
            reader = csv.reader(f)
            data = []
            for r in reader:
                data += [r[0]]
        return data

    def _scale(self, X):
        pass

    def _impute(self, X):
        imp = SimpleImputer()
        imp.fit(X)
        with open():
            joblib.dump()

    def _prune(self, X):
        pass

    def _drop_nan_columns(self, X):
        return X[:, ~np.isnan(X).any(axis=0)]

    def calculate_iter(self):
        batches = self._find_batches()
        for batch, batch_dir in batches:
            logger.debug("Calculating descriptors for batch {0}".format(batch))
            dir = os.path.join(self.dir, DESCRIPTORS_SUBFOLDER, batch)
            if not os.path.exists(dir):
                os.mkdir(dir)
            data = self._read_batch_data(batch_dir)
            desc = DescriptorsCalculator(data)
            for X, name in desc.calculate():
                X = self._drop_nan_columns(X)
                dir_ = os.path.join(dir, name)
                if not os.path.exists(dir_):
                    os.mkdir(dir_)
                with open(os.path.join(dir_, _X_DESC_FILENAME), "wb") as f:
                    np.save(f, X, allow_pickle=False)
                yield batch, name

    def calculate(self):
        for _ in self.calculate_iter():
            pass
