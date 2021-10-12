import os
import numpy as np
import pandas as pd
from ..tools.melloddy import MELLODDY_SUBFOLDER, TAG
from . import COMPOUNDS_FILENAME, VALUES_FILENAME, COMPOUND_IDENTIFIER_COLUMN, SMILES_COLUMN, FOLDS_FILENAME

from sklearn.model_selection import KFold


N_FOLDS = 5


class RandomFolding(object):
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(os.path.join(self.path, COMPOUNDS_FILENAME))

    def get_folds(self):
        splitter = KFold(n_splits=N_FOLDS, shuffle=True)
        folds = np.zeros(self.df.shape[0], dtype=int)
        i = 0
        for _, test_idx in splitter.split(folds):
            folds[test_idx] = i
            i += 1
        return list(folds)


class ScaffoldFolding(object):
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(os.path.join(self.path, COMPOUNDS_FILENAME))

    def get_folds(self):
        dfm = pd.read_csv(os.path.join(self.path, MELLODDY_SUBFOLDER, "results", "results_tmp", "folding", "T2_folds.csv"))[["input_compound_id", "fold_id"]]
        folds_dict = {}
        for cid, fld in dfm.values:
            folds_dict[cid] = fld
        folds = []
        for cid in list(self.df[COMPOUND_IDENTIFIER_COLUMN]):
            folds += [folds_dict[cid]]
        return folds


class LshFolding(object):
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(os.path.join(self.path, COMPOUNDS_FILENAME))

    def get_folds(self):
        dfm = pd.read_csv(os.path.join(self.path, MELLODDY_SUBFOLDER, "results", "results_tmp", "lsh_folding", "T2_descriptors_lsh.csv"))[["input_compound_id", "fold_id"]]
        folds_dict = {}
        for cid, fld in dfm.values:
            folds_dict[cid] = fld
        folds = []
        for cid in list(self.df[COMPOUND_IDENTIFIER_COLUMN]):
            folds += [folds_dict[cid]]
        return folds


class GroupFolding(object):
    def __init__(self, path):
        pass

    def get_folds(self):
        pass


class DateFolding(object):
    def __init__(self, path):
        pass

    def get_folds(self):
        pass


# TODO: check minimum number of folds
class Folds(object):

    def __init__(self, path):
        self.path = path
        self.file_name = os.path.join(self.path, FOLDS_FILENAME)

    def run(self):
        data = {
            "random": RandomFolding(self.path).get_folds(),
            "scaffold": ScaffoldFolding(self.path).get_folds(),
            "lsh": LshFolding(self.path).get_folds()
        }
        df = pd.DataFrame(data)
        df.to_csv(self.file_name, index=False)
