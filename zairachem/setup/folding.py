import os
import numpy as np
import pandas as pd
from ..tools.melloddy import MELLODDY_SUBFOLDER
from . import COMPOUNDS_FILENAME, COMPOUND_IDENTIFIER_COLUMN, FOLDS_FILENAME

from sklearn.model_selection import KFold
from ..vars import DATA_FILENAME, N_FOLDS


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
        dfm = pd.read_csv(
            os.path.join(
                self.path,
                MELLODDY_SUBFOLDER,
                "results",
                "results_tmp",
                "folding",
                "T2_folds.csv",
            )
        )[["input_compound_id", "fold_id"]]
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
        dfm = pd.read_csv(
            os.path.join(
                self.path,
                MELLODDY_SUBFOLDER,
                "results",
                "results_tmp",
                "lsh_folding",
                "T2_descriptors_lsh.csv",
            )
        )[["input_compound_id", "fold_id"]]
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
        self.path = path
        self.df = pd.read_csv(os.path.join(self.path, DATA_FILENAME))
        pass

    def get_folds(self):
        pass


class AuxiliaryFolding(object):
    def __init__(self, df):
        self.df = df.copy()

    @staticmethod
    def split(a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))

    def get_folds(self):
        # TODO: Use time or pseudo-time (e.g. based on PCA)
        N = self.df.shape[0]
        self.df["index"] = [i for i in range(self.df.shape[0])]
        self.df = self.df.sort_values(["fld_scf", "fld_lsh", "fld_rnd"]).reset_index(
            drop=True
        )
        chunks = list(self.split(range(N), N_FOLDS))
        indices = list(self.df["index"])
        folds = [0] * N
        for j, c in enumerate(chunks):
            for i in c:
                folds[indices[i]] = j
        return folds


class ValidationFolding(object):
    def __init__(self, df):
        self.df = df.copy()

    def _has_date(self):
        if "fld_dat" in list(self.df.columns):
            return True
        else:
            return False

    def _reference_column(self):
        return "fld_rnd"
        if self._has_date():
            return "fld_dat"
        else:
            return "fld_aux"

    def get_folds(self):
        ref = self._reference_column()
        n = np.max(self.df[ref])
        folds = []
        for f in list(self.df[ref]):
            if f == n:
                folds += [1]
            else:
                folds += [0]
        return folds


# TODO: check minimum number of folds
class Folds(object):
    def __init__(self, path):
        self.path = path
        self.file_name = os.path.join(self.path, FOLDS_FILENAME)

    def run(self):
        data = {
            "fld_rnd": RandomFolding(self.path).get_folds(),
            "fld_scf": ScaffoldFolding(self.path).get_folds(),
            "fld_lsh": LshFolding(self.path).get_folds(),
        }
        df = pd.DataFrame(data)
        df["fld_aux"] = AuxiliaryFolding(df).get_folds()
        df["fld_val"] = ValidationFolding(df).get_folds()
        df.to_csv(self.file_name, index=False)
