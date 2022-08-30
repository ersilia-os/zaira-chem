import os
import random
import numpy as np
import pandas as pd
from lol import LOL
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from .sampler import Sampler

from .. import ZairaBase
from ..descriptors.baseline import Fingerprinter


from ..setup import COMPOUND_IDENTIFIER_COLUMN
from ..tools.molmap.molmap import SMILES_COLUMN
from ..vars import DATA_FILENAME, DATA_SUBFOLDER

MAX_N = 3000
MIN_N = 1000
AUGMENTATION_FACTOR = 100

_MAX_DIM_REDUCTION_COMPONENTS = 50
_KNEIGHBORS = 1

_TIME_BUDGET_SEC = 100  # TODO


class Augmenter(ZairaBase):
    def __init__(self, path):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        self.df_exp = pd.read_csv(
            os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME)
        )
        self.n_orig = self.df_exp.shape[0]
        self.sampler = Sampler()
        self.smiles_exp = list(self.df_exp[SMILES_COLUMN])
        self.save_path = os.path.join(self.path, DATA_SUBFOLDER, "augmenter")
        self.time_limit = _TIME_BUDGET_SEC

    def _heuristic_n(self):
        ne = self.df_exp.shape[0]
        n = int(np.clip(ne * AUGMENTATION_FACTOR, MIN_N, MAX_N))
        self.logger.debug("Augmented samples: {0}".format(n))
        return n

    def _sample_smiles(self):
        n = self._heuristic_n()
        self.smiles_sampled = self.sampler.sample(smiles_list=self.smiles_exp, n=n)
        self.cids_sampled = [
            "VID{0}".format(i) for i in range(len(self.smiles_sampled))
        ]
        self.all_smiles = self.smiles_exp + self.smiles_sampled

    def _get_xy(self):
        X = Fingerprinter().calculate(self.all_smiles)
        n_exp = len(self.smiles_exp)
        n_components = int(np.min([_MAX_DIM_REDUCTION_COMPONENTS, n_exp, X.shape[1]]))
        lmao = LOL(n_components=n_components)
        X_ = X[:n_exp]
        y_ = np.array(self.df_exp["clf_aux"])
        lmao.fit(X_, y_)
        X_exp = lmao.transform(X_)
        X_aug = lmao.transform(X[n_exp:])
        self.X_exp = X_exp
        self.X_aug = X_aug

    def _guilt_by_association(self):
        data = {}
        for col in list(self.df_exp.columns):
            if "fld_" in col:
                if "fld_val" in col:
                    continue
                if "fld_" in col:
                    mdl = KNeighborsClassifier(n_neighbors=1)
                    y = np.array(self.df_exp[col])
                    mdl.fit(self.X_exp, y)
                    y_hat = mdl.predict(self.X_aug)
                    data[col] = y_hat
            else:
                if "_aux" in col:
                    continue
                if "reg_" in col:
                    mdl = KNeighborsRegressor(n_neighbors=_KNEIGHBORS)
                    y = np.array(self.df_exp[col])
                    mdl.fit(self.X_exp, y)
                    y_hat = mdl.predict(self.X_aug)
                    data[col] = y_hat
                if "clf_" in col:
                    mdl = KNeighborsClassifier(n_neighbors=_KNEIGHBORS)
                    y = np.array(self.df_exp[col])
                    mdl.fit(self.X_exp, y)
                    y_hat = mdl.predict(self.X_aug)
                    data[col] = y_hat
        self.dfa = pd.DataFrame(data)
        self.dfa[SMILES_COLUMN] = self.smiles_sampled
        self.dfa[COMPOUND_IDENTIFIER_COLUMN] = self.cids_sampled

    def _assemble(self):
        dfe = self.df_exp
        dfa = self.dfa
        dfe["is_exp"] = 1
        dfa["is_exp"] = 0
        for c in list(dfe.columns):
            if "clf" in c and "_aux" not in c and "_skip" not in c:
                col = c
                if list(dfe[col]) == list(dfe["clf_aux"]):
                    break
        dfa["clf_aux"] = dfa[col]
        val_prop = np.sum(list(dfe["fld_val"])) / dfe.shape[0]
        val_a = [0] * dfa.shape[0]
        n = int(dfa.shape[0] * val_prop + 1)
        idxs = random.sample([i for i in range(dfa.shape[0])], n)
        for i in idxs:
            val_a[i] = 1
        dfa["fld_val"] = val_a
        dfe = dfe.append(dfa, ignore_index=True)
        dfe.to_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME), index=False)

    def run(self):
        self._sample_smiles()
        self._get_xy()
        self._guilt_by_association()
        self._assemble()
