import joblib
import os
import json
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from umap import UMAP

from .raw import DESCRIPTORS_SUBFOLDER, RawLoader
from .. import ZairaBase
from ..utils.matrices import Hdf5

MAX_NA = 0.2


class NanFilter(object):
    def __init__(self):
        self._name = "nan_filter"

    def fit(self, X):
        max_na = int((1-MAX_NA)*X.shape[0])
        idxs = []
        for j in range(X.shape[1]):
            c = np.sum(np.isnan(X[:, j]))
            if c > max_na:
                continue
            else:
                idxs += [j]
        self.col_idxs = idxs

    def transform(self, X):
        return X[:, self.col_idxs]

    def save(self, file_name):
        joblib.dump(self, file_name)

    def load(self, file_name):
        return joblib.load(file_name)


class Scaler(object):
    def __init__(self):
        self._name = "scaler"
        self.abs_limit = 10

    def fit(self, X):
        self.scaler = RobustScaler()
        self.scaler.fit(X)

    def transform(self, X):
        X = self.scaler.transform(X)
        X = np.clip(X, -self.abs_limit, self.abs_limit)
        return X

    def save(self, file_name):
        joblib.dump(self, file_name)

    def load(self, file_name):
        return joblib.load(file_name)


class Imputer(object):

    def __init__(self):
        self._name = "imputer"
        self._fallback = 0

    def fit(self, X):
        ms = []
        for j in range(X.shape[1]):
            vals = X[:, j]
            mask = ~np.isnan(vals)
            vals = vals[mask]
            if len(vals) == 0:
                m = self._fallback
            else:
                m = np.median(vals)
            ms += [m]
        self.impute_values = np.array(ms)

    def transform(self, X):
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = self.impute_values[j]
        return X

    def save(self, file_name):
        joblib.dump(self, file_name)

    def load(self, file_name):
        return joblib.load(file_name)


class VarianceFilter(object):
    def __init__(self):
        self._name = "variance_filter"

    def fit(self, X):
        self.sel = VarianceThreshold()
        self.sel.fit(X)

    def transform(self, X):
        return self.sel.transform(X)

    def save(self, file_name):
        joblib.dump(self, file_name)

    def load(self, file_name):
        return joblib.load(file_name)


class Pca(object):
    def __init__(self):
        self._name = "pca"

    def fit(self, X):
        self.pca = PCA(n_components=0.9)
        self.pca.fit(X)

    def transform(self, X):
        return self.pca.transform(X)

    def save(self, file_name):
        joblib.dump(self, file_name)

    def load(self, file_name):
        return joblib.load(file_name)


class OptSne(object):
    def __init__(self):
        self._name = "opt_tsne"

    def fit(self):
        pass

    def transform(self):
        pass


class UnsupervisedUmap(object):
    def __init__(self):
        self._name = "unsupervised_umap"

    def fit(self, X):
        self.reducer = UMAP(densmap=False)
        self.reducer.fit(X)

    def transform(self, X):
        return self.reducer.transform(X)

    def save(self, file_name):
        joblib.dump(self, file_name)

    def load(self, file_name):
        return joblib.load(file_name)



class IndividualUnsupervisedTransformations(ZairaBase):

    def __init__(self):
        ZairaBase.__init__(self)
        self.path = self.get_output_dir()
        self.pipeline = [
            NanFilter(),
            Imputer(),
            VarianceFilter(),
            Scaler(),
        ]
        self._name = "individual_unsupervised.h5"

    def done_eos_iter(self):
        with open(os.path.join(self.path, DESCRIPTORS_SUBFOLDER, "done_eos.json"), "r") as f:
            data = json.load(f)
        for eos_id in data:
            yield eos_id

    def run(self):
        rl = RawLoader()
        for eos_id in self.done_eos_iter():
            path = os.path.join(self.path, DESCRIPTORS_SUBFOLDER, eos_id)
            data = rl.open(eos_id)
            data = data.load()
            X = data.values()
            for algo in self.pipeline:
                algo.fit(X)
                X = algo.transform(X)
                algo.save(os.path.join(path, algo._name+".joblib"))
            file_name = os.path.join(self.path, DESCRIPTORS_SUBFOLDER, eos_id, self._name)
            data._values = X
            Hdf5(file_name).save(data)


class StackedUnsupervisedTransformations(ZairaBase):

    def __init__(self):
        ZairaBase.__init__(self)
        self.path = self.get_output_dir()

    def run(self):
        pass
