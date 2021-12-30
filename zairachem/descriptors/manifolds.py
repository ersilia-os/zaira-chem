import os
import numpy as np
import joblib
from sklearn.decomposition import PCA
from umap import UMAP

from . import DescriptorBase
from .raw import DESCRIPTORS_SUBFOLDER
from .reference import REFERENCE_FILE_NAME
from ..utils.matrices import Hdf5, Data

MAX_COMPONENTS = 4


class Pca(object):
    def __init__(self):
        self._name = "pca"

    def fit(self, X):
        n_components = np.min([MAX_COMPONENTS, X.shape[0], X.shape[1]])
        self.pca = PCA(n_components=n_components, whiten=True)
        self.pca.fit(X)

    def transform(self, X):
        return self.pca.transform(X)

    def save(self, file_name):
        joblib.dump(self, file_name)

    def load(self, file_name):
        return joblib.load(file_name)


class Umap(object):
    def __init__(self):
        self._name = "umap"

    def fit(self, X):
        self.reducer = UMAP(densmap=False)
        self.reducer.fit(X)

    def transform(self, X):
        return self.reducer.transform(X)

    def save(self, file_name):
        joblib.dump(self, file_name)

    def load(self, file_name):
        return joblib.load(file_name)


class Manifolds(DescriptorBase):
    def __init__(self):
        DescriptorBase.__init__(self)

    def _algo(self, algo):
        algo_path = os.path.join(self.trained_path, algo._name + ".joblib")
        if not self._is_predict:
            algo.fit(self.X)
            algo.save(algo_path)
        else:
            algo = algo.load(algo_path)
        Xp = algo.transform(self.X)
        file_name = os.path.join(self.path, DESCRIPTORS_SUBFOLDER, algo._name + ".h5")
        data = Data()
        data.set(inputs=self.inputs, keys=self.keys, values=Xp, features=None)
        Hdf5(file_name).save(data)
        data.save_info(file_name.split(".")[0] + ".json")

    def run(self):
        file_name = os.path.join(self.path, DESCRIPTORS_SUBFOLDER, REFERENCE_FILE_NAME)
        data = Hdf5(file_name).load()
        self.keys = data.keys()
        self.inputs = data.inputs()
        self.X = data.values()
        if not self._is_predict:
            self.trained_path = os.path.join(self.path, DESCRIPTORS_SUBFOLDER)
        else:
            self.trained_path = os.path.join(self.trained_path, DESCRIPTORS_SUBFOLDER)
        algo = Pca()
        self._algo(algo)
        algo = Umap()
        self._algo(algo)
