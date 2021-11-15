import os
import joblib
import pandas as pd
import numpy as np
from lol import LOL
from sklearn.decomposition import PCA
from umap import UMAP

from . import DescriptorBase
from ..utils.matrices import Hdf5, Data

from .raw import DESCRIPTORS_SUBFOLDER
from .unsupervised import GLOBAL_UNSUPERVISED_FILE_NAME
from ..setup import AUXILIARY_TASK_COLUMN
from ..vars import DATA_SUBFOLDER, DATA_FILENAME

from . import GLOBAL_SUPERVISED_FILE_NAME

MAX_COMPONENTS = 512
MAX_COMPONENTS_LOLLIPOP_FACTOR = 0.8


class RfeCv(object):
    def __init__(self):
        pass

    def fit(self):
        pass

    def transform(self):
        pass


class LolliPop(object):
    def __init__(self):
        self._name = "lollipop"

    def fit(self, X, y):
        n_components = np.min(
            [
                MAX_COMPONENTS,
                X.shape[0],
                int(X.shape[1] * MAX_COMPONENTS_LOLLIPOP_FACTOR),
            ]
        )
        self.lmao = LOL(n_components=n_components, svd_solver="full")
        self.lmao.fit(X, y)
        X = self.lmao.transform(X)
        self.pca = PCA(n_components=X.shape[1], whiten=True)
        self.pca.fit(X)

    def transform(self, X):
        X = self.lmao.transform(X)
        X = self.pca.transform(X)
        return X

    def save(self, file_name):
        joblib.dump(self, file_name)

    def load(self, file_name):
        return joblib.load(file_name)


class SupervisedUmap(object):
    def __init__(self):
        self._name = "supervised_umap"

    def fit(self, X, y):
        self.reducer = UMAP(densmap=False)
        self.reducer.fit(X, y)

    def transform(self, X):
        return self.reducer.transform(X)

    def save(self, file_name):
        joblib.dump(self, file_name)

    def load(self, file_name):
        return joblib.load(file_name)


class SupervisedTransformations(DescriptorBase):
    def __init__(self):
        DescriptorBase.__init__(self)
        self.pipeline = None  # TODO

    def load_y(self):
        file_name = os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME)
        return np.array(pd.read_csv(file_name)[AUXILIARY_TASK_COLUMN])

    def run(self):
        data = Hdf5(
            os.path.join(
                self.path, DESCRIPTORS_SUBFOLDER, GLOBAL_UNSUPERVISED_FILE_NAME
            )
        ).load()
        X = data.values()
        y = self.load_y()
        algo = LolliPop()
        algo.fit(X, y)
        algo.save(
            os.path.join(self.path, DESCRIPTORS_SUBFOLDER, algo._name + ".joblib")
        )
        X = algo.transform(X)
        data_ = Data()
        data_.set(inputs=data.inputs(), keys=data.keys(), values=X, features=None)
        file_name = os.path.join(
            self.path, DESCRIPTORS_SUBFOLDER, GLOBAL_SUPERVISED_FILE_NAME
        )
        Hdf5(file_name).save(data_)
        data_.save_info(file_name.split(".")[0] + ".json")
