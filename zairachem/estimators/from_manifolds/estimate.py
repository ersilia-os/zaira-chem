import os
import h5py
import numpy as np
import pandas as pd

from ... import ZairaBase
from ..utils.autogluon import AutoGluonUtil
from ...vars import DATA_FILENAME, DESCRIPTORS_SUBFOLDER, ESTIMATORS_SUBFOLDER
from . import ESTIMATORS_FAMILY_SUBFOLDER


class XGetter(ZairaBase):
    def __init__(self, path):
        ZairaBase.__init__(self)
        self.path = path
        self.X = []
        self.columns = []

    def _get_manifold(self, tag):
        with h5py.File(
            os.path.join(self.path, DESCRIPTORS_SUBFOLDER, "{0}.h5".format(tag)), "r"
        ) as f:
            X_ = f["Values"][:]
            self.X += [X_]
            for i in range(X_.shape[1]):
                self.columns += ["{0}-{1}".format(tag, i)]

    def _get_manifolds(self):
        self._get_manifold("pca")
        self._get_manifold("umap")
        self._get_manifold("lolp")

    def get(self):
        self._get_manifolds()
        X = np.hstack(self.X)
        df = pd.DataFrame(X, columns=self.columns)
        df.to_csv(
            os.path.join(
                self.path,
                ESTIMATORS_SUBFOLDER,
                ESTIMATORS_FAMILY_SUBFOLDER,
                DATA_FILENAME,
            ),
            index=False,
        )
        return df


class Estimator(ZairaBase):
    def __init__(self, path=None):
        ZairaBase.__init__(self)
        self.path = path
        self.runner = AutoGluonUtil(
            path=path,
            estimators_family_subfolder=ESTIMATORS_FAMILY_SUBFOLDER,
            x_getter=XGetter,
        )

    def run(self, time_budget_sec=None):
        self.runner.run(time_budget_sec=time_budget_sec)
