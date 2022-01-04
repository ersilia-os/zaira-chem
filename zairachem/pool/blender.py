import os
import numpy as np
import pandas as pd
import json
import h5py

from .. import ZairaBase
from ..estimators.evaluate import ResultsIterator
from ..estimators.base import BaseEstimator
from ..automl.autogluon import AutoGluonEstimator
from ..estimators import RESULTS_UNMAPPED_FILENAME
from ..setup import COMPOUND_IDENTIFIER_COLUMN, SMILES_COLUMN
from ..vars import (
    DATA_FILENAME,
    DATA_SUBFOLDER,
    DESCRIPTORS_SUBFOLDER,
    ESTIMATORS_SUBFOLDER,
    POOL_SUBFOLDER,
)

RESULTS_VALIDATION_FILENAME = "results_validation.csv"
AUTOGLUON_SAVE_SUBFOLDER = "autogluon"


class XGetter(ZairaBase):
    def __init__(self, path):
        ZairaBase.__init__(self)
        self.path = path
        self.X = []
        self.columns = []

    @staticmethod
    def _read_results_file(file_path):
        df = pd.read_csv(file_path)
        df = df[
            [
                c
                for c in list(df.columns)
                if c not in [SMILES_COLUMN, COMPOUND_IDENTIFIER_COLUMN]
            ]
        ]
        return df

    def _read_model_ids(self):
        with open(
            os.path.join(self.path, DESCRIPTORS_SUBFOLDER, "done_eos.json"), "r"
        ) as f:
            model_ids = list(json.load(f))
        return model_ids

    def _get_manifolds(self):
        with h5py.File(
            os.path.join(self.path, DESCRIPTORS_SUBFOLDER, "pca.h5"), "r"
        ) as f:
            X_ = f["Values"][:]
            self.X += [X_]
            for i in range(X_.shape[1]):
                self.columns += ["pca-{0}".format(i)]
        with h5py.File(
            os.path.join(self.path, DESCRIPTORS_SUBFOLDER, "umap.h5"), "r"
        ) as f:
            X_ = f["Values"][:]
            self.X += [X_]
            for i in range(X_.shape[1]):
                self.columns += ["umap-{0}".format(i)]

    def _get_results(self):
        prefixes = []
        dfs = []
        for rpath in ResultsIterator(path=self.path).iter_relpaths():
            prefixes += ["-".join(rpath)]
            file_name = "/".join(
                [self.path, ESTIMATORS_SUBFOLDER] + rpath + [RESULTS_UNMAPPED_FILENAME]
            )
            dfs += [self._read_results_file(file_name)]
        for i in range(len(dfs)):
            df = dfs[i]
            prefix = prefixes[i]
            self.X += [np.array(df)]
            self.columns += ["{0}-{1}".format(prefix, c) for c in list(df.columns)]
        self.logger.debug(
            "Number of columns: {0} ... from {1} estimators".format(
                len(self.columns), len(dfs)
            )
        )

    def get(self):
        self._get_manifolds()
        self._get_results()
        X = np.hstack(self.X)
        df = pd.DataFrame(X, columns=self.columns)
        df.to_csv(os.path.join(self.path, POOL_SUBFOLDER, DATA_FILENAME), index=False)
        return df


class Fitter(BaseEstimator):
    def __init__(self, path):
        BaseEstimator.__init__(self, path=path)
        self.trained_path = os.path.join(
            self.get_output_dir(), POOL_SUBFOLDER, AUTOGLUON_SAVE_SUBFOLDER
        )

    def _get_X(self):
        df = XGetter(path=self.path).get()
        return df

    def _get_y(self, task):
        df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
        return np.array(df[task])

    def _get_Y(self):
        Y = []
        columns = []
        for t in self._get_reg_tasks():
            y = self._get_y(t)
            Y += [y]
            columns += [t]
        for t in self._get_clf_tasks():
            y = self._get_y(t)
            Y += [y]
            columns += [t]
        Y = np.array(Y).T
        df = pd.DataFrame(Y, columns=columns)
        return df

    def run(self, time_budget_sec=None):
        self.reset_time()
        if time_budget_sec is None:
            time_budget_sec = self._estimate_time_budget()
        else:
            time_budget_sec = time_budget_sec
        valid_idxs = self.get_validation_indices(path=self.path)
        df_X = self._get_X()
        df_Y = self._get_Y()
        df = pd.concat([df_X, df_Y], axis=1)
        labels = list(df_Y.columns)
        self.logger.debug("Staring AutoGluon estimation")
        estimator = AutoGluonEstimator(
            save_path=self.trained_path, time_budget=time_budget_sec
        )
        self.logger.debug("Fitting")
        estimator.fit(data=df.iloc[valid_idxs, :], labels=labels, groups=None)
        results_oos = estimator.get_out_of_sample()
        results_oos.to_csv(
            os.path.join(self.path, POOL_SUBFOLDER, RESULTS_VALIDATION_FILENAME),
            index=False,
        )
        estimator = estimator.load()
        results = estimator.run(df)
        self.update_elapsed_time()
        return results


class Predictor(BaseEstimator):
    def __init__(self, path):
        BaseEstimator.__init__(self, path=path)
        self.trained_path = os.path.join(
            self.get_trained_dir(), POOL_SUBFOLDER, AUTOGLUON_SAVE_SUBFOLDER
        )

    def run(self):
        self.reset_time()
        df = XGetter(path=self.path).get()
        model = AutoGluonEstimator(save_path=self.trained_path).load()
        results = model.run(df)
        self.update_elapsed_time()
        return results
