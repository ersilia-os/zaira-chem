import os
import numpy as np
import pandas as pd
import json
import joblib
import h5py

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegressionCV

from .. import ZairaBase
from ..estimators.evaluate import ResultsIterator
from ..estimators.base import BaseEstimator
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

BAGGER_SUBFOLDER = "bagger"


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
        self.trained_path = os.path.join(self.get_output_dir(), POOL_SUBFOLDER)

    def _get_X(self):
        df = XGetter(path=self.path).get()
        return df

    def _get_X_clf(self, df):
        return df[[c for c in list(df.columns) if "clf_" in c and "_bin" not in c]]

    def _get_X_reg(self, df):
        return df[[c for c in list(df.columns) if "reg_" in c]]

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

    def _get_Y_clf(self, df):
        return df[[c for c in list(df.columns) if "clf_" in c]]

    def _get_Y_reg(self, df):
        return df[[c for c in list(df.columns) if "reg_" in c]]

    def run(self, time_budget_sec=None):
        self.reset_time()
        if time_budget_sec is None:
            time_budget_sec = self._estimate_time_budget()
        else:
            time_budget_sec = time_budget_sec
        valid_idxs = self.get_validation_indices(path=self.path)
        df_X = self._get_X()
        df_X_reg = self._get_X_reg(df_X)
        df_X_clf = self._get_X_clf(df_X)
        df_Y = self._get_Y()
        df_Y_reg = self._get_Y_reg(df_Y)
        df_Y_clf = self._get_Y_clf(df_Y)
        # regression
        X_reg = np.array(df_X_reg)
        Y_reg = np.array(df_Y_reg)
        reg = LinearRegression()
        reg.fit(X_reg[valid_idxs], Y_reg[valid_idxs])
        Y_reg_hat = reg.predict(X_reg)
        # classification
        X_clf = np.array(df_X_clf)
        Y_clf = np.array(df_Y_clf)
        clf = MultiOutputClassifier(LogisticRegressionCV())
        clf.fit(X_clf[valid_idxs], Y_clf[valid_idxs])
        B_clf_hat = clf.predict(X_clf)
        Y_clf_hat = np.zeros(Y_clf.shape)
        for j, yh in enumerate(clf.predict_proba(X_clf)):
            Y_clf_hat[:, j] = yh[:, 1]
        path_ = os.path.join(self.path, POOL_SUBFOLDER, BAGGER_SUBFOLDER)
        if not os.path.exists(path_):
            os.makedirs(path_)
        joblib.dump(reg, os.path.join(path_, "reg.joblib"))
        joblib.dump(clf, os.path.join(path_, "clf.joblib"))
        with open(
            os.path.join(self.path, POOL_SUBFOLDER, BAGGER_SUBFOLDER, "columns.json"),
            "w",
        ) as f:
            columns = {"reg": list(df_Y_reg.columns), "clf": list(df_Y_clf.columns)}
            json.dump(columns, f)
        P = []
        columns = []
        for j, c in enumerate(list(df_Y_reg.columns)):
            columns += [c]
            P += [Y_reg_hat[:, j]]
        for j, c in enumerate(list(df_Y_clf.columns)):
            columns += [c, c + "_bin"]
            P += [Y_clf_hat[:, j]]
            P += [B_clf_hat[:, j]]
        P = np.array(P).T
        results = pd.DataFrame(P, columns=columns)
        self.update_elapsed_time()
        return results


class Predictor(BaseEstimator):
    def __init__(self, path):
        BaseEstimator.__init__(self, path=path)
        self.trained_path = os.path.join(
            self.get_trained_dir(), POOL_SUBFOLDER, BAGGER_SUBFOLDER
        )

    def _get_X(self):
        df = XGetter(path=self.path).get()
        return df

    def _get_X_clf(self, df):
        return df[[c for c in list(df.columns) if "clf_" in c and "_bin" not in c]]

    def _get_X_reg(self, df):
        return df[[c for c in list(df.columns) if "reg_" in c]]

    def run(self):
        self.reset_time()
        df = self._get_X()
        df_X_clf = self._get_X_clf(df)
        df_X_reg = self._get_X_reg(df)
        with open(os.path.join(self.trained_path, "columns.json"), "r") as f:
            columns = json.load(f)
            reg_cols = columns["reg"]
            clf_cols = columns["clf"]
        # regression
        X_reg = np.array(df_X_reg)
        reg = joblib.load(os.path.join(self.trained_path, "reg.joblib"))
        Y_reg_hat = reg.predict(X_reg)
        # classification
        X_clf = np.array(df_X_clf)
        clf = joblib.load(os.path.join(self.trained_path, "clf.joblib"))
        B_clf_hat = clf.predict(X_clf)
        Y_clf_hat = np.zeros(B_clf_hat.shape)
        for j, yh in enumerate(clf.predict_proba(X_clf)):
            Y_clf_hat[:, j] = yh[:, 1]
        path_ = os.path.join(self.path, POOL_SUBFOLDER, BAGGER_SUBFOLDER)
        if not os.path.exists(path_):
            os.makedirs(path_)
        joblib.dump(reg, os.path.join(path_, "reg.joblib"))
        joblib.dump(clf, os.path.join(path_, "clf.joblib"))
        P = []
        columns = []
        for j, c in enumerate(reg_cols):
            columns += [c]
            P += [Y_reg_hat[:, j]]
        for j, c in enumerate(clf_cols):
            columns += [c, c + "_bin"]
            P += [Y_clf_hat[:, j]]
            P += [B_clf_hat[:, j]]
        P = np.array(P).T
        results = pd.DataFrame(P, columns=columns)
        self.update_elapsed_time()
        return results
