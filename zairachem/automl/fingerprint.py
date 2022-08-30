import os
import pandas as pd
import numpy as np
import joblib
import json

from rdkit.Chem import rdMolDescriptors as rd
from rdkit import Chem

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from flaml import AutoML

from ..setup import SMILES_COLUMN

COLUMNS_FILENAME = "columns.json"

_TIME_BUDGET_SEC = 60

RADIUS = 3
NBITS = 2048
DTYPE = np.uint8


def clip_sparse(vect, nbits):
    l = [0] * nbits
    for i, v in vect.GetNonzeroElements().items():
        l[i] = v if v < 255 else 255
    return l


class _MorganDescriptor(object):
    def __init__(self):
        self.nbits = NBITS
        self.radius = RADIUS

    def calc(self, mol):
        v = rd.GetHashedMorganFingerprint(mol, radius=self.radius, nBits=self.nbits)
        return clip_sparse(v, self.nbits)


def morgan_featurizer(smiles):
    d = _MorganDescriptor()
    X = np.zeros((len(smiles), NBITS), dtype=DTYPE)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        X[i, :] = d.calc(mol)
    return X


class FingerprintDescriptor(object):
    def __init__(self):
        pass

    def fit(self, smiles):
        X = morgan_featurizer(smiles)
        self.features = ["fp-{0}".format(i) for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=self.features)

    def transform(self, smiles):
        X = morgan_featurizer(smiles)
        return pd.DataFrame(X, columns=self.features)


class FingerprintClassifier(object):
    def __init__(
        self, automl=True, time_budget_sec=_TIME_BUDGET_SEC, estimator_list=None
    ):
        self.time_budget_sec = time_budget_sec
        self.estimator_list = estimator_list
        self.model = None
        self.explainer = None
        self._automl = automl
        self.descriptor = FingerprintDescriptor()

    def fit_automl(self, smiles, y):
        model = AutoML(task="classification", time_budget=self.time_budget_sec)
        X = self.descriptor.fit(smiles)
        y = np.array(y)
        model.fit(
            X, y, time_budget=self.time_budget_sec, estimator_list=self.estimator_list
        )
        self._n_pos = int(np.sum(y))
        self._n_neg = len(y) - self._n_pos
        self._auroc = 1 - model.best_loss
        self.meta = {"n_pos": self._n_pos, "n_neg": self._n_neg, "auroc": self._auroc}
        self.model = model.model.estimator
        self.model.fit(X, y)

    def fit_default(self, smiles, y):
        model = RandomForestClassifier()
        X = self.descriptor.fit(smiles)
        y = np.array(y)
        model.fit(X, y)
        self.model = model

    def fit(self, smiles, y):
        if self._automl:
            self.fit_automl(smiles, y)
        else:
            self.fit_default(smiles, y)

    def predict(self, smiles):
        X = self.descriptor.transform(smiles)
        return self.model.predict_proba(X)[:, 1].reshape(-1, 1)

    def save(self, path):
        joblib.dump(self, path)

    def load(self, path):
        return joblib.load(path)


class FingerprintRegressor(object):
    def __init__(
        self, automl=True, time_budget_sec=_TIME_BUDGET_SEC, estimator_list=None
    ):
        if estimator_list is None:
            estimator_list = ["rf", "extra_tree", "xgboost", "lgbm", "lr"]
        self.time_budget_sec = time_budget_sec
        self.estimator_list = estimator_list
        self.model = None
        self.explainer = None
        self._automl = automl
        self.descriptor = FingerprintDescriptor()

    def fit_automl(self, smiles, y):
        model = AutoML(task="regression", time_budget=self.time_budget_sec)
        X = self.descriptor.fit(smiles)
        y = np.array(y)
        model.fit(
            X, y, time_budget=self.time_budget_sec, estimator_list=self.estimator_list
        )
        self._n_pos = int(np.sum(y))
        self._n_neg = len(y) - self._n_pos
        self._auroc = 1 - model.best_loss
        self.meta = {"n_pos": self._n_pos, "n_neg": self._n_neg, "auroc": self._auroc}
        self.model = model.model.estimator
        self.model.fit(X, y)

    def fit_default(self, smiles, y):
        model = RandomForestRegressor()
        X = self.descriptor.fit(smiles)
        y = np.array(y)
        model.fit(X, y)
        self.model = model

    def fit(self, smiles, y):
        if self._automl:
            self.fit_automl(smiles, y)
        else:
            self.fit_default(smiles, y)

    def predict(self, smiles):
        X = self.descriptor.transform(smiles)
        return self.model.predict(X).reshape(-1, 1)

    def save(self, path):
        joblib.dump(self, path)

    def load(self, path):
        return joblib.load(path)


class FingerprintEstimator(object):
    def __init__(self, save_path):
        self.save_path = save_path
        self.save_path_clf = os.path.join(self.save_path, "clf.joblib")
        self.save_path_reg = os.path.join(self.save_path, "reg.joblib")
        self.reg_estimator = None
        self.clf_estimator = None

    def _coltype_splitter(self, data, labels):
        x_cols = []
        clf_cols = []
        reg_cols = []
        for c in list(data.columns):
            if c not in labels:
                x_cols += [c]
            else:
                if "clf" in c:
                    clf_cols += [c]
                else:
                    reg_cols += [c]
        self.columns = {
            "smiles": [SMILES_COLUMN],
            "reg": reg_cols,
            "clf": clf_cols,
            "labels": labels,
        }
        data_x = data[x_cols]
        data_clf = data[clf_cols]
        data_reg = data[reg_cols]
        return data_x, data_clf, data_reg

    def fit(self, data, labels):
        data_smiles, data_clf, data_reg = self._coltype_splitter(data, labels)
        X = np.array(data_smiles).ravel()
        if len(data_reg.columns) == 1:
            self.reg_estimator = FingerprintRegressor()
            self.reg_estimator.fit(X, np.array(data_reg).ravel())
        if len(data_clf.columns) == 1:
            self.clf_estimator = FingerprintClassifier()
            self.clf_estimator.fit(X, np.array(data_clf).ravel())

    def save(self):
        if self.reg_estimator is not None:
            self.reg_estimator.save(self.save_path_reg)
        if self.clf_estimator is not None:
            self.clf_estimator.save(self.save_path_clf)
        with open(os.path.join(self.save_path, COLUMNS_FILENAME), "w") as f:
            json.dump(self.columns, f)

    def load(self):
        with open(os.path.join(self.save_path, COLUMNS_FILENAME), "r") as f:
            columns = json.load(f)
        if os.path.exists(self.save_path_reg):
            reg_estimator = joblib.load(self.save_path_reg)
        else:
            reg_estimator = None
        if os.path.exists(self.save_path_clf):
            clf_estimator = joblib.load(self.save_path_clf)
        else:
            clf_estimator = None
        return FingerprintArtifact(
            reg_estimator=reg_estimator, clf_estimator=clf_estimator, columns=columns
        )


class FingerprintArtifact(object):
    def __init__(self, reg_estimator, clf_estimator, columns):
        self.reg_estimator = reg_estimator
        self.clf_estimator = clf_estimator
        self.columns = columns

    def predict(self, data):
        X = np.array(data[self.columns[SMILES_COLUMN]]).ravel()
        if self.reg_estimator is not None:
            y_reg = self.reg_estimator.predict(X)
        else:
            y_reg = None
        if self.clf_estimator is not None:
            y_clf = self.clf_estimator.predict(X)
            y_clf_bin = np.zeros(y_clf.shape)
            y_clf_bin[y_clf > 0.5] = 1
        else:
            y_clf = None
            y_clf_bin = None
        P = []
        labels = []
        for label in self.columns["labels"]:
            if "clf" in label:
                idx = self.columns["clf"].index(label)
                P += [list(y_clf[:, idx])]
                P += [list(y_clf_bin[:, idx])]
                labels += [label, label + "_bin"]
            else:
                idx = self.columns["reg"].index(label)
                P += [list(y_reg[:, idx])]
                labels += [label]
        P = np.array(P).T
        df = pd.DataFrame(P, columns=labels)
        return df

    def run(self, data):
        results = self.predict(data)
        return results
