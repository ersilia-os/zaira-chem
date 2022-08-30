import os
import pandas as pd
import numpy as np
import joblib
import json
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import MolFromSmarts

from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from flaml import AutoML

from ..setup import SMILES_COLUMN
from ..vars import ZAIRACHEM_DATA_PATH

COLUMNS_FILENAME = "columns.json"

_TIME_BUDGET_SEC = 60

# Helpers
## These helpers are similar or equal to the ones in the descriptors module
## TODO: import from descriptors section

MAX_NA = 0.2


class NanFilter(object):
    def __init__(self):
        self._name = "nan_filter"

    def fit(self, X):
        max_na = int((1 - MAX_NA) * X.shape[0])
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
        self.skip = False

    def set_skip(self):
        self.skip = True

    def fit(self, X):
        if self.skip:
            return
        self.scaler = RobustScaler()
        self.scaler.fit(X)

    def transform(self, X):
        if self.skip:
            return X
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
        self.col_idxs = self.sel.transform([[i for i in range(X.shape[1])]]).ravel()

    def transform(self, X):
        return self.sel.transform(X)

    def save(self, file_name):
        joblib.dump(self, file_name)

    def load(self, file_name):
        return joblib.load(file_name)


# Descriptors


@dataclass
class Descriptors:
    """Molecular descriptors"""

    #: Descriptor type
    descriptor_type: str
    #: Descriptor values
    descriptors: tuple
    # Descriptor name
    descriptor_names: tuple
    # t_stats for each molecule
    tstats: tuple = ()


def _calculate_rdkit_descriptors(mol):
    from rdkit.ML.Descriptors import MoleculeDescriptors  # type: ignore

    dlist = [
        "NumHDonors",
        "NumHAcceptors",
        "MolLogP",
        "NumHeteroatoms",
        "RingCount",
        "NumRotatableBonds",
    ]
    c = MoleculeDescriptors.MolecularDescriptorCalculator(dlist)
    d = c.CalcDescriptors(mol)

    def calc_aromatic_bonds(mol):
        return sum(1 for b in mol.GetBonds() if b.GetIsAromatic())

    def _create_smarts(SMARTS):
        s = ",".join("$(" + s + ")" for s in SMARTS)
        _mol = MolFromSmarts("[" + s + "]")
        return _mol

    def calc_acid_groups(mol):
        acid_smarts = (
            "[O;H1]-[C,S,P]=O",
            "[*;-;!$(*~[*;+])]",
            "[NH](S(=O)=O)C(F)(F)F",
            "n1nnnc1",
        )
        pat = _create_smarts(acid_smarts)
        return len(mol.GetSubstructMatches(pat))

    def calc_basic_groups(mol):
        basic_smarts = (
            "[NH2]-[CX4]",
            "[NH](-[CX4])-[CX4]",
            "N(-[CX4])(-[CX4])-[CX4]",
            "[*;+;!$(*~[*;-])]",
            "N=C-N",
            "N-C=N",
        )
        pat = _create_smarts(basic_smarts)
        return len(mol.GetSubstructMatches(pat))

    def calc_apol(mol, includeImplicitHs=True):
        # atomic polarizabilities available here:
        # https://github.com/mordred-descriptor/mordred/blob/develop/mordred/data/polarizalibity78.txt

        ap = os.path.join(ZAIRACHEM_DATA_PATH, "atom_pols.txt")
        with open(ap, "r") as f:
            atom_pols = [float(x) for x in next(f).split(",")]
        res = 0.0
        for atom in mol.GetAtoms():
            anum = atom.GetAtomicNum()
            if anum <= len(atom_pols):
                apol = atom_pols[anum]
                if includeImplicitHs:
                    apol += atom_pols[1] * atom.GetTotalNumHs(includeNeighbors=False)
                res += apol
            else:
                raise ValueError(f"atomic number {anum} not found")
        return res

    d = d + (
        calc_aromatic_bonds(mol),
        calc_acid_groups(mol),
        calc_basic_groups(mol),
        calc_apol(mol),
    )
    return d


def classic_featurizer(smiles):
    names = tuple(
        [
            "number of hydrogen bond donor",
            "number of hydrogen bond acceptor",
            "Wildman-Crippen LogP",
            "number of heteroatoms",
            "ring count",
            "number of rotatable bonds",
            "aromatic bonds count",
            "acidic group count",
            "basic group count",
            "atomic polarizability",
        ]
    )
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    R = []
    cols = None
    for m in mols:
        descriptors = _calculate_rdkit_descriptors(m)
        descriptor_names = names
        descriptors = Descriptors(
            descriptor_type="Classic",
            descriptors=descriptors,
            descriptor_names=descriptor_names,
        )
        R += [list(descriptors.descriptors)]
        if cols is None:
            cols = list(descriptors.descriptor_names)
    data = pd.DataFrame(R, columns=cols)
    return data


class ClassicDescriptor(object):
    def __init__(self):
        self.nan_filter = NanFilter()
        self.imputer = Imputer()
        self.variance_filter = VarianceFilter()
        self.scaler = Scaler()

    def fit(self, smiles):
        df = classic_featurizer(smiles)
        X = np.array(df, dtype=np.float32)
        self.nan_filter.fit(X)
        X = self.nan_filter.transform(X)
        self.imputer.fit(X)
        X = self.imputer.transform(X)
        self.variance_filter.fit(X)
        X = self.variance_filter.transform(X)
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        self.features = list(df.columns)
        self.features = [self.features[i] for i in self.nan_filter.col_idxs]
        self.features = [self.features[i] for i in self.variance_filter.col_idxs]
        return pd.DataFrame(X, columns=self.features)

    def transform(self, smiles):
        df = classic_featurizer(smiles)
        X = np.array(df, dtype=np.float32)
        X = self.nan_filter.transform(X)
        X = self.imputer.transform(X)
        X = self.variance_filter.transform(X)
        X = self.scaler.transform(X)
        return pd.DataFrame(X, columns=self.features)


class ClassicClassifier(object):
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
        self.descriptor = ClassicDescriptor()

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


class ClassicRegressor(object):
    def __init__(
        self, automl=True, time_budget_sec=_TIME_BUDGET_SEC, estimator_list=None
    ):
        self.time_budget_sec = time_budget_sec
        self.estimator_list = estimator_list
        self.model = None
        self.explainer = None
        self._automl = automl
        self.descriptor = ClassicDescriptor()

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


class ClassicEstimator(object):
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
            self.reg_estimator = ClassicRegressor()
            self.reg_estimator.fit(X, np.array(data_reg).ravel())
        if len(data_clf.columns) == 1:
            self.clf_estimator = ClassicClassifier()
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
        return ClassicArtifact(
            reg_estimator=reg_estimator, clf_estimator=clf_estimator, columns=columns
        )


class ClassicArtifact(object):
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
