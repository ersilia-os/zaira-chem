import os
import numpy as np
import joblib
import shutil
import json
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold
import tempfile
import h5py

from . import DescriptorBase
from .raw import DESCRIPTORS_SUBFOLDER, RawLoader
from ..utils.matrices import Hdf5
from .. import ZairaBase
from ..tools.fpsim2.searcher import SimilaritySearcher


TREATED_FILE_NAME = "treated.h5"
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
        print(
            "Nan filtering, original columns {0}, final columns {1}".format(
                X.shape[1], len(self.col_idxs)
            )
        )

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


class FullLineSimilarityImputer(object):
    def __init__(self):
        self._name = "full_line_imputer"
        self._prefix = "fp2sim"
        self._n_hits = 3
        self._sim_cutoff = 0.5

    def _set_filenames(self, basedir):
        self.basedir = basedir
        self.fp_filename = os.path.join(basedir, "{0}.h5".format(self._prefix))
        self.smiles_filename = os.path.join(basedir, "{0}.csv".format(self._prefix))
        self.x_notnan_filename = os.path.join(
            basedir, "{0}_X_notnan.h5".format(self._prefix)
        )

    def _get_filenames(self):
        return (
            self.basedir,
            self.fp_filename,
            self.smiles_filename,
            self.x_notnan_filename,
        )

    def fit(self, X, smiles_list):
        idxs = []
        for i in range(X.shape[0]):
            if np.all(np.isnan(X[i, :])):
                continue
            else:
                idxs += [i]
        X_ = X[idxs]
        smiles_list_ = [smiles_list[i] for i in idxs]
        tmp_dir = tempfile.mkdtemp(prefix="ersilia-")
        self._set_filenames(tmp_dir)
        similarity_searcher = SimilaritySearcher(fp_filename=self.fp_filename)
        similarity_searcher.fit(smiles_list_)
        with h5py.File(self.x_notnan_filename, "w") as f:
            f.create_dataset("Values", data=X_)

    def transform(self, X, smiles_list):
        with h5py.File(self.x_notnan_filename, "r") as f:
            R = f["Values"][:]
        idxs = []
        for i in range(X.shape[0]):
            if np.all(np.isnan(X[i, :])):
                idxs += [i]
            else:
                continue
        similarity_searcher = SimilaritySearcher(fp_filename=self.fp_filename)
        for idx in idxs:
            smi = smiles_list[idx]
            hits = similarity_searcher.search(smi, cutoff=self._sim_cutoff)
            hits = np.array([x[0] for x in hits][: self._n_hits])
            print(hits)
            if len(hits) == 0:
                hits = np.random.choice(
                    [i for i in range(R.shape[0])],
                    min(R.shape[0], self._n_hits),
                    replace=False,
                )
            r = np.mean(R[hits, :], axis=0)
            X[idx, :] = r
        return X

    def save(self, file_name):
        _, fp_filename, smiles_filename, x_notnan_filename = self._get_filenames()
        basedir = os.path.dirname(file_name)
        self._set_filenames(basedir)
        shutil.move(fp_filename, self.fp_filename)
        shutil.move(smiles_filename, self.smiles_filename)
        shutil.move(x_notnan_filename, self.x_notnan_filename)
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


class TreatedLoader(ZairaBase):
    def __init__(self):
        ZairaBase.__init__(self)
        self.path = self.get_output_dir()

    def open(self, eos_id):
        path = os.path.join(self.path, DESCRIPTORS_SUBFOLDER, eos_id, TREATED_FILE_NAME)
        return Hdf5(path)


class TreatedDescriptors(DescriptorBase):
    def __init__(self):
        DescriptorBase.__init__(self)
        self.pipeline = [
            (0, -1, NanFilter),
            (1, 0, Imputer),
            (2, 1, VarianceFilter),
            (3, 2, Scaler),
        ]
        self._name = TREATED_FILE_NAME
        self._is_predict = self.is_predict()

    def done_eos_iter(self):
        with open(
            os.path.join(self.path, DESCRIPTORS_SUBFOLDER, "done_eos.json"), "r"
        ) as f:
            data = json.load(f)
        for eos_id in data:
            yield eos_id

    def run(self):
        rl = RawLoader()
        for eos_id in self.done_eos_iter():
            path = os.path.join(self.path, DESCRIPTORS_SUBFOLDER, eos_id)
            if not self._is_predict:
                trained_path = path
            else:
                trained_path = os.path.join(
                    self.trained_path, DESCRIPTORS_SUBFOLDER, eos_id
                )
            data = rl.open(eos_id)
            data = data.load()
            X = {}
            X[-1] = data.values()
            for step in self.pipeline:
                curr, prev = step[0], step[1]
                algo = step[-1]()
                if algo._name == "scaler":
                    if data.is_sparse():
                        print(
                            "Skipping normalization of {0} as it is sparse".format(
                                eos_id
                            )
                        )
                        algo.set_skip()
                if not self._is_predict:
                    algo.fit(X[prev])
                    algo.save(os.path.join(trained_path, algo._name + ".joblib"))
                else:
                    algo = algo.load(os.path.join(trained_path, algo._name + ".joblib"))
                X[curr] = algo.transform(X[prev])
            file_name = os.path.join(
                self.path, DESCRIPTORS_SUBFOLDER, eos_id, self._name
            )
            data._values = X[len(self.pipeline) - 1]
            Hdf5(file_name).save(data)
            data.save_info(file_name.split(".")[0] + ".json")
