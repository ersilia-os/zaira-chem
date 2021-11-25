import joblib
import h5py
import json
import numpy as np
import pandas as pd
import collections

SNIFF_N = 100000


class Data(object):
    def __init__(self):
        self._is_sparse = None

    def _arbitrary_features(self, n):
        return ["f{0}".format(i) for i in range(n)]

    def set(self, keys, inputs, values, features):
        self._keys = keys
        self._inputs = inputs
        self._values = values
        if features is None:
            self._features = self._arbitrary_features(len(values[0]))
        else:
            self._features = features

    def keys(self):
        return self._keys

    def inputs(self):
        return self._inputs

    def values(self):
        return self._values

    def features(self):
        return self._features

    def is_sparse(self):
        return self._is_sparse

    def save(self, file_name):
        joblib.dump(self, file_name)

    def load(self, file_name):
        return joblib.load(file_name)

    def save_info(self, file_name):
        info = {
            "keys": len(self._keys),
            "inputs": len(self._inputs),
            "features": len(self._features),
            "values": np.array(self._values).shape,
            "is_sparse": self._is_sparse,
        }
        with open(file_name, "w") as f:
            json.dump(info, f, indent=4)


class Hdf5(object):
    def __init__(self, file_name):
        self.file_name = file_name

    def values(self):
        with h5py.File(self.file_name, "r") as f:
            return f["Values"][:]

    def keys(self):
        with h5py.File(self.file_name, "r") as f:
            return [x.decode("utf-8") for x in f["Keys"][:]]

    def inputs(self):
        with h5py.File(self.file_name, "r") as f:
            return [x.decode("utf-8") for x in f["Inputs"][:]]

    def features(self):
        with h5py.File(self.file_name, "r") as f:
            return [x.decode("utf-8") for x in f["Features"][:]]

    def _sniff_ravel(self):
        with h5py.File(self.file_name, "r") as f:
            V = f["Values"][:SNIFF_N]
        return V.ravel()

    def is_sparse(self):
        V = self._sniff_ravel()
        n_zeroes = np.sum(V == 0)
        if n_zeroes / len(V) > 0.8:
            return True
        return False

    def is_binary(self):
        V = self._sniff_ravel()
        vals = set(V)
        if len(vals) > 2:
            return False
        return True

    def is_dense(self):
        return not self.is_sparse()

    def load(self):
        data = Data()
        data.set(
            keys=self.keys(),
            inputs=self.inputs(),
            values=self.values(),
            features=self.features(),
        )
        data._is_sparse = self.is_sparse()
        return data

    def save(self, data):
        with h5py.File(self.file_name, "w") as f:
            f.create_dataset("Values", data=data.values())
            f.create_dataset("Keys", data=data.keys())
            f.create_dataset("Inputs", data=data.inputs())
            f.create_dataset("Features", data=data.features())

    def save_summary_as_csv(self):
        file_name = self.file_name.split(".h5")[0] + "_summary.csv"
        keys = self.keys()
        values = self.values()
        features = self.features()
        f_row = [float(x) for x in values[0]]
        l_row = [float(x) for x in values[-1]]
        means = [np.nanmean(values[:, j]) for j in range(values.shape[1])]
        stds = [np.nanstd(values[:, j]) for j in range(values.shape[1])]
        columns = ["keys"] + features
        data = collections.defaultdict(list)
        data["keys"] = ["first", "last", "mean", "std"]
        for i, f in enumerate(features):
            data[f] += [f_row[i], l_row[i], means[i], stds[i]]
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(file_name, index=False)
