import joblib
import h5py

SNIFF_N = 100000


class Data(object):

    def __init__(self):
        pass

    def set(self, keys, inputs, values, features):
        self._keys = keys
        self._inputs = inputs
        self._values = values
        self._features = features

    def keys(self):
        return self._keys

    def inputs(self):
        return self._inputs

    def values(self):
        return self._values

    def features(self):
        return self._features

    def save(self, file_name):
        joblib.dump(self, file_name)

    def load(self, file_name):
        return joblib.load(file_name)


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
        if n_zeroes/len(V) > 0.5:
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
        data.set(keys=self.keys(), inputs=self.inputs(), values=self.values(), features=self.features())
        return data

    def save(self, data):
        with h5py.File(self.file_name, "w") as f:
            f.create_dataset("Values", data=data.values())
            f.create_dataset("Keys", data=data.keys())
            f.create_dataset("Inputs", data=data.inputs())
            f.create_dataset("Features", data=data.features())
