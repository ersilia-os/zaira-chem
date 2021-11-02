import h5py


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
