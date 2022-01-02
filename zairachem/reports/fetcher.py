import os
import pandas as pd

from zairachem.vars import DATA_FILENAME, DATA_SUBFOLDER

from .. import ZairaBase


class ResultsFetcher(ZairaBase):
    def __init__(self, path):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path

    def _read_data(self):
        df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
        return df

    def get_actives_inactives(self):
        df = self._read_data()
        for c in list(df.columns):
            if "clf" in c and "skip" not in c:
                return list(df[c])

    def get_raw_smoothened(self):
        df = self._read_data()
        for c in list(df.columns):
            if "reg" in c and "raw" in c:
                return list(df[c])

    def get_transformed(self):
        df = self._read_data()
        for c in list(df.columns):
            if "reg" in c and "skip" not in c:
                return list(df[c])
