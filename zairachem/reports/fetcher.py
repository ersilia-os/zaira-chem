import os
import pandas as pd

from zairachem.vars import DATA_FILENAME, DATA_SUBFOLDER, POOL_SUBFOLDER, RESULTS_FILENAME

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

    def _read_results(self):
        df = pd.read_csv(os.path.join(self.path, POOL_SUBFOLDER, RESULTS_FILENAME))
        return df
    
    def _read_processed_data(self):
        df = pd.read_csv(os.path.join(self.path, POOL_SUBFOLDER, DATA_FILENAME))
        return df

    def get_actives_inactives(self):
        df = self._read_data()
        for c in list(df.columns):
            if "clf" in c and "skip" not in c:
                return list(df[c])

    def get_raw(self):
        df = self._read_data()
        for c in list(df.columns):
            if "reg" in c and "raw" in c:
                return list(df[c])

    def get_transformed(self):
        df = self._read_data()
        for c in list(df.columns):
            if "reg" in c and "skip" not in c:
                return list(df[c])

    def get_pred_binary_clf(self):
        df = self._read_results()
        for c in list(df.columns):
            if "clf" in c and "bin" in c:
                return list(df[c])

    def get_pred_proba_clf(self):
        df = self._read_results()
        for c in list(df.columns):
            if "clf" in c and "bin" not in c:
                return list(df[c])

    def get_pred_reg_trans(self):
        df = self._read_results()
        for c in list(df.columns):
            if "reg" in c and "raw" not in c:
                return list(df[c])
    
    def get_pred_reg_raw(self):
        df = self._read_results()
        for c in list(df.columns):
            if "reg" in c and "raw" in c:
                return list(df[c])

    def get_projections(self):
        df = self._read_processed_data()
        for c in list(df.columns):
            if "umap-0" in c:
                umap0 = list(df["umap-0"])
            if "umap-1" in c:
                umap1 = list(df["umap-1"])
        return umap0, umap1