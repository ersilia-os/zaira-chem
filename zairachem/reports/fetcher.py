import os
import pandas as pd
import collections
import json

from ..estimators.evaluate import ResultsIterator
from ..vars import (
    DATA_FILENAME,
    DATA_SUBFOLDER,
    ESTIMATORS_SUBFOLDER,
    POOL_SUBFOLDER,
    RESULTS_FILENAME,
)
from ..estimators import RESULTS_UNMAPPED_FILENAME
from ..setup import PARAMETERS_FILE
from ..setup.tasks import USED_CUTS_FILE

from .. import ZairaBase


class ResultsFetcher(ZairaBase):
    def __init__(self, path):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        self.trained_path = self.get_trained_dir()
        self.individual_results_iterator = ResultsIterator(path=self.path)

    def _read_data(self):
        df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
        return df

    def _read_pooled_results(self):
        df = pd.read_csv(os.path.join(self.path, POOL_SUBFOLDER, RESULTS_FILENAME))
        return df

    def _read_individual_estimator_results(self, task):
        prefixes = []
        R = []
        for rpath in ResultsIterator(path=self.path).iter_relpaths():
            prefixes += ["-".join(rpath)]
            file_name = "/".join(
                [self.path, ESTIMATORS_SUBFOLDER] + rpath + [RESULTS_UNMAPPED_FILENAME]
            )
            df = pd.read_csv(file_name)
            R += [list(df[task])]
        d = collections.OrderedDict()
        for i in range(len(R)):
            d[prefixes[i]] = R[i]
        return pd.DataFrame(d)

    def _read_processed_data(self):
        df = pd.read_csv(os.path.join(self.path, POOL_SUBFOLDER, DATA_FILENAME))
        return df

    def _read_processed_data_train(self):
        df = pd.read_csv(os.path.join(self.trained_path, POOL_SUBFOLDER, DATA_FILENAME))
        return df

    def get_tasks(self):
        df = self._read_data()
        tasks = [
            c
            for c in list(df.columns)
            if ("clf_" in c or "reg_" in c) and "_skip" not in c and "_aux" not in c
        ]
        return tasks

    def get_reg_tasks(self):
        df = self._read_data()
        tasks = [
            c
            for c in list(df.columns)
            if "reg_" in c and "_skip" not in c and "_aux" not in c
        ]
        return tasks

    def get_clf_tasks(self):
        df = self._read_data()
        tasks = [
            c
            for c in list(df.columns)
            if "clf_" in c and "_skip" not in c and "_aux" not in c
        ]
        return tasks

    def get_actives_inactives(self):
        df = self._read_data()
        for c in list(df.columns):
            if "clf" in c and "_skip" not in c and "_aux" not in c:
                return list(df[c])

    def get_raw(self):
        df = self._read_data()
        for c in list(df.columns):
            if "reg" in c and "raw" in c:
                return list(df[c])

    def get_transformed(self):
        df = self._read_data()
        for c in list(df.columns):
            if "reg" in c and "_skip" not in c and "_aux" not in c:
                return list(df[c])

    def get_pred_binary_clf(self):
        df = self._read_pooled_results()
        for c in list(df.columns):
            if "clf" in c and "bin" in c:
                return list(df[c])

    def get_pred_proba_clf(self):
        df = self._read_pooled_results()
        for c in list(df.columns):
            if "clf" in c and "bin" not in c:
                return list(df[c])

    def get_pred_reg_trans(self):
        df = self._read_pooled_results()
        for c in list(df.columns):
            if "reg" in c and "raw" not in c:
                return list(df[c])

    def get_pred_reg_raw(self):
        df = self._read_pooled_results()
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

    def get_projections_trained(self):
        df = self._read_processed_data_train()
        for c in list(df.columns):
            if "umap-0" in c:
                umap0 = list(df["umap-0"])
            if "umap-1" in c:
                umap1 = list(df["umap-1"])
        return umap0, umap1

    def get_parameters(self):
        with open(os.path.join(self.trained_path, DATA_SUBFOLDER, PARAMETERS_FILE), "r") as f:
            return json.load(f)

    def get_direction(self):
        return self.get_parameters()["direction"]

    def get_used_cuts(self):
        with open(os.path.join(self.trained_path, DATA_SUBFOLDER, USED_CUTS_FILE), "r") as f:
            return json.load(f)

    def get_used_cut(self):
        used_cuts = self.get_used_cuts()
        for k,v in used_cuts["ecuts"].items():
            if "_skip" not in k:
                return v
        for k,v in used_cuts["pcuts"].items():
            if "_skip" not in k:
                return v
