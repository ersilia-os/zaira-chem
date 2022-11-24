import json
import os
import pandas as pd
import collections

from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score

from ..vars import (
    DATA_FILENAME,
    DATA_SUBFOLDER,
    DESCRIPTORS_SUBFOLDER,
    ESTIMATORS_SUBFOLDER,
)
from . import RESULTS_UNMAPPED_FILENAME

from .. import ZairaBase

SIMPLE_EVALUATION_FILENAME = "evaluation.json"
SIMPLE_EVALUATION_VALIDATION_FILENAME = "evaluation_validation_set.json"


class ResultsIterator(ZairaBase):
    def __init__(self, path):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path

    def _read_model_ids(self):
        if self.is_lazy():
            return []
        with open(
            os.path.join(self.path, DESCRIPTORS_SUBFOLDER, "done_eos.json"), "r"
        ) as f:
            model_ids = list(json.load(f))
        return model_ids

    def iter_relpaths(self):
        estimators_folder = os.path.join(self.path, ESTIMATORS_SUBFOLDER)
        model_ids = self._read_model_ids()
        rpaths = []
        for est_fam in os.listdir(estimators_folder):
            if os.path.isdir(os.path.join(estimators_folder, est_fam)):
                focus_folder = os.path.join(estimators_folder, est_fam)
                is_individual = True
                # first look for results in the root
                for f in os.listdir(focus_folder):
                    if f == RESULTS_UNMAPPED_FILENAME:
                        rpaths += [[est_fam]]
                        is_individual = False
                # now look for results in individual predictors
                if is_individual:
                    for d in os.listdir(focus_folder):
                        if d in model_ids:
                            rpaths += [[est_fam, d]]
        for rpath in rpaths:
            yield rpath

    def iter_abspaths(self):
        for rpath in self.iter_relpaths:
            yield "/".join([self.path] + rpath)


class SimpleEvaluator(ZairaBase):
    def __init__(self, path):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        self.results_iterator = ResultsIterator(path=self.path)

    def _run(self, valid_idxs):
        df_true = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
        if valid_idxs is not None:
            df_true = df_true.iloc[valid_idxs, :]
        avail_columns = list(df_true.columns)
        for relpath in self.results_iterator.iter_relpaths():
            abspath = "/".join([self.path, ESTIMATORS_SUBFOLDER] + relpath)
            file_path = os.path.join(abspath, RESULTS_UNMAPPED_FILENAME)
            df_pred = pd.read_csv(file_path)
            if valid_idxs is not None:
                df_pred = df_pred.iloc[valid_idxs, :]
            data = collections.OrderedDict()
            for c in list(df_pred.columns):
                if "reg_" in c or "clf_" in c:
                    if c in avail_columns:
                        if "clf_" in c:
                            if len(set(df_true[c])) > 1:
                                data[c] = {
                                    "roc_auc_score": roc_auc_score(
                                        df_true[c], df_pred[c]
                                    )
                                }
                            else:
                                data[c] = 0.0
                        else:
                            data[c] = {"r2_score": r2_score(df_true[c], df_pred[c])}
            if valid_idxs is not None:
                file_name = SIMPLE_EVALUATION_VALIDATION_FILENAME
            else:
                file_name = SIMPLE_EVALUATION_FILENAME
            with open(os.path.join(abspath, file_name), "w") as f:
                json.dump(data, f, indent=4)

    def run(self):
        self._run(None)
        if not self.is_predict():
            valid_idxs = self.get_validation_indices(path=self.path)
            self._run(valid_idxs)
