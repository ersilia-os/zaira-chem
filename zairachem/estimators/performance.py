import os
import numpy as np
import pandas as pd

from sklearn import metrics

from . import RESULTS_UNMAPPED_FILENAME
from .. import ZairaBase

from ..vars import DATA_SUBFOLDER, MODELS_SUBFOLDER, DATA_FILENAME

from . import CLF_REPORT_FILENAME, REG_REPORT_FILENAME


class BasePerformance(ZairaBase):

    def __init__(self, path=None):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path

    def _relevant_columns(self, df):
        return [c for c in list(df.columns) if "clf_" in c or "reg_" in c]

    def get_obs_data(self):
        df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
        return df[self._relevant_columns(df)]

    def get_prd_data(self):
        df = pd.read_csv(os.path.join(self.path, MODELS_SUBFOLDER, RESULTS_UNMAPPED_FILENAME))
        return df[self._relevant_columns(df)]


class ClassificationPerformance(BasePerformance):

    def __init__(self, path):
        BasePerformance.__init__(self, path=path)
        self.df_obs = self.get_obs_data()
        self.df_prd = self.get_prd_data()
        self._prefix = self._get_prefix()

    def _get_prefix(self):
        for c in list(self.df_obs.columns):
            if "clf_" in c:
                return c

    def calculate(self):
        y_true = np.array(self.df_obs[self._prefix])
        y_pred = np.array(self.df_prd[self._prefix])
        b_pred = np.array(self.df_prd[self._prefix+"_bin"])
        confu = metrics.confusion_matrix(y_true, b_pred, labels=[0,1])
        report = {
            "roc_auc_score": metrics.roc_auc_score(y_true, y_pred),
            "precision_score": metrics.precision_score(y_true, b_pred),
            "recall_score": metrics.recall_score(y_true, b_pred),
            "tp": confu[1,1],
            "tn": confu[0,0],
            "fp": confu[0,1],
            "fn": confu[1,0],
            "y_true": list(y_true),
            "y_pred": list(y_pred),
            "b_pred": list(b_pred),
        }
        return report


class RegressionPerformance(BasePerformance):

    def __init__(self, path):
        BasePerformance.__init__(self, path=path)
        self.df_obs = self.get_obs_data()
        self.df_prd = self.get_prd_data()
        self._prefix = self._get_prefix()

    def _get_prefix(self):
        for c in list(self.df_obs.columns):
            if "reg_" in c:
                return c

    def calculate(self):
        y_true = np.array(self.df_obs[self._prefix])
        y_pred = np.array(self.df_prd[self._prefix])
        report = {
            "r2_score": metrics.r2_score(y_true, y_pred),
            "mean_absolute_error": metrics.mean_absolute_error(y_true, y_pred),
            "mean_squared_error": metrics.mean_squared_error(y_true, y_pred),
            "y_true": list(y_true),
            "y_pred": list(y_pred),
        }
        return report


class PerformanceReporter(object):

    def __init__(self, path=None):
        self.clf = ClassificationPerformance(path=path)
        self.reg = RegressionPerformance(path=path)

    def run(self):
        clf_rep = self.clf.calculate()
        with open(os.path.join(self.path, MODELS_SUBFOLDER, CLF_REPORT_FILENAME), "w") as f:
            json.dump(clf_rep, f, indent=4)
        reg_rep = self.reg.calculate()
        with open(os.path.join(self.path, MODELS_SUBFOlder, REG_REPORT_FILENAME), "w") as f:
            json.dump(reg_rep, f, indent=4)
