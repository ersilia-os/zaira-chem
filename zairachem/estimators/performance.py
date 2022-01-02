import os
import json
import numpy as np
import pandas as pd
import joblib
import collections

from sklearn import metrics

from . import Y_HAT_FILE
from .. import ZairaBase

from ..vars import (
    DATA_SUBFOLDER,
    DESCRIPTORS_SUBFOLDER,
    MODELS_SUBFOLDER,
    DATA_FILENAME,
)

from . import CLF_REPORT_FILENAME, REG_REPORT_FILENAME


class BasePerformance(ZairaBase):
    def __init__(self, path=None, model_id=None):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        self.model_id = model_id

    def _get_y_hat_dict(self):
        return joblib.load(
            os.path.join(self.path, MODELS_SUBFOLDER, self.model_id, Y_HAT_FILE)
        )


class ClassificationPerformance(BasePerformance):
    def __init__(self, path, model_id):
        BasePerformance.__init__(self, path=path, model_id=model_id)
        self.results = self._get_y_hat_dict()
        self._prefix = self._get_prefix()
        self.results = self.results[self._prefix]

    def _get_prefix(self):
        for c in list(self.results.keys()):
            if "clf_" in c:
                return c

    def _try_metric(self, fun, t, p):
        try:
            return float(fun(t, p))
        except:
            return None

    def _calculate(self, key):
        r = self.results[key]
        y_true = np.array(r["y"])
        y_pred = np.array(r["y_hat"])
        b_pred = np.array(r["b_hat"])
        try:
            confu = metrics.confusion_matrix(y_true, b_pred, labels=[0, 1])
        except:
            confu = np.array([[-1, -1], [-1, -1]])
        report = {
            "roc_auc_score": self._try_metric(metrics.roc_auc_score, y_true, y_pred),
            "precision_score": self._try_metric(
                metrics.precision_score, y_true, b_pred
            ),
            "recall_score": self._try_metric(metrics.recall_score, y_true, b_pred),
            "tp": int(confu[1, 1]),
            "tn": int(confu[0, 0]),
            "fp": int(confu[0, 1]),
            "fn": int(confu[1, 0]),
            "y_true": [int(y) for y in y_true],
            "y_pred": [float(y) for y in y_pred],
            "b_pred": [int(y) for y in b_pred],
        }
        return report

    def calculate(self):
        report = collections.OrderedDict()
        for k in self.results.keys():
            report[k] = self._calculate(k)
        return report


class RegressionPerformance(BasePerformance):
    def __init__(self, path, model_id):
        BasePerformance.__init__(self, path=path, model_id=model_id)
        self.results = self._get_y_hat_dict()
        self._prefix = self._get_prefix()
        self.results = self.results[self._prefix]

    def _get_prefix(self):
        for c in list(self.results.keys()):
            if "reg_" in c:
                return c

    def _calculate(self, key):
        r = self.results[key]
        y_true = np.array(r["y"])
        y_pred = np.array(r["y_hat"])
        report = {
            "r2_score": float(metrics.r2_score(y_true, y_pred)),
            "mean_absolute_error": float(metrics.mean_absolute_error(y_true, y_pred)),
            "mean_squared_error": float(metrics.mean_squared_error(y_true, y_pred)),
            "y_true": [float(y) for y in y_true],
            "y_pred": [float(y) for y in y_pred],
        }
        return report

    def calculate(self):
        report = collections.OrderedDict()
        for k in self.results.keys():
            print(k)
            report[k] = self._calculate(k)
        return report


class IndividualPerformanceReporter(ZairaBase):
    def __init__(self, path=None, model_id=None):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        self.has_tasks = self._has_tasks()
        if self._has_clf_tasks():
            self.clf = ClassificationPerformance(path=path, model_id=model_id)
        else:
            self.clf = None
        if self._has_reg_tasks():
            self.reg = RegressionPerformance(path=path, model_id=model_id)
        else:
            self.reg = None
        self.model_id = model_id

    def _has_tasks(self):
        df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
        for c in list(df.columns):
            if "clf_" in c or "reg_" in c:
                return True
        return False

    def _has_reg_tasks(self):
        df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
        for c in list(df.columns):
            if "reg_" in c:
                return True
        return False

    def _has_clf_tasks(self):
        df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
        for c in list(df.columns):
            if "clf_" in c:
                return True
        return False

    def run(self):
        if not self.has_tasks:
            return
        if self.clf is not None:
            clf_rep = self.clf.calculate()
            with open(
                os.path.join(
                    self.path, MODELS_SUBFOLDER, self.model_id, CLF_REPORT_FILENAME
                ),
                "w",
            ) as f:
                json.dump(clf_rep, f, indent=4)
        if self.reg is not None:
            reg_rep = self.reg.calculate()
            with open(
                os.path.join(
                    self.path, MODELS_SUBFOLDER, self.model_id, REG_REPORT_FILENAME
                ),
                "w",
            ) as f:
                json.dump(reg_rep, f, indent=4)


class PerformanceReporter(ZairaBase):
    def __init__(self, path=None):
        ZairaBase.__init__(self)
        self.path = path

    def _get_model_ids(self):
        if self.path is None:
            path = self.get_output_dir()
        else:
            path = self.path
        if self.is_predict():
            path = self.get_trained_dir()
        with open(os.path.join(path, DESCRIPTORS_SUBFOLDER, "done_eos.json"), "r") as f:
            model_ids = list(json.load(f))
        return model_ids

    def run(self):
        model_ids = self._get_model_ids()
        for model_id in model_ids:
            p = IndividualPerformanceReporter(path=self.path, model_id=model_id)
            p.run()
