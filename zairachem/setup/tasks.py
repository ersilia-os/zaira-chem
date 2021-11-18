import os
import numpy as np
import pandas as pd
from collections import OrderedDict
from scipy.stats import rankdata, gumbel_r
from scipy import interpolate
import joblib
import json

from . import (
    VALUES_FILENAME,
    VALUES_COLUMN,
    QUALIFIER_COLUMN,
    TASKS_FILENAME,
    AUXILIARY_TASK_COLUMN,
)
from .files import ParametersFile
from ..vars import CLF_PERCENTILES, MIN_CLASS, DATA_SUBFOLDER
from .. import ZairaBase


class RegTasks(object):
    def __init__(self, data, params):
        self.values = np.array(data[VALUES_COLUMN])
        self.direction = params["direction"]
        self.range = params["credibility_range"]

    def raw(self):
        min_cred = self.range["min"]
        max_cred = self.range["max"]
        if min_cred is None and max_cred is None:
            return self.values
        return np.clip(self.values, min_cred, max_cred)

    def log(self):
        raw = self.raw()
        return -np.log10(raw)

    def rnk(self):
        assert self.direction in ["high", "low"]
        if self.direction == "high":
            ranked = rankdata(self.values, method="ordinal")
        if self.direction == "low":
            ranked = rankdata(-self.values, method="ordinal")
        return ranked / np.max(ranked)

    def gum(self, rnk):
        rnk = np.array(rnk)
        sampled = gumbel_r.rvs(size=len(rnk))
        idxs = np.argsort(sampled)
        sampled = sampled[idxs]
        idxs = np.argsort(rnk)
        gum = np.zeros(len(rnk))
        for i, idx in enumerate(idxs):
            gum[idx] = sampled[i]
        gum = np.array(gum)
        return gum

    def as_dict(self):
        res = OrderedDict()
        res["reg_raw_skip"] = self.raw()
        res["reg_log_skip"] = self.log()
        rnk = self.rnk()
        res["reg_rnk_skip"] = rnk
        res["reg_gum"] = self.gum(rnk)
        return res

    def save(self, res, path):
        y_cols = []
        for k in res.keys():
            if "raw" in k:
                x_col = k
            else:
                y_cols += [k]
        x = np.array(res[x_col])
        for y_col in y_cols:
            y = res[y_col]
            interpolator = interpolate.interp1d(x, y)
            joblib.dump(
                interpolator,
                os.path.join(path, DATA_SUBFOLDER, y_col + "_interpolator.joblib"),
            )


class RegTasksForPrediction(RegTasks):
    def __init__(self, data, params):
        RegTasks.__init__(self, data, params)

    def load(self, path):
        self.interpolators = {}
        for f in os.listdir(os.path.join(path, DATA_SUBFOLDER)):
            if "_interpolator.joblib" not in f:
                continue
            interpolator = joblib.load(os.path.join(path, DATA_SUBFOLDER, f))
            y_col = f.split("_interpolator.joblib")[0]
            self.interpolators[y_col] = interpolator

    def log(self, raw):
        for k in self.interpolators.keys():
            if "log" in k:
                break
        return self.interpolators[k](raw)

    def rnk(self, raw):
        for k in self.interpolators.keys():
            if "rnk" in k:
                break
        return self.interpolators[k](raw)

    def gum(self, raw):
        for k in self.interpolators.keys():
            if "gum" in k:
                break
        return self.interpolators[k](raw)

    def as_dict(self):
        res = OrderedDict()
        raw = self.raw()
        res["reg_raw_skip"] = raw
        res["reg_log_skip"] = self.log(raw)
        res["reg_rnk_skip"] = self.rnk(raw)
        res["reg_gum"] = self.gum(raw)
        return res


class ClfTasks(object):
    def __init__(self, data, params):
        self.values = np.array(data[VALUES_COLUMN])
        self.direction = params["direction"]
        self.thresholds = params["thresholds"]

    def _is_high(self):
        if self.direction == "high":
            return True
        if self.direction == "low":
            return False

    def _binarize(self, cut):
        b = []
        is_high = self._is_high()
        y = []
        for v in self.values:
            if is_high:
                if v >= cut:
                    y += [1]
                else:
                    y += [0]
            else:
                if v <= cut:
                    y += [1]
                else:
                    y += [0]
        return np.array(y, dtype=np.uint8)

    def _has_enough_min_class(self, bin):
        n1 = np.sum(bin)
        n0 = len(bin) - n1
        if n1 < MIN_CLASS or n0 < MIN_CLASS:
            return False
        return True

    def experts(self):
        cuts = []
        keys = sorted(self.thresholds.keys())
        for k in keys:
            v = self.thresholds[k]
            if v is not None:
                cuts += [v]
        return cuts

    def percentiles(self):
        is_high = self._is_high()
        cuts = []
        for p in CLF_PERCENTILES:
            if is_high:
                p = 100 - p
            cuts += [np.percentile(self.values, p)]
        return cuts

    def as_dict(self):
        ecuts = self.experts()
        pcuts = self.percentiles()
        res = OrderedDict()
        do_skip = False
        self._ecuts = {}
        self._pcuts = {}
        self._columns = []
        for i, cut in enumerate(ecuts):
            k = "clf_ex{0}".format(i + 1)
            v = self._binarize(cut)
            if self._has_enough_min_class(v):
                if not do_skip:
                    res[k] = v
                    do_skip = True
                else:
                    k = k + "_skip"
                    res[k] = v
                self._ecuts[k] = float(cut)
                self._columns += [k]
        for p, cut in zip(CLF_PERCENTILES, pcuts):
            k = "clf_p{0}".format(str(p).zfill(2))
            v = self._binarize(cut)
            if self._has_enough_min_class(v):
                if not do_skip:
                    res[k] = v
                    do_skip = True
                else:
                    k = k + "_skip"
                    res[k] = v
                self._pcuts[k] = float(cut)
                self._columns += [k]
        return res

    def save(self, path):
        data = {"columns": self._columns, "ecuts": self._ecuts, "pcuts": self._pcuts}
        with open(os.path.join(path, DATA_SUBFOLDER, "used_cuts.json"), "w") as f:
            json.dump(data, f)


class ClfTasksForPrediction(ClfTasks):
    def __init__(self, data, params):
        ClfTasks.__init__(self, data, params)

    def load(self, path):
        with open(os.path.join(path, DATA_SUBFOLDER, "used_cuts.json"), "r") as f:
            data = json.load(f)
        self._columns = data["columns"]
        self._ecuts = data["ecuts"]
        self._pcuts = data["pcuts"]

    def as_dict(self):
        res = OrderedDict()
        for col in self._columns:
            if col in self._ecuts:
                cut = self._ecuts[col]
            if col in self._pcuts:
                cut = self._pcuts[col]
            v = self._binarize(cut)
            res[col] = v
        return res


class AuxiliaryBinaryTask(object):
    def __init__(self, data):
        self.df = data
        for c in list(self.df.columns):
            if "clf_" in c:
                self.reference = c  # At the moment only one clf is done. TODO
                break
        # TODO

    def get(self):
        # TODO: Work with multitask
        return self.df[self.reference]


class SingleTasks(ZairaBase):
    def __init__(self, path):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        if self.is_predict():
            self.trained_path = self.get_trained_dir()
        else:
            self.trained_path = self.get_output_dir()

    def _get_params(self):
        params = ParametersFile(path=os.path.join(self.trained_path, DATA_SUBFOLDER))
        return params.params

    def _get_data(self):
        df = pd.read_csv(os.path.join(self.path, VALUES_FILENAME))
        return df.drop(columns=[QUALIFIER_COLUMN])

    def _is_simply_binary_classification(self, data):
        if len(set(data[VALUES_COLUMN])) == 2:
            return True
        else:
            return False

    def run(self):
        df = self._get_data()
        if self._is_simply_binary_classification(df):
            df["clf_ex1"] = [int(x) for x in list(df[VALUES_COLUMN])]
        else:
            params = self._get_params()
            reg_tasks = RegTasks(df, params)
            reg = reg_tasks.as_dict()
            reg_tasks.save(reg, self.trained_path)
            for k, v in reg.items():
                df[k] = v
            clf_tasks = ClfTasks(df, params)
            clf = clf_tasks.as_dict()
            clf_tasks.save(self.trained_path)
            for k, v in clf.items():
                df[k] = v
        df = df.drop(columns=[VALUES_COLUMN])
        auxiliary = AuxiliaryBinaryTask(df)
        df[AUXILIARY_TASK_COLUMN] = auxiliary.get()
        df.to_csv(os.path.join(self.path, TASKS_FILENAME), index=False)


class SingleTasksForPrediction(SingleTasks):
    def __init__(self, path):
        SingleTasks.__init__(self, path=path)

    def run(self):
        df = self._get_data()
        if self._is_simply_binary_classification(df):
            df["clf_ex1"] = [int(x) for x in list(df[VALUES_COLUMN])]
        else:
            params = self._get_params()
            reg_tasks = RegTasksForPrediction(df, params)
            reg_tasks.load(self.trained_path)
            reg = reg_tasks.as_dict()
            for k, v in reg.items():
                df[k] = v
            clf_tasks = ClfTasksForPrediction(df, params)
            clf_tasks.load(self.trained_path)
            clf = clf_tasks.as_dict()
            for k, v in clf.items():
                df[k] = v
        df = df.drop(columns=[VALUES_COLUMN])
        df.to_csv(os.path.join(self.path, TASKS_FILENAME), index=False)
