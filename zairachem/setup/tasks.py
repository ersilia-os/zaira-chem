import os
import numpy as np
import pandas as pd
import collections
from collections import OrderedDict
import joblib
import json

from . import (
    COMPOUNDS_FILENAME,
    COMPOUND_IDENTIFIER_COLUMN,
    PARAMETERS_FILE,
    SMILES_COLUMN,
    VALUES_FILENAME,
    VALUES_COLUMN,
    QUALIFIER_COLUMN,
    TASKS_FILENAME,
    AUXILIARY_TASK_COLUMN,
)
from .files import ParametersFile
from ..vars import CLF_PERCENTILES, MIN_CLASS, DATA_SUBFOLDER
from .. import ZairaBase
from .utils import SmoothenY

from sklearn.preprocessing import PowerTransformer, QuantileTransformer


USED_CUTS_FILE = "used_cuts.json"


class ExpectedTaskType(ZairaBase):
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

    def get(self):
        params = self._get_params()
        return params["task"]


class RegTasks(object):
    def __init__(self, data, params, path):
        file_name = os.path.join(path, DATA_SUBFOLDER, COMPOUNDS_FILENAME)
        if not os.path.exists(file_name):
            file_name = os.path.join(path, COMPOUNDS_FILENAME)
        compounds = pd.read_csv(file_name)
        cid2smiles = {}
        for r in compounds[[COMPOUND_IDENTIFIER_COLUMN, SMILES_COLUMN]].values:
            cid2smiles[r[0]] = r[1]
        self.smiles_list = []
        for cid in list(data[COMPOUND_IDENTIFIER_COLUMN]):
            self.smiles_list += [cid2smiles[cid]]
        self.values = np.array(data[VALUES_COLUMN])
        self.direction = params["direction"]
        self.range = params["credibility_range"]
        self.path = path
        self._raw = None

    def smoothen(self, raw):
        return SmoothenY(self.smiles_list, raw).run()

    def raw(self, smoothen=None):
        if self._raw is None:
            min_cred = self.range["min"]
            max_cred = self.range["max"]
            if min_cred is None and max_cred is None:
                raw = self.values
            else:
                raw = np.clip(self.values, min_cred, max_cred)
            if smoothen:
                self._raw = self.smoothen(raw)
            else:
                self._raw = raw
        return self._raw

    def pwr(self):
        raw = self.raw().reshape(-1, 1)
        tr = PowerTransformer(method="yeo-johnson")
        tr.fit(raw)
        joblib.dump(
            tr, os.path.join(self.path, DATA_SUBFOLDER, "pwr_transformer.joblib")
        )
        return tr.transform(raw).ravel()

    def rnk(self):
        raw = self.raw().reshape(-1, 1)
        tr = QuantileTransformer(output_distribution="uniform")
        tr.fit(raw)
        joblib.dump(
            tr, os.path.join(self.path, DATA_SUBFOLDER, "rnk_transformer.joblib")
        )
        return tr.transform(raw).ravel()

    def qnt(self):
        raw = self.raw().reshape(-1, 1)
        tr = QuantileTransformer(output_distribution="normal")
        tr.fit(raw)
        joblib.dump(
            tr, os.path.join(self.path, DATA_SUBFOLDER, "qnt_transformer.joblib")
        )
        return tr.transform(raw).ravel()

    def as_dict(self):
        res = OrderedDict()
        res["reg_raw_skip"] = self.raw(smoothen=True)
        res["reg_pwr_skip"] = self.pwr()
        res["reg_rnk_skip"] = self.rnk()
        res["reg_qnt"] = self.qnt()
        return res


class RegTasksForPrediction(RegTasks):
    def __init__(self, data, params, path):
        RegTasks.__init__(self, data, params, path)

    def load(self, path):
        self._load_path = path

    def pwr(self, raw):
        tr = joblib.load(
            os.path.join(self._load_path, DATA_SUBFOLDER, "pwr_transformer.joblib")
        )
        return tr.transform(raw.reshape(-1, 1)).ravel()

    def rnk(self, raw):
        tr = joblib.load(
            os.path.join(self._load_path, DATA_SUBFOLDER, "rnk_transformer.joblib")
        )
        return tr.transform(raw.reshape(-1, 1)).ravel()

    def qnt(self, raw):
        tr = joblib.load(
            os.path.join(self._load_path, DATA_SUBFOLDER, "qnt_transformer.joblib")
        )
        return tr.transform(raw.reshape(-1, 1)).ravel()

    def as_dict(self):
        res = OrderedDict()
        raw = self.raw(smoothen=False)
        res["reg_raw_skip"] = raw
        res["reg_pwr_skip"] = self.pwr(raw)
        res["reg_rnk_skip"] = self.rnk(raw)
        res["reg_qnt"] = self.qnt(raw)
        return res


class ClfTasks(object):
    def __init__(self, data, params, path):
        self.values = np.array(data[VALUES_COLUMN])
        self.direction = params["direction"]
        self.thresholds = params["thresholds"]
        self.path = path

    def _is_high(self):
        if self.direction == "high":
            return True
        if self.direction == "low":
            return False

    def _binarize(self, cut):
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
        with open(os.path.join(path, DATA_SUBFOLDER, USED_CUTS_FILE), "w") as f:
            json.dump(data, f, indent=4)


class ClfTasksForPrediction(ClfTasks):
    def __init__(self, data, params, path):
        ClfTasks.__init__(self, data, params, path)

    def load(self, path):
        json_file = os.path.join(path, DATA_SUBFOLDER, USED_CUTS_FILE)
        with open(json_file, "r") as f:
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


def task_skipper(df, task):
    columns = list(df.columns)
    new_columns = []
    if task == "regression":
        for c in columns:
            if c.startswith("clf"):
                if "_skip" not in c and "_aux" not in c:
                    c = c + "_skip"
            new_columns += [c]
    if task == "classification":
        for c in columns:
            if c.startswith("reg"):
                if "_skip" not in c and "_aux" not in c:
                    c = c + "_skip"
            new_columns += [c]
    coldict = {}
    for o, n in zip(columns, new_columns):
        coldict[o] = n
    df = df.rename(columns=coldict)
    return df


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
        self._task = ExpectedTaskType(path=path).get()

    def _get_params(self):
        params = ParametersFile(path=os.path.join(self.trained_path, DATA_SUBFOLDER))
        return params.params

    def _get_data(self):
        df = pd.read_csv(os.path.join(self.path, VALUES_FILENAME))
        columns = list(df.columns)
        if QUALIFIER_COLUMN in columns:
            return df.drop(columns=[QUALIFIER_COLUMN])
        else:
            return df

    def _rewrite_data(self, df):
        df.to_csv(os.path.join(self.path, VALUES_FILENAME), sep=",", index=False)

    def _is_simply_binary_classification(self, data):
        unique_values = set(data[VALUES_COLUMN])
        if len(unique_values) == 2:
            if unique_values == {0, 1}:
                self.logger.debug(
                    "It is a binary classification, and values are already expressed as 0 and 1."
                )
            else:
                self.logger.debug("This looks like a binary classification")
                unique_values_count = collections.defaultdict(int)
                for v in list(data[VALUES_COLUMN]):
                    unique_values_count[v] += 1
                unique_values_count = sorted(
                    unique_values_count.items(), key=lambda x: -x[1]
                )
                val_0 = unique_values_count[0][0]  # majority class
                val_1 = unique_values_count[0][1]  # minority class
                self.logger.debug("0: {0}, 1: {1}".format(val_0, val_1))
                data.loc[data[VALUES_COLUMN] == val_0, [VALUES_COLUMN]] = 0
                data.loc[data[VALUES_COLUMN] == val_1, [VALUES_COLUMN]] = 1
                self._rewrite_data(data)
            self.logger.debug("It is simply classification")
            self._force_classification_task()
            return True
        if len(unique_values) == 3:
            if unique_values == {0, 0.5, 1}:
                self.logger.debug(
                    "This looks like a binary classification where there is a third 0.5 value that corresponds to unknowns"
                )
                data.loc[data[VALUES_COLUMN] == 0.5, [VALUES_COLUMN]] = np.nan
                data = data[data[VALUES_COLUMN].notnull()]
                self._rewrite_data(data)
                self._force_classification_task()
                return True
        else:
            self.logger.debug("There is continuous data")
            return False
    
    def _force_classification_task(self):
        params = self._get_params()
        params["task"] = "classification"
        with open(os.path.join(self.trained_path, DATA_SUBFOLDER, PARAMETERS_FILE), "w") as f:
            json.dump(params, f, indent=4)
        self.task = "classification"

    def run(self):
        df = self._get_data()
        if self._is_simply_binary_classification(df):
            self.logger.debug("It is simply a binary classification")
            if self.task is not None:
                assert self.task == "classification"
            df = self._get_data()
            df["clf_ex1"] = [int(x) for x in list(df[VALUES_COLUMN])]
        else:
            self.logger.debug("Data is not simply a binary")
            df = self._get_data()
            params = self._get_params()
            reg_tasks = RegTasks(df, params, path=self.trained_path)
            reg = reg_tasks.as_dict()
            for k, v in reg.items():
                self.logger.debug("Setting {0}".format(k))
                df[k] = v
            clf_tasks = ClfTasks(df, params, path=self.trained_path)
            clf = clf_tasks.as_dict()
            clf_tasks.save(self.trained_path)
            for k, v in clf.items():
                self.logger.debug("Setting {0}".format(k))
                df[k] = v
        df = df.drop(columns=[VALUES_COLUMN])
        auxiliary = AuxiliaryBinaryTask(df)
        df[AUXILIARY_TASK_COLUMN] = auxiliary.get()
        df = task_skipper(df, self._task)
        df.to_csv(os.path.join(self.path, TASKS_FILENAME), index=False)


class SingleTasksForPrediction(SingleTasks):
    def __init__(self, path):
        SingleTasks.__init__(self, path=path)

    def run(self):
        df = self._get_data()
        if self._is_simply_binary_classification(df):
            self.logger.debug("It is simply a binary classification")
            df = self._get_data()
            if self.task is not None:
                assert self.task == "classification"
            df["clf_ex1"] = [int(x) for x in list(df[VALUES_COLUMN])]
        else:
            self.logger.debug("Data is not simply a binary classification")
            df = self._get_data()
            params = self._get_params()
            reg_tasks = RegTasksForPrediction(df, params, self.path)
            reg_tasks.load(self.trained_path)
            reg = reg_tasks.as_dict()
            for k, v in reg.items():
                df[k] = v
            clf_tasks = ClfTasksForPrediction(df, params, self.path)
            clf_tasks.load(self.trained_path)
            clf = clf_tasks.as_dict()
            for k, v in clf.items():
                df[k] = v
        df = df.drop(columns=[VALUES_COLUMN])
        df = task_skipper(df, self._task)
        df.to_csv(os.path.join(self.path, TASKS_FILENAME), index=False)
