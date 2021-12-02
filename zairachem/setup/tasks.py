import os
import numpy as np
import pandas as pd
from collections import OrderedDict
from scipy.stats import rankdata, gumbel_r
from scipy import interpolate
import joblib
import json

from . import (
    COMPOUNDS_FILENAME,
    COMPOUND_IDENTIFIER_COLUMN,
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


class RegTasks(object):
    def __init__(self, data, params, path):
        compounds = pd.read_csv(os.path.join(path, DATA_SUBFOLDER, COMPOUNDS_FILENAME))
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

    def raw(self):
        if self._raw is None:
            min_cred = self.range["min"]
            max_cred = self.range["max"]
            if min_cred is None and max_cred is None:
                raw = self.values
            else:
                raw = np.clip(self.values, min_cred, max_cred)
            self._raw = self.smoothen(raw)
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
        res["reg_raw_skip"] = self.raw()
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
        raw = self.raw()
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
    def __init__(self, data, params, path):
        ClfTasks.__init__(self, data, params, path)

    def load(self, path):
        json_file = os.path.join(path, DATA_SUBFOLDER, "used_cuts.json")
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
            reg_tasks = RegTasks(df, params, path=self.trained_path)
            reg = reg_tasks.as_dict()
            for k, v in reg.items():
                df[k] = v
            clf_tasks = ClfTasks(df, params, path=self.trained_path)
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
        df.to_csv(os.path.join(self.path, TASKS_FILENAME), index=False)
