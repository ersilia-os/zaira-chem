import os
import json
import h5py
import pandas as pd
import numpy as np
import collections
import joblib

from .. import ZairaBase
from ..automl.flaml import FlamlClassifier, FlamlRegressor

from ..vars import (
    DESCRIPTORS_SUBFOLDER,
    DATA_SUBFOLDER,
    DATA_FILENAME,
    MODELS_SUBFOLDER,
)
from ..setup import SCHEMA_MERGE_FILENAME, PARAMETERS_FILE
from ..descriptors import GLOBAL_SUPERVISED_FILE_NAME, GLOBAL_UNSUPERVISED_FILE_NAME

from . import Y_HAT_FILE


# TODO Select best between supervised and unsupervised (head-to-head, same number of dimensions)

ESTIMATORS = None


class BaseEstimator(ZairaBase):
    def __init__(self, path):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        self.logger.debug(self.path)
        if self.is_predict():
            self.trained_path = self.get_trained_dir()
        else:
            self.trained_path = self.path
        with open(
            os.path.join(
                self.trained_path,
                DATA_SUBFOLDER,
                SCHEMA_MERGE_FILENAME,
            )
        ) as f:
            self.schema = json.load(f)

    def _get_X_unsupervised(self):
        f = os.path.join(
            self.path, DESCRIPTORS_SUBFOLDER, GLOBAL_UNSUPERVISED_FILE_NAME
        )
        with h5py.File(f, "r") as f:
            X = f["Values"][:]
        return X

    def _get_X_supervised(self):
        f = os.path.join(self.path, DESCRIPTORS_SUBFOLDER, GLOBAL_SUPERVISED_FILE_NAME)
        with h5py.File(f, "r") as f:
            X = f["Values"][:]
        return X

    def _get_clf_tasks(self):
        return [
            t
            for t in self.schema["tasks"]
            if "clf_" in t and "_aux" not in t and "skip" not in t
        ]

    def _get_reg_tasks(self):
        return [
            t
            for t in self.schema["tasks"]
            if "reg_" in t and "_aux" not in t and "skip" not in t
        ]

    def _get_total_time_budget_sec(self):
        with open(os.path.join(self.path, DATA_SUBFOLDER, PARAMETERS_FILE), "r") as f:
            time_budget = json.load(f)["time_budget"]
        return int(time_budget) * 60 + 1

    def _estimate_time_budget(self):
        elapsed_time = self.get_elapsed_time()
        print("Elapsed time: {0}".format(elapsed_time))
        total_time_budget = self._get_total_time_budget_sec()
        print("Total time budget: {0}".format(total_time_budget))
        available_time = total_time_budget - elapsed_time
        # Assuming classification and regression will be done
        available_time = available_time / 2.0
        # Substract retraining and subsequent tasks
        available_time = available_time * 0.8
        available_time = int(available_time) + 1
        print("Available time: {0}".format(available_time))
        return available_time


class Fitter(BaseEstimator):
    def __init__(self, path):
        BaseEstimator.__init__(self, path=path)
        self.trained_path = os.path.join(self.get_output_dir(), MODELS_SUBFOLDER)

    def _get_flds(self):
        # for now only auxiliary folds are used
        col = [f for f in self.schema["folds"] if "_aux" in f][0]
        df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
        return np.array(df[col])

    def _get_y(self, task):
        # for now iterate task by task
        df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
        return np.array(df[task])

    def run(self, time_budget_sec=None):
        self.reset_time()
        if time_budget_sec is None:
            time_budget_sec = self._estimate_time_budget()
        else:
            time_budget_sec = time_budget_sec
        groups = self._get_flds()
        tasks = collections.OrderedDict()
        X = self._get_X_unsupervised()
        print("Shape", X.shape)
        for t in self._get_reg_tasks():
            y = self._get_y(t)
            print("Y", y[:100])
            model = FlamlRegressor()
            model.fit(
                X,
                y,
                time_budget=time_budget_sec,
                estimators=ESTIMATORS,
                groups=groups,
            )
            file_name = os.path.join(self.trained_path, t + ".joblib")
            model.save(file_name)
            tasks[t] = model.results
        X = self._get_X_unsupervised()
        print(X.shape)
        for t in self._get_clf_tasks():
            y = self._get_y(t)
            model = FlamlClassifier()
            model.fit(
                X,
                y,
                time_budget=time_budget_sec,
                estimators=ESTIMATORS,
                groups=groups,
            )
            file_name = os.path.join(self.trained_path, t + ".joblib")
            model.save(file_name)
            tasks[t] = model.results
        self.update_elapsed_time()
        return tasks


class Predictor(BaseEstimator):
    def __init__(self, path):
        BaseEstimator.__init__(self, path=path)
        self.trained_path = os.path.join(self.get_trained_dir(), MODELS_SUBFOLDER)

    def _get_y(self, task):
        # for now iterate task by task
        df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
        columns = set(df.columns)
        print(columns)
        if task in columns:
            return np.array(df[task])
        else:
            print(task, "not found")
            return None

    def run(self, time_budget_sec=None):
        self.reset_time()
        tasks = collections.OrderedDict()
        X = self._get_X_unsupervised()
        for t in self._get_reg_tasks():
            y = self._get_y(t)
            print("THIS IS THE Y", y)
            model = FlamlRegressor()
            file_name = os.path.join(self.trained_path, t + ".joblib")
            model = model.load(file_name)
            tasks[t] = model.run(X, y)
        X = self._get_X_unsupervised()
        for t in self._get_clf_tasks():
            y = self._get_y(t)
            print("THIS IS THE Y", y)
            model = FlamlClassifier()
            file_name = os.path.join(self.trained_path, t + ".joblib")
            model = model.load(file_name)
            tasks[t] = model.run(X, y)
        self.update_elapsed_time()
        return tasks


class Estimator(ZairaBase):
    def __init__(self, path=None):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        if not self.is_predict():
            self.estimator = Fitter(path=self.path)
        else:
            self.estimator = Predictor(path=self.path)

    def run(self, time_budget_sec=None):
        if time_budget_sec is not None:
            self.time_budget_sec = int(time_budget_sec)
        else:
            self.time_budget_sec = None
        results = self.estimator.run(time_budget_sec=self.time_budget_sec)
        joblib.dump(results, os.path.join(self.path, MODELS_SUBFOLDER, Y_HAT_FILE))
