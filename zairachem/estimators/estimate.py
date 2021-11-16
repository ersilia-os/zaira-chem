import os
import json
import h5py
import pandas as pd
import numpy as np
import collections

from .. import ZairaBase
from ..automl.flaml import FlamlClassifier, FlamlRegressor

from ..vars import DESCRIPTORS_SUBFOLDER, DATA_SUBFOLDER, DATA_FILENAME
from ..setup import SCHEMA_MERGE_FILENAME
from ..descriptors import GLOBAL_SUPERVISED_FILE_NAME


ESTIMATORS = ["lrl1"]


class BaseEstimator(ZairaBase):
    def __init__(self, path):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        self.logger.debug(self.path)
        with open(
            os.path.join(
                self.path,
                DATA_SUBFOLDER,
                SCHEMA_MERGE_FILENAME,
            )
        ) as f:
            self.schema = json.load(f)

    def _get_X(self):
        # for now only use supervised processing
        f = os.path.join(self.path, DESCRIPTORS_SUBFOLDER, GLOBAL_SUPERVISED_FILE_NAME)
        with h5py.File(f, "r") as f:
            X = f["Values"][:]
        return X

    def _get_clf_tasks(self):
        return [t for t in self.schema["tasks"] if "clf_" in t and "_aux" not in t]

    def _get_reg_tasks(self):
        return [t for t in self.schema["tasks"] if "reg_" in t and "_aux" not in t]


class Fitter(BaseEstimator):
    def __init__(self, path):
        BaseEstimator.__init__(self, path=path)

    def _get_flds(self):
        # for now only auxiliary folds are used
        col = [f for f in self.schema["folds"] if "_aux" in f][0]
        df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
        return np.array(df[col])

    def _get_y(self, task):
        # for now iterate task by task
        df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
        return np.array(df[task])

    def run(self):
        groups = self._get_flds()
        X = self._get_X()
        tasks = collections.OrderedDict()
        for t in self._get_clf_tasks():
            y = self._get_y(t)
            model = FlamlClassifier()
            model.fit(X, y, estimators=ESTIMATORS, groups=groups)
            model.save()
            tasks[t] = model.predict(X)
        for t in self._get_reg_tasks():
            y = self._get_y(t)
            model = FlamlRegressor()
            model.fit(X, y)
            model.save()
            tasks[t] = model.predict(X)


class Predictor(BaseEstimator):
    def __init__(self, path):
        BaseEstimator.__init__(self, path=path)

    def run(self):
        X = self._get_X()
        for t in self._get_clf_tasks():
            model = FlamlClassifier()
            model = model.load(XXX)
            model.predict(X)
        for t in self._get_reg_tasks():
            model = FlamlRegressor()
            model = model.load(XXX)
            model.predict()


class Estimator(ZairaBase):
    def __init__(self, path=None):
        ZairaBase.__init__(self)
        if not self.is_predict():
            self.estimator = Fitter()
        else:
            self.estimator = Predictor()

    def run(self):
        results = self.estimator.run()
        df = pd.DataFrame(results)
        df.to_csv(
            os.path.join(
                self.path,
                MODELS_SUBFOLDER,
            )
        )
