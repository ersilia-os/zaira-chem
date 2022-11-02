import os
import numpy as np
import pandas as pd

from ... import ZairaBase
from ..base import BaseEstimator, BaseOutcomeAssembler
from ...automl.molmap import MolMapEstimator

from . import ESTIMATORS_FAMILY_SUBFOLDER
from .. import RESULTS_UNMAPPED_FILENAME, RESULTS_MAPPED_FILENAME
from ...vars import (
    DATA_FILENAME,
    DATA_SUBFOLDER,
    ESTIMATORS_SUBFOLDER,
    DATA_AUGMENTED_FILENAME,
)
from ...setup import SMILES_COLUMN


_USE_AUGMENTED = True


class Fitter(BaseEstimator):
    def __init__(self, path):
        BaseEstimator.__init__(self, path=path)
        self.trained_path = os.path.join(
            self.get_output_dir(), ESTIMATORS_SUBFOLDER, ESTIMATORS_FAMILY_SUBFOLDER
        )
        if _USE_AUGMENTED:
            self._data_filename = DATA_AUGMENTED_FILENAME
        else:
            self._data_filename = DATA_FILENAME

    def _get_X(self):
        df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, self._data_filename))
        return df[[SMILES_COLUMN]]

    def _get_y(self, task):
        df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, self._data_filename))
        return np.array(df[task])

    def _get_Y(self):
        Y = []
        columns = []
        for t in self._get_reg_tasks():
            y = self._get_y(t)
            Y += [y]
            columns += [t]
        for t in self._get_clf_tasks():
            y = self._get_y(t)
            Y += [y]
            columns += [t]
        Y = np.array(Y).T
        df = pd.DataFrame(Y, columns=columns)
        return df

    def _get_X_forward(self):
        df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
        return df[[SMILES_COLUMN]]

    def _get_y_forward(self, task):
        df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
        return np.array(df[task])

    def _get_Y_forward(self):
        Y = []
        columns = []
        for t in self._get_reg_tasks():
            y = self._get_y_forward(t)
            Y += [y]
            columns += [t]
        for t in self._get_clf_tasks():
            y = self._get_y_forward(t)
            Y += [y]
            columns += [t]
        Y = np.array(Y).T
        df = pd.DataFrame(Y, columns=columns)
        return df

    def run(self, time_budget_sec=None):
        self.reset_time()
        if time_budget_sec is None:
            time_budget_sec = self._estimate_time_budget()
        else:
            time_budget_sec = time_budget_sec
        train_idxs = self.get_train_indices(path=self.path)
        df_X = self._get_X()
        df_Y = self._get_Y()
        df = pd.concat([df_X, df_Y], axis=1)
        labels = list(df_Y.columns)
        self.logger.debug("Starting BIDD MolMap estimation")
        estimator = MolMapEstimator(save_path=self.trained_path)
        self.logger.debug("Fitting")
        estimator.fit(data=df.iloc[train_idxs, :], labels=labels)
        estimator.save()
        estimator = estimator.load()
        df_X = self._get_X_forward()
        df_Y = self._get_Y_forward()
        df = pd.concat([df_X, df_Y], axis=1)
        results = estimator.run(df)
        self.update_elapsed_time()
        return results


class Predictor(BaseEstimator):
    def __init__(self, path):
        BaseEstimator.__init__(self, path=path)
        self.trained_path = os.path.join(
            self.get_trained_dir(), ESTIMATORS_SUBFOLDER, ESTIMATORS_FAMILY_SUBFOLDER
        )

    def run(self):
        self.reset_time()
        df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))[
            [SMILES_COLUMN]
        ]
        model = MolMapEstimator(save_path=self.trained_path).load()
        results = model.run(df)
        self.update_elapsed_time()
        return results


class Assembler(BaseOutcomeAssembler):
    def __init__(self, path=None):
        BaseOutcomeAssembler.__init__(self, path=path)

    def run(self, df):
        df_c = self._get_compounds()
        df_y = df
        df = pd.concat([df_c, df_y], axis=1)
        df.to_csv(
            os.path.join(
                self.path,
                ESTIMATORS_SUBFOLDER,
                ESTIMATORS_FAMILY_SUBFOLDER,
                RESULTS_UNMAPPED_FILENAME,
            ),
            index=False,
        )
        mappings = self._get_mappings()
        df = self._remap(df, mappings)
        df.to_csv(
            os.path.join(
                self.path,
                ESTIMATORS_SUBFOLDER,
                ESTIMATORS_FAMILY_SUBFOLDER,
                RESULTS_MAPPED_FILENAME,
            ),
            index=False,
        )


class Estimator(ZairaBase):
    def __init__(self, path=None):
        ZairaBase.__init__(self)
        self.logger.debug(path)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        path_ = os.path.join(
            self.path, ESTIMATORS_SUBFOLDER, ESTIMATORS_FAMILY_SUBFOLDER
        )
        if not os.path.exists(path_):
            os.makedirs(path_, exist_ok=True)
        if not self.is_predict():
            self.logger.debug("Starting molmap fitter")
            self.estimator = Fitter(path=self.path)
        else:
            self.logger.debug("Starting molmap predictor")
            self.estimator = Predictor(path=self.path)
        self.assembler = Assembler(path=self.path)

    def run(self, time_budget_sec=None):
        if time_budget_sec is not None:
            self.time_budget_sec = int(time_budget_sec)
        else:
            self.time_budget_sec = None
        if not self.is_predict():
            self.logger.debug("Mode: fit")
            results = self.estimator.run()
        else:
            self.logger.debug("Mode: predict")
            results = self.estimator.run()
        self.assembler.run(results)
