import os
import numpy as np
import pandas as pd

from ... import ZairaBase
from ..base import BaseEstimator, BaseOutcomeAssembler
from ...automl.autogluon import AutoGluonEstimator
from .. import RESULTS_MAPPED_FILENAME, RESULTS_UNMAPPED_FILENAME
from ...vars import DATA_FILENAME, DATA_SUBFOLDER, ESTIMATORS_SUBFOLDER


AUTOGLUON_SAVE_SUBFOLDER = "autogluon"


class Fitter(BaseEstimator):
    def __init__(self, path, estimators_family_subfolder, x_getter):
        BaseEstimator.__init__(self, path=path)
        self.trained_path = os.path.join(
            self.get_output_dir(),
            ESTIMATORS_SUBFOLDER,
            estimators_family_subfolder,
            AUTOGLUON_SAVE_SUBFOLDER,
        )
        self.x_getter = x_getter

    def _get_X(self):
        df = self.x_getter(path=self.path).get()
        return df

    def _get_y(self, task):
        df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
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
        self.logger.debug("Staring AutoGluon estimation")
        estimator = AutoGluonEstimator(
            save_path=self.trained_path, time_budget=time_budget_sec
        )
        self.logger.debug("Fitting")
        estimator.fit(data=df.iloc[train_idxs, :], labels=labels, groups=None)
        estimator = estimator.load()
        results = estimator.run(df)
        self.update_elapsed_time()
        return results


class Predictor(BaseEstimator):
    def __init__(self, path, estimators_family_subfolder, x_getter):
        BaseEstimator.__init__(self, path=path)
        self.trained_path = os.path.join(
            self.get_trained_dir(),
            ESTIMATORS_SUBFOLDER,
            estimators_family_subfolder,
            AUTOGLUON_SAVE_SUBFOLDER,
        )
        self.x_getter = x_getter

    def run(self):
        self.reset_time()
        df = self.x_getter(path=self.path).get()
        model = AutoGluonEstimator(save_path=self.trained_path).load()
        results = model.run(df)
        self.update_elapsed_time()
        return results


class Assembler(BaseOutcomeAssembler):
    def __init__(self, path=None, estimators_family_subfolder=None):
        BaseOutcomeAssembler.__init__(self, path=path)
        self.estimators_family_subfolder = estimators_family_subfolder

    def run(self, df):
        df_c = self._get_compounds()
        df_y = df
        df = pd.concat([df_c, df_y], axis=1)
        df.to_csv(
            os.path.join(
                self.path,
                ESTIMATORS_SUBFOLDER,
                self.estimators_family_subfolder,
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
                self.estimators_family_subfolder,
                RESULTS_MAPPED_FILENAME,
            ),
            index=False,
        )


class AutoGluonUtil(ZairaBase):
    def __init__(self, path=None, estimators_family_subfolder=None, x_getter=None):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        path_ = os.path.join(
            self.path, ESTIMATORS_SUBFOLDER, estimators_family_subfolder
        )
        if not os.path.exists(path_):
            os.makedirs(path_, exist_ok=True)
        if not self.is_predict():
            self.logger.debug("Starting autogluon fitter")
            self.estimator = Fitter(
                path=self.path,
                estimators_family_subfolder=estimators_family_subfolder,
                x_getter=x_getter,
            )
        else:
            self.logger.debug("Starting autogluon predictor")
            self.estimator = Predictor(
                path=self.path,
                estimators_family_subfolder=estimators_family_subfolder,
                x_getter=x_getter,
            )
        self.assembler = Assembler(
            path=self.path, estimators_family_subfolder=estimators_family_subfolder
        )

    def run(self, time_budget_sec=None):
        if time_budget_sec is not None:
            self.time_budget_sec = int(time_budget_sec)
        else:
            self.time_budget_sec = None
        if not self.is_predict():
            self.logger.debug("Mode: fit")
            results = self.estimator.run(time_budget_sec=self.time_budget_sec)
        else:
            self.logger.debug("Mode: predict")
            results = self.estimator.run()
        self.assembler.run(results)
