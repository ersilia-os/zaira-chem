import os
import numpy as np
import pandas as pd
import json
import h5py

from zairachem.estimators.assemble import BaseOutcomeAssembler

from .. import ZairaBase
from ..estimators.estimate import BaseEstimator
from ..automl.autogluon import AutoGluonEstimator
from ..estimators import RESULTS_MAPPED_FILENAME, RESULTS_UNMAPPED_FILENAME
from ..setup import COMPOUND_IDENTIFIER_COLUMN, SMILES_COLUMN
from ..vars import (
    DATA_FILENAME,
    DATA_SUBFOLDER,
    DESCRIPTORS_SUBFOLDER,
    MODELS_SUBFOLDER,
    POOL_SUBFOLDER,
)


AUTOGLUON_SAVE_SUBFOLDER = "autogluon"


class XGetter(ZairaBase):
    def __init__(self, path):
        ZairaBase.__init__(self)
        self.path = path
        self.X = []
        self.columns = []

    def _get_folds(self):
        df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
        if "fld_aux" in list(df.columns):
            self.columns += ["fld_aux"]
            self.X += [np.array(df[["fld_aux"]])]

    def _get_manifolds(self):
        with h5py.File(
            os.path.join(self.path, DESCRIPTORS_SUBFOLDER, "pca.h5"), "r"
        ) as f:
            X_ = f["Values"][:]
            self.X += [X_]
            for i in range(X_.shape[1]):
                self.columns += ["pca-{0}".format(i)]
        with h5py.File(
            os.path.join(self.path, DESCRIPTORS_SUBFOLDER, "umap.h5"), "r"
        ) as f:
            X_ = f["Values"][:]
            self.X += [X_]
            for i in range(X_.shape[1]):
                self.columns += ["umap-{0}".format(i)]

    def _get_out_of_sample_predictions(self):
        with open(
            os.path.join(self.path, DESCRIPTORS_SUBFOLDER, "done_eos.json"), "r"
        ) as f:
            model_ids = list(json.load(f))
        for model_id in model_ids:
            file_name = os.path.join(
                self.path, MODELS_SUBFOLDER, model_id, RESULTS_UNMAPPED_FILENAME
            )
            df = pd.read_csv(file_name)
            df = df[
                [
                    c
                    for c in list(df.columns)
                    if c not in [SMILES_COLUMN, COMPOUND_IDENTIFIER_COLUMN]
                ]
            ]
            self.X += [np.array(df)]
            self.columns += ["{0}-{1}".format(c, model_id) for c in list(df.columns)]

    def get(self):
        if not self.is_predict():
            self._get_folds()
        self._get_manifolds()
        self._get_out_of_sample_predictions()
        X = np.hstack(self.X)
        df = pd.DataFrame(X, columns=self.columns)
        df.to_csv(os.path.join(self.path, POOL_SUBFOLDER, "data.csv"), index=False)
        return df


class Fitter(BaseEstimator):
    def __init__(self, path):
        BaseEstimator.__init__(self, path=path)
        self.trained_path = os.path.join(
            self.get_output_dir(), POOL_SUBFOLDER, AUTOGLUON_SAVE_SUBFOLDER
        )

    def _get_X(self):
        df = XGetter(path=self.path).get()
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
        df_X = self._get_X()
        df_Y = self._get_Y()
        df = pd.concat([df_X, df_Y], axis=1)
        labels = list(df_Y.columns)
        estimator = AutoGluonEstimator(
            save_path=self.trained_path, time_budget=time_budget_sec
        )
        if "fld_aux" in list(df_X.columns):
            groups = "fld_aux"
        else:
            groups = None
        results = estimator.fit(data=df, labels=labels, groups=groups)
        self.update_elapsed_time()
        return results


class Predictor(BaseEstimator):
    def __init__(self, path):
        BaseEstimator.__init__(self, path=path)
        self.trained_path = os.path.join(
            self.get_trained_dir(), POOL_SUBFOLDER, AUTOGLUON_SAVE_SUBFOLDER
        )

    def run(self):
        self.reset_time()
        df = XGetter(path=self.path).get()
        model = AutoGluonEstimator(save_path=self.trained_path).load()
        results = model.run(df)
        self.update_elapsed_time()
        return results


class PoolAssembler(BaseOutcomeAssembler):
    def __init__(self, path=None):
        BaseOutcomeAssembler.__init__(self, path=path)

    def run(self, df):
        df_c = self._get_compounds()
        df_y = df
        df = pd.concat([df_c, df_y], axis=1)
        df.to_csv(
            os.path.join(self.path, POOL_SUBFOLDER, RESULTS_UNMAPPED_FILENAME),
            index=False,
        )
        mappings = self._get_mappings()
        df = self._remap(df, mappings)
        df.to_csv(
            os.path.join(self.path, POOL_SUBFOLDER, RESULTS_MAPPED_FILENAME),
            index=False,
        )


class Pooler(ZairaBase):
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
        if not self.is_predict():
            results = self.estimator.run(time_budget_sec=self.time_budget_sec)
        else:
            results = self.estimator.run()
        pa = PoolAssembler(path=self.path)
        pa.run(results)
