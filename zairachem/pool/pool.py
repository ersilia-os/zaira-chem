import os
from re import I
import numpy as np
import pandas as pd
import joblib
import importlib

from .. import ZairaBase
from ..estimators.base import BaseOutcomeAssembler
from ..estimators import RESULTS_MAPPED_FILENAME, RESULTS_UNMAPPED_FILENAME
from ..vars import DATA_SUBFOLDER, POOL_SUBFOLDER, ENSEMBLE_MODE

if ENSEMBLE_MODE == "bagging":
    pooler = importlib.import_module("zairachem.pool.bagger")
elif ENSEMBLE_MODE == "blending":
    pooler = importlib.import_module("zairachem.pool.blender")
else:
    pooler = None

Fitter = pooler.Fitter
Predictor = pooler.Predictor


class PoolAssembler(BaseOutcomeAssembler):
    def __init__(self, path=None):
        BaseOutcomeAssembler.__init__(self, path=path)

    def _back_to_raw(self, df):
        for c in list(df.columns):
            if "reg_" in c:
                transformer = joblib.load(
                    os.path.join(
                        self.trained_path,
                        DATA_SUBFOLDER,
                        "{0}_transformer.joblib".format(c.split("_")[1]),
                    )
                )
                trn = np.array(df[c]).reshape(-1, 1)
                raw = transformer.inverse_transform(trn)[:, 0]
                df["reg_raw"] = raw
        return df

    def run(self, df):
        df = self._back_to_raw(df)
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
            self.logger.debug("Starting pooled fitter")
            self.estimator = Fitter(path=self.path)
        else:
            self.logger.debug("Starting pooled predictor")
            self.estimator = Predictor(path=self.path)

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
        pa = PoolAssembler(path=self.path)
        pa.run(results)
