import os
import pandas as pd
import joblib

from .basicprops import BasicProperties
from .similarity import TanimotoSimilarity
from .oneclass import OneClassClassifier

from ..utils.pipeline import PipelineStep
from .. import ZairaBase

from . import (
    BASIC_PROPERTIES_FILENAME,
    TANIMOTO_SIMILARITY_NEAREST_NEIGHBORS_FILENAME,
    ONECLASS_FILENAME,
)
from ..vars import DATA_SUBFOLDER, DATA_FILENAME, APPLICABILITY_SUBFOLDER
from ..setup import SMILES_COLUMN


class ApplicabilityEvaluator(ZairaBase):
    def __init__(self, path):
        self._orig_path = path
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        self.df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
        self._is_predict = self.is_predict()
        self._smiles = self._get_smiles()
        if self._is_predict:
            self.trained_path = self.get_trained_dir()
        else:
            self.trained_path = self.path

    def _get_smiles(self):
        return list(self.df[SMILES_COLUMN])

    def _basic_properties(self):
        step = PipelineStep("applicability_basic_properties", self.path)
        if not step.is_done():
            db = BasicProperties().calculate(self._smiles)
            db.to_csv(
                os.path.join(
                    self.path, APPLICABILITY_SUBFOLDER, BASIC_PROPERTIES_FILENAME
                ),
                index=False,
            )
            step.update()

    def _tanimoto_similarity(self):
        step = PipelineStep("applicability_tanimoto_similarity", self.path)
        if not step.is_done():
            filename = os.path.join(
                self.trained_path, APPLICABILITY_SUBFOLDER, "tanimoto_similarity.joblib"
            )
            if self._is_predict:
                ts = joblib.load(filename)
                ds = ts.kneighbors(self._smiles)
            else:
                ts = TanimotoSimilarity()
                ts.fit(self._smiles)
                joblib.dump(ts, filename)
                ds = ts.kneighbors(self._smiles)
            ds.to_csv(
                os.path.join(
                    self.path,
                    APPLICABILITY_SUBFOLDER,
                    TANIMOTO_SIMILARITY_NEAREST_NEIGHBORS_FILENAME,
                ),
                index=False,
            )
            step.update()

    def _oneclass(self):
        step = PipelineStep("applicability_oneclass", self.path)
        if not step.is_done():
            filename = os.path.join(
                self.trained_path, APPLICABILITY_SUBFOLDER, "oneclass.joblib"
            )
            if self._is_predict:
                oc = joblib.load(filename)
                df = oc.predict(self._smiles)
            else:
                oc = OneClassClassifier()
                oc.fit(self._smiles)
                joblib.dump(oc, filename)
                df = oc.predict(self._smiles)
            df.to_csv(
                os.path.join(self.path, APPLICABILITY_SUBFOLDER, ONECLASS_FILENAME),
                index=False,
            )
            step.update()

    def run(self):
        self._basic_properties()
        self._tanimoto_similarity()
        #  self._oneclass() # TODO Applicability one class classifier
