import pandas as pd
import json
import os
import joblib
import collections

from . import ESTIMATORS_FAMILY_SUBFOLDER
from ... import ZairaBase
from ...vars import DESCRIPTORS_SUBFOLDER, ESTIMATORS_SUBFOLDER
from .. import Y_HAT_FILE, RESULTS_UNMAPPED_FILENAME, RESULTS_MAPPED_FILENAME
from ..base import BaseOutcomeAssembler


class IndividualOutcomeAssembler(BaseOutcomeAssembler):
    def __init__(self, path=None, model_id=None):
        BaseOutcomeAssembler.__init__(self, path=path)
        self.model_id = model_id

    def _get_y_hat(self):
        results = joblib.load(
            os.path.join(
                self.path,
                ESTIMATORS_SUBFOLDER,
                ESTIMATORS_FAMILY_SUBFOLDER,
                self.model_id,
                Y_HAT_FILE,
            )
        )
        data = collections.OrderedDict()
        for c, r in results.items():
            r = r["main"]
            data[c] = r["y_hat"]
            if "b_hat" in r:
                data[c + "_bin"] = r["b_hat"]
        return pd.DataFrame(data)

    def run(self):
        df_c = self._get_compounds()
        df_y = self._get_y_hat()
        df = pd.concat([df_c, df_y], axis=1)
        df.to_csv(
            os.path.join(
                self.path,
                ESTIMATORS_SUBFOLDER,
                ESTIMATORS_FAMILY_SUBFOLDER,
                self.model_id,
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
                self.model_id,
                RESULTS_MAPPED_FILENAME,
            ),
            index=False,
        )


class OutcomeAssembler(ZairaBase):
    def __init__(self, path=None):
        ZairaBase.__init__(self)
        self.path = path

    def _get_model_ids(self):
        if self.path is None:
            path = self.get_output_dir()
        else:
            path = self.path
        if self.is_predict():
            path_trained = self.get_trained_dir()
        else:
            path_trained = path
        with open(
            os.path.join(path_trained, DESCRIPTORS_SUBFOLDER, "done_eos.json"), "r"
        ) as f:
            model_ids = list(json.load(f))
        model_ids_successful = []
        for model_id in model_ids:
            if os.path.isfile(
                os.path.join(
                    path,
                    ESTIMATORS_SUBFOLDER,
                    ESTIMATORS_FAMILY_SUBFOLDER,
                    model_id,
                    "y_hat.joblib",
                )
            ):
                model_ids_successful += [model_id]
        return model_ids_successful

    def run(self):
        model_ids = self._get_model_ids()
        for model_id in model_ids:
            o = IndividualOutcomeAssembler(path=self.path, model_id=model_id)
            o.run()
