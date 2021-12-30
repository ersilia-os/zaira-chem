import pandas as pd
import json
import os
import joblib
import collections

from .. import ZairaBase
from ..vars import (
    DATA_SUBFOLDER,
    DESCRIPTORS_SUBFOLDER,
    MODELS_SUBFOLDER,
    DATA_FILENAME,
)
from ..setup import (
    INPUT_SCHEMA_FILENAME,
    MAPPING_FILENAME,
    COMPOUND_IDENTIFIER_COLUMN,
    SMILES_COLUMN,
)
from . import Y_HAT_FILE, RESULTS_UNMAPPED_FILENAME, RESULTS_MAPPED_FILENAME


class BaseOutcomeAssembler(ZairaBase):
    def __init__(self, path=None, model_id=None):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        if self.is_predict():
            self.trained_path = self.get_trained_dir()
        else:
            self.trained_path = self.path

    def _get_mappings(self):
        return pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, MAPPING_FILENAME))

    def _get_compounds(self):
        return pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))[
            [COMPOUND_IDENTIFIER_COLUMN, SMILES_COLUMN]
        ]

    def _get_original_input_size(self):
        with open(
            os.path.join(self.path, DATA_SUBFOLDER, INPUT_SCHEMA_FILENAME), "r"
        ) as f:
            schema = json.load(f)
        file_name = schema["input_file"]
        return pd.read_csv(file_name).shape[0]

    def _remap(self, df, mappings):
        n = self._get_original_input_size()
        ncol = df.shape[1]
        R = [[None] * ncol for _ in range(n)]
        for m in mappings.values:
            i, j = m[0], m[1]
            R[i] = list(df.iloc[j])
        return pd.DataFrame(R, columns=list(df.columns))


class IndividualOutcomeAssembler(BaseOutcomeAssembler):
    def __init__(self, path=None, model_id=None):
        BaseOutcomeAssembler.__init__(self, path=path)
        self.model_id = model_id

    def _get_y_hat(self):
        results = joblib.load(
            os.path.join(self.path, MODELS_SUBFOLDER, self.model_id, Y_HAT_FILE)
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
                self.path, MODELS_SUBFOLDER, self.model_id, RESULTS_UNMAPPED_FILENAME
            ),
            index=False,
        )
        mappings = self._get_mappings()
        df = self._remap(df, mappings)
        df.to_csv(
            os.path.join(
                self.path, MODELS_SUBFOLDER, self.model_id, RESULTS_MAPPED_FILENAME
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
            path = self.get_trained_dir()
        with open(os.path.join(path, DESCRIPTORS_SUBFOLDER, "done_eos.json"), "r") as f:
            model_ids = list(json.load(f))
        return model_ids

    def run(self):
        model_ids = self._get_model_ids()
        for model_id in model_ids:
            o = IndividualOutcomeAssembler(path=self.path, model_id=model_id)
            o.run()
