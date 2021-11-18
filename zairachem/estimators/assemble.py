import pandas as pd
import json
import os

from .. import ZairaBase
from ..vars import DATA_SUBFOLDER, MODELS_SUBFOLDER, DATA_FILENAME
from ..setup import (
    INPUT_SCHEMA_FILENAME,
    MAPPING_FILENAME,
    COMPOUND_IDENTIFIER_COLUMN,
    SMILES_COLUMN,
)
from . import Y_HAT_FILE, RESULTS_UNMAPPED_FILENAME, RESULTS_MAPPED_FILENAME


class OutcomeAssembler(ZairaBase):
    def __init__(self, path=None):
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

    def _get_y_hat(self):
        return pd.read_csv(os.path.join(self.path, MODELS_SUBFOLDER, Y_HAT_FILE))

    def _binarize_y_hat(self, y_hat):
        df = y_hat.copy()
        for c in list(y_hat.columns):
            if "clf_" not in c:
                continue
            with open(
                os.path.join(self.trained_path, MODELS_SUBFOLDER, c + ".json"), "r"
            ) as f:
                data = json.load(f)
                threshold = data["threshold"]
            y_raw = list(y_hat[c])
            y_bin = []
            for y in y_raw:
                if y > threshold:
                    y_bin += [1]
                else:
                    y_bin += [0]
            new_c = c + "_bin"
            df[new_c] = y_bin
        return df

    def _get_original_input_size(self):
        with open(
            os.path.join(self.path, DATA_SUBFOLDER, INPUT_SCHEMA_FILENAME), "r"
        ) as f:
            schema = json.load(f)
        file_name = schema["input_file"]
        return pd.read_csv(file_name).shape[0]

    def _projections(self, df):
        return df

    def _remap(self, df, mappings):
        n = self._get_original_input_size()
        ncol = df.shape[1]
        R = [[None] * ncol for _ in range(n)]
        for m in mappings.values:
            i, j = m[0], m[1]
            R[i] = list(df.iloc[j])
        return pd.DataFrame(R, columns=list(df.columns))

    def run(self):
        df_c = self._get_compounds()
        df_y = self._get_y_hat()
        df_y = self._binarize_y_hat(df_y)
        df_y = self._projections(df_y)
        df = pd.concat([df_c, df_y], axis=1)
        df.to_csv(
            os.path.join(self.path, MODELS_SUBFOLDER, RESULTS_UNMAPPED_FILENAME),
            index=False,
        )
        mappings = self._get_mappings()
        df = self._remap(df, mappings)
        df.to_csv(
            os.path.join(self.path, MODELS_SUBFOLDER, RESULTS_MAPPED_FILENAME),
            index=False,
        )
