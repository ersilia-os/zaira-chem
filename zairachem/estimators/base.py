import os
import pandas as pd
import json
import numpy as np

from .. import ZairaBase
from ..setup import (
    INPUT_SCHEMA_FILENAME,
    MAPPING_FILENAME,
    COMPOUND_IDENTIFIER_COLUMN,
    PARAMETERS_FILE,
    SCHEMA_MERGE_FILENAME,
    SMILES_COLUMN,
)
from ..vars import DATA_SUBFOLDER, DATA_FILENAME


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
            os.path.join(self.trained_path, DATA_SUBFOLDER, SCHEMA_MERGE_FILENAME)
        ) as f:
            self.schema = json.load(f)

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


class BaseOutcomeAssembler(ZairaBase):
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
            if np.isnan(j):
                continue
            R[i] = list(df.iloc[int(j)])
        return pd.DataFrame(R, columns=list(df.columns))
