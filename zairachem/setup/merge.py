import os
import pandas as pd
import json
from ..vars import DATA_SUBFOLDER, DATA_FILENAME
from . import FOLDS_FILENAME, COMPOUNDS_FILENAME, TASKS_FILENAME, SCHEMA_MERGE_FILENAME
from . import COMPOUND_IDENTIFIER_COLUMN, SMILES_COLUMN


class DataMerger(object):
    def __init__(self, path):
        self.path = path

    def run(self):
        df_cpd = pd.read_csv(os.path.join(self.path, COMPOUNDS_FILENAME))[
            [COMPOUND_IDENTIFIER_COLUMN, SMILES_COLUMN]
        ]
        df_fld = pd.read_csv(os.path.join(self.path, FOLDS_FILENAME))
        df_tsk = pd.read_csv(os.path.join(self.path, TASKS_FILENAME))
        df_tsk = df_tsk[[c for c in list(df_tsk.columns) if "reg_" in c or "clf_" in c or c == COMPOUND_IDENTIFIER_COLUMN]]
        df = pd.concat([df_cpd, df_fld], axis=1)
        df = df.merge(df_tsk, on="compound_id")
        schema = {
            "compounds": list(df_cpd.columns),
            "folds": list(df_fld.columns),
            "tasks": [c for c in list(df_tsk.columns) if c != COMPOUND_IDENTIFIER_COLUMN],
        }
        df.to_csv(os.path.join(self.path, DATA_FILENAME), index=False)
        with open(os.path.join(self.path, SCHEMA_MERGE_FILENAME), "w") as f:
            json.dump(schema, f, indent=4)


class DataMergerForPrediction(object):
    def __init__(self, path):
        self.path = path

    def run(self):
        # TODO: This is a placeholder for more elaborate predict-time scenearios (e.g. validation data available)
        df_cpd = pd.read_csv(os.path.join(self.path, COMPOUNDS_FILENAME))[[COMPOUND_IDENTIFIER_COLUMN, SMILES_COLUMN]]
        schema = {
            "compounds": list(df_cpd.columns)
        }
        with open(os.path.join(self.path, SCHEMA_MERGE_FILENAME), "w") as f:
            json.dump(schema, f, indent=4)
        df = df_cpd
        df.to_csv(os.path.join(self.path, DATA_FILENAME), index=False)
