import os
import tempfile
import numpy as np
import json
import pandas as pd

from .utils.conda import SimpleConda

root = os.path.dirname(os.path.abspath(__file__))

MOLMAP_CONDA_ENVIRONMENT = "molmap"
COLUMNS_MAPPING_FILENAME = "columns.json"

SMILES_COLUMN = "smiles"


class MolMapModel(object):
    def __init__(self, save_path):
        self.tmp_folder = tempfile.mkdtemp(prefix="ersilia-")
        self.save_path = os.path.abspath(save_path)

    def fit(self, data):
        with open(os.path.join(self.save_path, COLUMNS_MAPPING_FILENAME), "w") as f:
            reg_columns = [
                c
                for c in list(data.columns)
                if "reg_" in c and "_skip" not in c and "_aux" not in c
            ]
            clf_columns = [
                c
                for c in list(data.columns)
                if "clf_" in c and "_skip" not in c and "_aux" not in c
            ]
            json.dump(
                {"reg": reg_columns, "clf": clf_columns, "all": list(data.columns)},
                f,
                indent=4,
            )
        data_file = os.path.join(self.tmp_folder, "data.csv")
        data.to_csv(data_file, index=False)
        script_path = os.path.join(root, "scripts", "fit.py")
        cmd = "python {0} {1} {2}".format(script_path, data_file, self.save_path)
        SimpleConda().run_commandlines(MOLMAP_CONDA_ENVIRONMENT, cmd)

    def predict(self, data):
        with open(os.path.join(self.save_path, COLUMNS_MAPPING_FILENAME), "r") as f:
            columns = json.load(f)
        data_file = os.path.join(self.tmp_folder, "data.csv")
        data.to_csv(data_file, index=False)
        script_path = os.path.join(root, "scripts", "predict.py")
        cmd = "python {0} {1} {2}".format(script_path, data_file, self.save_path)
        SimpleConda().run_commandlines(MOLMAP_CONDA_ENVIRONMENT, cmd)
        data = pd.DataFrame({SMILES_COLUMN: data[SMILES_COLUMN]})
        if len(columns["reg"]) > 0:
            reg_pred = np.load(
                open(os.path.join(self.tmp_folder, "reg_preds.npy"), "rb")
            )
            data[columns["reg"][0]] = list(reg_pred)
        if len(columns["clf"]) > 0:
            clf_pred = np.load(
                open(os.path.join(self.tmp_folder, "clf_preds.npy"), "rb")
            )
            data[columns["clf"][0]] = list(clf_pred)
        return data
