import os
import json
import pandas as pd
from rdkit import DataStructs
from rdkit import Chem

from . import RAW_INPUT_FILENAME
from . import INPUT_SCHEMA_FILENAME
from . import SCHEMA_MERGE_FILENAME
from . import MAPPING_FILENAME

from ..vars import DATA_SUBFOLDER
from ..vars import DATA_FILENAME


class SetupChecker(object):
    def __init__(self, path):
        for f in os.listdir(path):
            if RAW_INPUT_FILENAME in f:
                self.input_file = os.path.join(path, f)
        self.input_schema = os.path.join(path, DATA_SUBFOLDER, INPUT_SCHEMA_FILENAME)
        self.data_schema = os.path.join(path, DATA_SUBFOLDER, SCHEMA_MERGE_FILENAME)
        self.data_file = os.path.join(path, DATA_SUBFOLDER, DATA_FILENAME)
        self.mapping_file = os.path.join(path, DATA_SUBFOLDER, MAPPING_FILENAME)

    def check_smiles(self):
        with open(self.data_schema, "r") as f:
            data_schema = json.load(f)
        with open(self.input_schema, "r") as f:
            input_schema = json.load(f)
        input_smiles_column = input_schema["smiles_column"]
        data_smiles_column = data_schema["compounds"][1]
        di = pd.read_csv(self.input_file)
        dd = pd.read_csv(self.data_file)
        ismi = list(di[input_smiles_column])
        dsmi = list(dd[data_smiles_column])
        mapping = pd.read_csv(self.mapping_file)
        for oidx, uidx, cid in mapping.values:
            if str(oidx) == "nan" or str(uidx) == "nan":
                continue
            oidx = int(oidx)
            uidx = int(uidx)
            ofp = Chem.RDKFingerprint(Chem.MolFromSmiles(ismi[oidx]))
            ufp = Chem.RDKFingerprint(Chem.MolFromSmiles(dsmi[uidx]))
            sim = DataStructs.FingerprintSimilarity(ofp, ufp)
            if sim < 0.9:
                print(sim, cid, ismi[oidx], dsmi[uidx])

    def check_activity(self):
        with open(self.data_schema, "r") as f:
            data_schema = json.load(f)
        with open(self.input_schema, "r") as f:
            input_schema = json.load(f)
        input_values_column = input_schema["values_column"]
        if "reg_raw_skip" in data_schema["tasks"]:
            data_values_column = "reg_raw_skip"
        else:
            data_values_column = "clf_aux"
        di = pd.read_csv(self.input_file)
        dd = pd.read_csv(self.data_file)
        ival = list(di[input_values_column])
        dval = list(dd[data_values_column])
        mapping = pd.read_csv(self.mapping_file)
        for oidx, uidx, cid in mapping.values:
            if str(oidx) == "nan" or str(uidx) == "nan":
                continue
            oidx = int(oidx)
            uidx = int(uidx)
            difference = ival[oidx] - dval[uidx]
            if difference > 0.01:
                print(difference, cid, oidx, uidx)

    def run(self):
        self.check_smiles()
        self.check_activity()
