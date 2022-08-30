import os
import json
import pandas as pd
import csv
from rdkit import DataStructs
from rdkit import Chem
from standardiser import standardise

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

    def _get_schemas(self):
        with open(self.data_schema, "r") as f:
            self.data_schema_dict = json.load(f)
        with open(self.input_schema, "r") as f:
            self.input_schema_dict = json.load(f)

    def remap(self):
        self._get_schemas()
        dm = pd.read_csv(self.mapping_file)
        dd = pd.read_csv(self.data_file)
        data_schema = self.data_schema_dict
        cid_mapping_column = "compound_id"
        cid_mapping = list(dm[cid_mapping_column])
        cid_data_column = data_schema["compounds"][0]
        cid_data = list(dd[cid_data_column])
        cid_data_idx = {}
        for i, cid in enumerate(cid_data):
            cid_data_idx[cid] = i
        new_idxs = []
        for cid in cid_mapping:
            if cid not in cid_data_idx:
                new_idxs += [""]
            else:
                new_idxs += [cid_data_idx[cid]]
        orig_idxs = list(dm["orig_idx"])
        with open(self.mapping_file, "w") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(["orig_idx", "uniq_idx", "compound_id"])
            for o, u, c in zip(orig_idxs, new_idxs, cid_mapping):
                writer.writerow([o, u, c])

    def check_smiles(self):
        self._get_schemas()
        input_schema = self.input_schema_dict
        data_schema = self.data_schema_dict
        input_smiles_column = input_schema["smiles_column"]
        data_smiles_column = data_schema["compounds"][1]
        di = pd.read_csv(self.input_file)
        dd = pd.read_csv(self.data_file)
        ismi = list(di[input_smiles_column])
        dsmi = list(dd[data_smiles_column])
        mapping = pd.read_csv(self.mapping_file)
        discrepancies = 0
        for oidx, uidx, cid in mapping.values:
            if str(oidx) == "nan" or str(uidx) == "nan":
                continue
            oidx = int(oidx)
            uidx = int(uidx)
            omol = Chem.MolFromSmiles(ismi[oidx])
            umol = Chem.MolFromSmiles(dsmi[uidx])
            ofp = Chem.RDKFingerprint(omol)
            ufp = Chem.RDKFingerprint(umol)
            sim = DataStructs.FingerprintSimilarity(ofp, ufp)
            if sim < 0.6:
                try:
                    omol = standardise.run(omol)
                except:
                    continue
                ofp = Chem.RDKFingerprint(omol)
                sim = DataStructs.FingerprintSimilarity(ofp, ufp)
                if sim < 0.6:
                    print("Low similarity", sim, cid, ismi[oidx], dsmi[uidx])
                    discrepancies += 1
        assert discrepancies < mapping.shape[0] * 0.25

    def check_activity(self):
        with open(self.data_schema, "r") as f:
            data_schema = json.load(f)
        with open(self.input_schema, "r") as f:
            input_schema = json.load(f)
        input_values_column = input_schema["values_column"]
        if "reg_raw_skip" in data_schema["tasks"]:
            data_values_column = "reg_raw_skip"
        else:
            if "clf_aux" in data_schema["tasks"]:
                data_values_column = "clf_aux"
            else:
                if input_values_column in data_schema["tasks"]:
                    data_values_column = input_values_column
                else:
                    data_values_column = data_schema["tasks"][0]
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
                print("High activity difference", difference, cid, oidx, uidx)

    def run(self):
        self.remap()
        self.check_smiles()
        self.check_activity()
