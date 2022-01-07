import os
import numpy as np
import pandas as pd
from ..tools.melloddy import MELLODDY_SUBFOLDER, TAG
from . import (
    COMPOUNDS_FILENAME,
    MAPPING_DEDUPE_COLUMN,
    MAPPING_FILENAME,
    VALUES_FILENAME,
    COMPOUND_IDENTIFIER_COLUMN,
    SMILES_COLUMN,
)


class Standardize(object):
    def __init__(self, path):
        self.path = path
        self.tuner_filename = os.path.join(
            self.path,
            MELLODDY_SUBFOLDER,
            TAG,
            "results_tmp",
            "standardization",
            "T2_standardized.csv",
        )

    def run(self):
        dfm = pd.read_csv(self.tuner_filename)[
            ["input_compound_id", "canonical_smiles"]
        ]
        dfc = pd.read_csv(os.path.join(self.path, COMPOUNDS_FILENAME))
        mapping = pd.read_csv(os.path.join(self.path, MAPPING_FILENAME))
        dfc = dfc[
            dfc[COMPOUND_IDENTIFIER_COLUMN].isin(dfm["input_compound_id"])
        ].reset_index(drop=True)
        std_smiles_dict = {}
        for v in dfm.values:
            std_smiles_dict[v[0]] = v[1]
        std_smiles = []
        for cid in list(dfc[COMPOUND_IDENTIFIER_COLUMN]):
            std_smiles += [std_smiles_dict[cid]]
        dfc[SMILES_COLUMN] = std_smiles
        cid2idx = {}
        for i, cid in enumerate(list(dfc[COMPOUND_IDENTIFIER_COLUMN])):
            cid2idx[cid] = i
        new_mapping = []
        for cid in list(mapping[COMPOUND_IDENTIFIER_COLUMN]):
            if cid in cid2idx:
                new_mapping += [cid2idx[cid]]
            else:
                new_mapping += [None]
        mapping.loc[:, [MAPPING_DEDUPE_COLUMN]] = new_mapping
        dfc.to_csv(os.path.join(self.path, COMPOUNDS_FILENAME), index=False)
        mapping.to_csv(os.path.join(self.path, MAPPING_FILENAME), index=False)
        values_file = os.path.join(self.path, VALUES_FILENAME)
        if os.path.exists(values_file):
            dfv = pd.read_csv(values_file)
            dfv = dfv[
                dfv[COMPOUND_IDENTIFIER_COLUMN].isin(dfc[COMPOUND_IDENTIFIER_COLUMN])
            ].reset_index(drop=True)
            dfv.to_csv(os.path.join(self.path, VALUES_FILENAME), index=False)
