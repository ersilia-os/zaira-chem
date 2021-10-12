import os
import pandas as pd
import json

from . import MELLODDY_SUBFOLDER
from . import T0_FILE, T1_FILE, T2_FILE, DEFAULT_PARAMS_FILE

from ...setup import (
    ASSAYS_FILENAME,
    COMPOUNDS_FILENAME,
    VALUES_FILENAME,
    VALUES_COLUMN,
    QUALIFIER_COLUMN,
    EXPERT_THRESHOLD_COLUMN_PREFIX,
    ASSAY_IDENTIFIER_COLUMN,
    ASSAY_TYPE_COLUMN,
    DIRECTION_COLUMN,
    COMPOUND_IDENTIFIER_COLUMN,
    SMILES_COLUMN,
    PARAMETERS_FILE
)


class Prepare(object):
    def __init__(self, path):
        self.indir = os.path.abspath(path)
        self.outdir = os.path.abspath(os.path.join(path, MELLODDY_SUBFOLDER))
        os.makedirs(self.outdir, exist_ok=True)

    def t0(self):
        dfi = pd.read_csv(os.path.join(self.indir, ASSAYS_FILENAME))
        expert_threshold_columns = [EXPERT_THRESHOLD_COLUMN_PREFIX + x for x in "12345"]
        dfo = dfi[
            [ASSAY_IDENTIFIER_COLUMN, ASSAY_TYPE_COLUMN, DIRECTION_COLUMN]
            + expert_threshold_columns
        ]
        dfo = dfo.rename(columns={ASSAY_IDENTIFIER_COLUMN: "input_assay_id"})
        dfo["use_in_regression"] = True
        dfo = dfo[
            [
                "input_assay_id",
                "assay_type",
                "use_in_regression",
                "expert_threshold_1",
                "expert_threshold_2",
                "expert_threshold_3",
                "expert_threshold_4",
                "expert_threshold_5",
            ]
        ]
        dfo.to_csv(os.path.join(self.outdir, T0_FILE), index=False)

    def t1(self):
        dfi = pd.read_csv(os.path.join(self.indir, VALUES_FILENAME))
        dfi = dfi.rename(
            columns={
                COMPOUND_IDENTIFIER_COLUMN: "input_compound_id",
                ASSAY_IDENTIFIER_COLUMN: "input_assay_id",
                QUALIFIER_COLUMN: "standard_qualifier",
                VALUES_COLUMN: "standard_value",
            }
        )
        dfo = dfi[
            [
                "input_compound_id",
                "input_assay_id",
                "standard_qualifier",
                "standard_value",
            ]
        ]
        dfo.to_csv(os.path.join(self.outdir, T1_FILE), index=False)

    def t2(self):
        dfi = pd.read_csv(os.path.join(self.indir, COMPOUNDS_FILENAME))
        dfi = dfi.rename(
            columns={
                COMPOUND_IDENTIFIER_COLUMN: "input_compound_id",
                SMILES_COLUMN: "smiles",
            }
        )
        dfo = dfi[["input_compound_id", "smiles"]]
        dfo.to_csv(os.path.join(self.outdir, T2_FILE), index=False)

    def params(self):
        with open(os.path.join(self.indir, PARAMETERS_FILE), "r") as f:
            uparam = json.load(f)
        with open(DEFAULT_PARAMS_FILE, "r") as f:
            dparam = json.load(f)
        
    def run(self):
        self.t0()
        self.t1()
        self.t2()
        self.params()
