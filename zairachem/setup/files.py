import os
import pandas as pd
import json

from ..vars import DATA_SUBFOLDER
from .schema import InputSchema
from . import COMPOUNDS_FILENAME, ASSAYS_FILENAME, VALUES_FILENAME
from . import (
    COMPOUND_IDENTIFIER_COLUMN,
    ASSAY_IDENTIFIER_COLUMN,
    SMILES_COLUMN,
    DATE_COLUMN,
    QUALIFIER_COLUMN,
    VALUES_COLUMN,
    GROUP_COLUMN,
    ASSAY_TYPE_COLUMN,
    DIRECTION_COLUMN,
    EXPERT_THRESHOLD_COLUMN_PREFIX,
    PERCENTILE_THRESHOLD_COLUMN_PREFIX,
    PARAMETERS_FILE,
)


class ParametersFile(object):
    def __init__(self, path):
        self.filename = os.path.join(path, PARAMETERS_FILE)
        self.params = self.load()

    def load(self):
        with open(self.filename, "r") as f:
            return json.load(f)


class SingleFile(InputSchema):
    def __init__(self, input_file, params):
        InputSchema.__init__(self, input_file)
        self.params = params
        self.df = pd.read_csv(input_file)

    def _make_identifiers(self):
        all_smiles = list(
            set(
                list(self.df[self.df[self.smiles_column].notnull()][self.smiles_column])
            )
        )
        smiles2identifier = {}
        n = len(str(len(all_smiles)))
        for i, smi in enumerate(all_smiles):
            identifier = "CID{0}".format(str(i).zfill(n))
            smiles2identifier[smi] = identifier
        identifiers = []
        for smi in list(self.df[self.smiles_column]):
            if smi in smiles2identifier:
                identifiers += [smiles2identifier[smi]]
            else:
                identifiers += [None]
        return identifiers

    def _impute_qualifiers(self):
        qualifiers = []
        for qual in list(self.df[self.qualifier_column]):
            qual = str(qual).lower()
            if qual == "" or qual == "nan" or qual == "none":
                qualifiers += ["="]
            else:
                qualifiers += [qual]
        return qualifiers

    def normalize_dataframe(self):
        self.identifier_column = self.find_identifier_column()
        self.smiles_column = self.find_smiles_column()
        self.qualifier_column = self.find_qualifier_column()
        self.values_column = self.find_values_column()
        self.date_column = self.find_date_column()
        self.group_column = self.find_group_column()

        if self.identifier_column is None:
            identifiers = self._make_identifiers()
        else:
            identifiers = list(self.df[self.identifier_column])
        df = pd.DataFrame({COMPOUND_IDENTIFIER_COLUMN: identifiers})

        df[SMILES_COLUMN] = self.df[self.smiles_column]

        if self.qualifier_column is None:
            qualifiers = ["="] * self.df.shape[0]
        else:
            qualifiers = self._impute_qualifiers()
        df[QUALIFIER_COLUMN] = qualifiers

        df[VALUES_COLUMN] = self.df[self.values_column]

        if self.date_column:
            df[DATE_COLUMN] = self.df[self.date_column]
        else:
            df[DATE_COLUMN] = None

        if self.group_column:
            df[GROUP_COLUMN] = self.df[self.group_column]
        else:
            df[GROUP_COLUMN] = None

        return df

    def compounds_table(self, df):
        dfc = (
            df[[COMPOUND_IDENTIFIER_COLUMN, SMILES_COLUMN, GROUP_COLUMN]]
            .drop_duplicates(inplace=False)
            .reset_index(drop=True)
        )
        return dfc

    def assays_table(self, df):
        dfa = pd.DataFrame({ASSAY_IDENTIFIER_COLUMN: [self.params["assay_id"]]})
        # TODO: Allowed types: ADME, OTHER, PANEL, AUX_HTS
        assay_type = self.params["assay_type"]
        if assay_type is None:
            assay_type = "OTHER"
        dfa[ASSAY_TYPE_COLUMN] = assay_type
        direction = self.params["direction"]
        dfa[DIRECTION_COLUMN] = direction
        if direction is None:
            raise Exception
        thresholds = self.params["thresholds"]
        for k, v in thresholds.items():
            num = k.split("_")[-1]
            dfa[EXPERT_THRESHOLD_COLUMN_PREFIX + num] = v
        return dfa

    def values_table(self, df):
        dfv = pd.DataFrame({COMPOUND_IDENTIFIER_COLUMN: df[COMPOUND_IDENTIFIER_COLUMN]})
        dfv[ASSAY_IDENTIFIER_COLUMN] = self.params["assay_id"]
        dfv[QUALIFIER_COLUMN] = df[QUALIFIER_COLUMN]
        dfv[VALUES_COLUMN] = df[VALUES_COLUMN]
        return dfv

    def process(self):
        path = os.path.join(self.get_output_dir(), DATA_SUBFOLDER)
        df = self.normalize_dataframe()
        dfc = self.compounds_table(df)
        dfc.to_csv(os.path.join(path, COMPOUNDS_FILENAME), index=False)
        dfa = self.assays_table(df)
        dfa.to_csv(os.path.join(path, ASSAYS_FILENAME), index=False)
        dfv = self.values_table(df)
        dfv.to_csv(os.path.join(path, VALUES_FILENAME), index=False)


# TODO: When three files are given, use the following


class CompoundsFile(InputSchema):
    def __init__(self, input_file):
        InputSchema.__init__(self)

    def process(self):
        pass


class AssaysFile(InputSchema):
    def __init__(self):
        InputSchema.__init__(self)

    def process(self):
        pass


class ValuesFile(InputSchema):
    def __init__(self):
        InputSchema.__init__(self)

    def process(self):
        pass
