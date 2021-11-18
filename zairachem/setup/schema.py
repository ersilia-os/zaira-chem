import os
import pandas as pd


from .. import ZairaBase


_SNIFF_SAMPLE_SIZE = 10000
_MAX_EMPTY = 0.2
_MIN_CORRECT = 0.8


class InputSchema(ZairaBase):
    def __init__(self, input_file):
        ZairaBase.__init__(self)
        self.input_file = os.path.abspath(input_file)
        self.df_ = pd.read_csv(self.input_file, nrows=_SNIFF_SAMPLE_SIZE)
        self.columns = list(self.df_.columns)

    def _prop_correct_smiles(self, col):
        values = list(self.df_[col])
        c = 0
        for v in values:
            try:
                mol = Chem.MolFromSmiles(v)
            except:
                continue
            if mol is not None:
                c += 1
        return float(c) / len(values)

    def _is_smiles_column(self, col):
        if "smiles" in col.lower():
            return True
        else:
            return False
        # TODO
        if self._prop_correct_smiles(col) > _MIN_CORRECT:
            return True
        else:
            return False

    def find_smiles_column(self):
        cols = []
        for col in self.columns:
            if self._is_smiles_column(col):
                cols += [col]
            else:
                continue
        if len(cols) == 0:
            return None
        if len(cols) > 1:
            raise Exception
        else:
            return cols[0]

    def _is_values_column(self, col):
        values = list(self.df_[self.df_[col].notnull()][col])
        c = 0
        for v in values:
            try:
                float(v)
            except:
                continue
            c += 1
        if c == len(values):
            return True
        else:
            return False

    def find_values_column(self):
        cols = []
        for col in self.columns:
            if self._is_values_column(col):
                cols += [col]
            else:
                continue
        if len(cols) == 0:
            return None
        if len(cols) == 1:
            return cols[0]
        if len(cols) > 1:
            raise Exception

    def _is_qualifier_column(self, col):
        # TODO
        if "qualifier" in col.lower():
            return True
        else:
            return False

    def find_qualifier_column(self):
        cols = []
        for col in cols:
            if self._is_qualifier_column(col):
                cols += [col]
        if len(cols) == 0:
            return None
        if len(cols) == 1:
            return cols[0]
        if len(cols) > 1:
            raise Exception

    def _is_date_column(self, col):
        return False  # TODO: Debug
        if "date" in col.lower():
            return True
        else:
            return False
        # TODO
        n = self.df_[self.df_[col].notnull()].shape[0]
        df_ = self.df_.copy()
        df_[col] = pd.to_datetime(df_[col], errors="coerce")
        m = df_[df_[col].notnull()].shape[0]
        if m / n > _MIN_CORRECT:
            return True
        else:
            return False

    def find_date_column(self):
        datecols = []
        for col in self.columns:
            if self._is_date_column(col):
                datecols += [col]
        if len(datecols) > 1:
            raise Exception
        if len(datecols) == 1:
            return datecols[0]
        return None

    def _is_identifier_column(self, col):
        # TODO
        if "identifier" in col.lower():
            return True
        else:
            return False

    def _is_group_column(self, col):
        # TODO
        if "series" in col.lower() or "group" in col.lower():
            return True
        else:
            return False

    def find_group_column(self):
        cols = []
        for col in self.columns:
            if self._is_group_column(col):
                cols += [col]
        if len(cols) > 1:
            raise Exception
        if len(cols) == 0:
            return None
        if len(cols) == 1:
            return cols[0]

    def find_identifier_column(self):
        cols = []
        for col in self.columns:
            if self._is_identifier_column(col):
                cols += [col]
        if len(cols) > 1:
            raise Exception
        if len(cols) == 1:
            return cols[0]
        return None
