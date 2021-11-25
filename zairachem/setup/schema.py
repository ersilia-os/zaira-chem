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
        self.columns = [
            c for c in list(self.df_.columns) if not self.df_[c].isnull().all()
        ]
        print(self.columns)
        self.assigned_columns = set()

    def columns_iter(self):
        for c in columns:
            if c not in self.assigned_columns:
                yield c

    def add_explored_column(self, col):
        self.assigned_columns.update([col])

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
        return cols

    def _is_values_column(self, col):
        try:
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
        except:
            return False

    def find_values_column(self):
        cols = []
        for col in self.columns:
            if self._is_values_column(col):
                cols += [col]
            else:
                continue
        return cols

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
        return cols

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
        cols = []
        for col in self.columns:
            if self._is_date_column(col):
                cols += [col]
        return cols

    def _is_identifier_column(self, col):
        if "identifier" in col.lower():
            return True
        else:
            values = list(self.df_[self.df_[col].notnull()][col])
            for v in values:
                try:
                    float(v)
                    return False
                except:
                    continue
            if len(set(values)) / len(values) > 0.8:
                return True
            else:
                return False

    def find_identifier_column(self):
        cols = []
        for col in self.columns:
            if self._is_identifier_column(col):
                cols += [col]
        return cols

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
        return cols

    def resolve_columns(self):
        smiles_column = self.find_smiles_column()
        assert len(smiles_column) > 0, "No SMILES column found!"
        assert len(smiles_column) == 1, "More than one SMILES column found! {0}".format(
            smiles_column
        )
        smiles_column = smiles_column[0]
        identifier_column = [
            x for x in self.find_identifier_column() if x != smiles_column
        ]
        if len(identifier_column) == 0:
            identifier_column = None
        else:
            identifier_column = identifier_column[0]
        data = {"smiles_column": smiles_column, "identifier_column": identifier_column}
        qualifier_column = self.find_qualifier_column()
        if len(qualifier_column) == 0:
            qualifier_column = None
        else:
            qualifier_column = qualifier_column[0]
        values_column = self.find_values_column()
        assert len(values_column) > 0, "No values column found!"
        assert len(values_column) == 1, "More than one values column found! {0}".format(
            values_column
        )
        values_column = values_column[0]
        group_column = self.find_group_column()
        if len(group_column) == 0:
            group_column = None
        else:
            group_column = group_column[0]
        date_column = self.find_date_column()
        if len(date_column) == 0:
            date_column = None
        else:
            date_column = date_column[0]
        data["qualifier_column"] = qualifier_column
        data["values_column"] = values_column
        data["group_column"] = group_column
        data["date_column"] = date_column
        return data
