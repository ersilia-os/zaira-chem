import pandas as pd

_SNIFF_SAMPLE_SIZE = 10000
_MAX_EMPTY = 0.2
_MIN_CORRECT = 0.8

_INCHIKEY_COLUMN = "inchikey"
_STANDARD_SMILES_COLUMN = "standard_smiles"


class InputSchema(object):

    def __init__(self, input_file):
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

    def _find_smiles_column(self):
        cands = []
        for col in self.columns:
            if self._prop_correct_smiles(col) > _MIN_CORRECT:
                cands += [col]
            else:
                continue
        if len(cands) != 1:
            raise Exception
        else:
            return cands[0]

    def _is_data_column(self, col):
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

    def _find_data_columns(self):
        datacols = []
        for col in self.columns:
            if self._is_data_column(col):
                datacols += [col]
            else:
                continue
        return datacols

    def _is_qualifier_column(self, col):
        pass

    def _is_date_column(self, col):
        n = self.df_[self.df_[col].notnull()].shape[0]
        df_ = self.df_.copy()
        df_[col] = pd.to_datetime(df_[col], errors="coerce")
        m = df_[df_[col].notnull()].shape[0]
        if m/n > _MIN_CORRECT:
            return True
        else:
            return False

    def _find_date_columns(self):
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
        pass

    def _find_identifier_column(self):
        pass

    def schema(self):
        logger.debug("Guessing schema")
        d = {
            "compound_identifier": self._find_identifier_column(),
            "inchikey": _INCHIKEY_COLUMN,
            "smiles": self._find_smiles_column(),
            "standard_smiles": _STANDARD_SMILES_COLUMN,
            "date": self._find_date_columns(),
            "data_columns": self._find_data_columns(),
        }
        logger.debug("Guessed schema {0}".format(d))
        return d
