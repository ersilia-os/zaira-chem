import numpy as np
from rdkit import Chem

from mordred import Calculator, descriptors


IGNORE_3D = False


class Mordred(object):
    def __init__(self):
        self.name = "mordred"
        self.is_scaled = False
        self.has_missing = True

    def calc(self, mols):
        calc = Calculator(descriptors, ignore_3D=IGNORE_3D)
        df = calc.pandas(mols)
        return np.array(df, dtype=np.float)
