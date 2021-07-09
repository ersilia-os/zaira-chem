from . import BaseDescriptorType
import numpy as np

from mordred import Calculator, descriptors


IGNORE_3D = False


class Mordred(BaseDescriptorType):

    def __init__(self):
        super().__init__()
        self.is_scaled = False
        self.has_missing = True

    def _calc(self, mols):
        calc = Calculator(descriptors, ignore_3D=IGNORE_3D)
        df = calc.pandas(mols)
        return np.array(df, dtype=np.float)
