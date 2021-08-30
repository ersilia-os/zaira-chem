from . import BaseDescriptorType
import numpy as np
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs


class Maccs(BaseDescriptorType):

    def __init__(self):
        super().__init__()

    def _calc(self, mols):
        fingerprints = []
        fps = [MACCSkeys.GenMACCSKeys(mol) for mol in mols]
        for fp in fps:
            fp_np = np.zeros((1,), dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, fp_np)
            fingerprints.append(fp_np)
        return np.array(fingerprints, dtype=np.int8)
