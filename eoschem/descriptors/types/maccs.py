import numpy as np
from rdkit.Chem import MACCSkeys


class Maccs(object):

    def __init__(self):
        self.name = "maccs"

    def calc(self, mols):
        fingerprints = []
        fps = [MACCSkeys.GenMACCSKeys(mol) for mol in mols]
        for fp in fps:
            fp_np = np.zeros((1, ), dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, fp_np)
            fingerprints.append(fp_np)
        return np.array(fingerprints, dtype=np.int8)
