from . import BaseDescriptorType
import numpy as np
from rdkit.Avalon import pyAvalonTools
from rdkit import DataStructs


nBits = 1024


class Avalon(BaseDescriptorType):

    def __init__(self):
        super().__init__()
        self.nBits = nBits

    def calc(self, mols):
        fingerprints = []
        fps = [pyAvalonTools.GetAvalonFP(mol, nBits=nBits) for mol in mols]
        for fp in fps:
            fp_np = np.zeros((1, nBits), dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, fp_np)
            fingerprints.append(fp_np)
        return np.array(fingerprints, dtype=np.int32)
