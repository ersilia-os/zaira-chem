import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rd

radius = 3
nBits = 2048

def clip(v):
    if v > 255:
        v = 255
    return v


class Ecfp(object):
    def __init__(self):
        self.name = "ecfp"
        self.radius = radius
        self.nbits = nBits
    
    def _smi2mol(self, smiles):
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        return mols

    def calc(self, smiles = None, mols = None):
        fingerprints = []
        if mols is None:
            mols = self._smi2mol(smiles)
        for mol in mols:
            counts = list(rd.GetHashedMorganFingerprint(mol, radius=self.radius, nBits=self.nbits))
            counts = [clip(x) for x in counts]
            fingerprints += [counts]
        return np.array(fingerprints, dtype=int)


def calculate_baseline_fingerprints(smiles):
    calculator = Ecfp()
    X = calculator.calc(smiles)
    return X
