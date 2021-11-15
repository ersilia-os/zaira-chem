import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rd

RADIUS = 3
NBITS = 2048
DTYPE = np.int8


def clip_sparse(vect, nbits):
    l = [0] * nbits
    for i, v in vect.GetNonzeroElements().items():
        l[i] = v if v < 255 else 255
    return l


class _Fingerprinter(object):
    def __init__(self):
        self.nbits = NBITS
        self.radius = RADIUS

    def calc(self, mol):
        v = rd.GetHashedMorganFingerprint(mol, radius=self.radius, nBits=self.nbits)
        return clip_sparse(v, self.nbits)


class Fingerprinter(object):
    def __init__(self):
        self.fingerprinter = _Fingerprinter()

    def calculate(self, smiles_list):
        X = np.zeros((len(smiles_list), NBITS), np.uint8)
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            X[i, :] = self.fingerprinter.calc(mol)
        return X
