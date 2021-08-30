from . import BaseDescriptorType
import numpy as np
from rdkit.Chem import rdMolDescriptors as rd

radius = 3
nBits = 2048


def clip(v):
    if v > 255:
        v = 255
    return v


class Ecfp(BaseDescriptorType):

    def __init__(self):
        super().__init__()
        self.radius = radius
        self.nbits = nBits

    def calc(self, mols):
        fingerprints = []
        for mol in mols:
            counts = list(rd.GetHashedMorganFingerprint(mol, radius=self.radius, nBits=self.nbits))
            counts = [clip(x) for x in counts]
            fingerprints += [counts]
        return np.array(fingerprints)
