from . import BaseDescriptorType
import numpy as np
from rdkit import Chem


_MIN_PATH_LEN = 1
_MAX_PATH_LEN = 7
_N_BITS = 2048


class RdkitFpBits(BaseDescriptorType):

    def __init__(self):
        super().__init__()
        self.minPathLen = _MIN_PATH_LEN
        self.maxPathLen = _MAX_PATH_LEN
        self.nbits = _N_BITS

    def _clip(self, v):
        if v > 255:
            v = 255
        return v

    def _calc(self, mols):
        fingerprints = []
        for mol in mols:
            counts = Chem.RDKFingerprint(
                mol, minPath=self.minPathLen, maxPath=self.maxPathLen,
                fpSize=self.nbits)
            fingerprints += [[self._clip(c) for c in counts]]
        return np.array(fingerprints)
