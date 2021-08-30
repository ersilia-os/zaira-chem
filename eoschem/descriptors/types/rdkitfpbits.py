from . import BaseDescriptorType
import numpy as np
from rdkit import Chem


_MIN_PATH_LEN = 1
_MAX_PATH_LEN = 7
_N_BITS = 2048


<<<<<<< HEAD
class RdkitFpBits(object):
=======
<<<<<<< HEAD
class RdkitFpBits(BaseDescriptorType):

=======
class RdkitFpBits(object):
>>>>>>> f7356c4... Major updates
>>>>>>> 8a05dcf7bcbfbee32aa23019194890e2b9ee485e
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
<<<<<<< HEAD
                mol, minPath=self.minPathLen, maxPath=self.maxPathLen, fpSize=self.nbits
            )
=======
<<<<<<< HEAD
                mol, minPath=self.minPathLen, maxPath=self.maxPathLen,
                fpSize=self.nbits)
=======
                mol, minPath=self.minPathLen, maxPath=self.maxPathLen, fpSize=self.nbits
            )
>>>>>>> f7356c4... Major updates
>>>>>>> 8a05dcf7bcbfbee32aa23019194890e2b9ee485e
            fingerprints += [[self._clip(c) for c in counts]]
        return np.array(fingerprints)
