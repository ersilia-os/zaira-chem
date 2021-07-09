from . import BaseDescriptorType
import numpy as np
from rdkit.Chem import AllChem

radius = 2048
useCounts = True
useFeatures = True


class Ecfp(BaseDescriptorType):

    def __init__(self):
        super().__init__()
        self.radius = radius
        self.useCounts = useCounts
        self.useFeatures = useFeatures

    def _calc(self, mols):
        fps = [
            AllChem.GetMorganFingerprint(
                mol, self.radius, useCounts=self.useCounts,
                useFeatures=self.useFeatures)
            for mol in mols
        ]
        size = 2048
        nfp = np.zeros((len(fps), size), np.int32)
        for i, fp in enumerate(fps):
            for idx, v in fp.GetNonzeroElements().items():
                nidx = idx % size
                nfp[i, nidx] += int(v)
        return np.array(nfp, dtype=np.int32)
