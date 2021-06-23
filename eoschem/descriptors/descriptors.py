import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Avalon import pyAvalonTools

try:
    from signaturizer import Signaturizer
except:
    Signaturizer = None


class Descriptors(object):

    def __init__(self, data):
        self._data = data

    @staticmethod
    def smiles_to_mols(query_smiles):
        mols = [Chem.MolFromSmiles(smile) for smile in query_smiles]
        valid = [0 if mol is None else 1 for mol in mols]
        valid_idxs = [idx for idx, boolean in enumerate(valid) if boolean == 1]
        valid_mols = [mols[idx] for idx in valid_idxs]
        return valid_mols, valid_idxs

    def ECFP_counts(self, radius=3, useFeatures=True, useCounts=True):
        mols, valid_idx = self.smiles_to_mols(self._data)
        fps = [AllChem.GetMorganFingerprint(mol, radius, useCounts=useCounts, useFeatures=useFeatures) for mol in mols]
        size = 2048
        nfp = np.zeros((len(fps), size), np.int32)
        for i, fp in enumerate(fps):
            for idx, v in fp.GetNonzeroElements().items():
                nidx = idx % size
                nfp[i, nidx] += int(v)
        return nfp, valid_idx

    def Avalon(self, nBits=1024):
        mols, valid_idx = self.smiles_to_mols(self._data)
        fingerprints = []
        fps = [pyAvalonTools.GetAvalonFP(mol, nBits=nBits) for mol in mols]
        for fp in fps:
            fp_np = np.zeros((1, nBits), dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, fp_np)
            fingerprints.append(fp_np)
        return fingerprints, valid_idx

    def MACCS_keys(self):
        mols, valid_idx = self.smiles_to_mols(self._data)
        fingerprints = []
        fps = [MACCSkeys.GenMACCSKeys(mol) for mol in mols]
        for fp in fps:
            fp_np = np.zeros((1, ), dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, fp_np)
            fingerprints.append(fp_np)
        return fingerprints, valid_idx

    def ChemicalChecker(self):
        if Signaturizer:
            sign = Signaturizer("GLOBAL")
            mols, valid_idx = self.smiles_to_mols(self._data)
            smiles = [Chem.MolToSmiles(mol) for mol in mols]
            fps = sign.predict(smiles)
            return fps, valid_idx


class DescriptorsCalculator(object):

    def __init__(self, data):
        self.descriptors = Descriptors(data)
        self.data = data
        self._funcs = self._functions()

    def _functions(self):
        functions = {
            "ecfp_counts": self.descriptors.ECFP_counts,
            "avalon": self.descriptors.Avalon,
            "maccs": self.descriptors.MACCS_keys
        }
        return functions

    def _calculate(self, n):
        func = self._funcs[n]
        res, idxs = func()
        X = np.zeros((len(self.data), np.array(res).shape[1])) # assume zeros for invalid smiles
        X[np.array(idxs)] = res
        return X

    def calculate_one(self, n):
        return self._calculate(n)

    def calculate(self):
        names = sorted([k for k, _ in self._funcs.items()])
        for n in names:
            X = self._calculate(n)
            yield X, n
